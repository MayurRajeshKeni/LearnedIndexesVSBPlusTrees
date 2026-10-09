"""
Recursive Model Index (RMI) - 2-Stage Implementation

Based on Kraska et al. 2018 ("The Case for Learned Index Structures").

How RMI Works (Plain English):
-------------------------------
1. Instead of a balanced tree of pointers, RMI uses a hierarchy of fast mathematical models.
2. Stage 1 (Root Model): A single linear regression mapping key -> [0, M-1] that routes
   the query to one of M specialized stage-2 models.
3. Stage 2 (Leaf Models): Each of the M models is trained on the subset of keys routed to it.
   It predicts the key's position in the sorted array: pred = w * key + b.
4. Error Bounds: During training, each stage-2 model records its minimum and maximum error:
   min_err = min(actual - pred), max_err = max(actual - pred).
5. Fast Lookup: For any lookup key, we calculate its predicted position, then perform a narrow
   binary search strictly within [pred + min_err, pred + max_err]. Because models capture the
   data trend, this window is vastly smaller than searching the full array.
6. Write Penalty (Delta Buffer): Because learned models are rigid, new inserts are staged in a
   fast sorted delta buffer. When the buffer grows beyond a threshold (default 1% of N), the
   index merges the buffer and retrains the entire model hierarchy.
"""

import time
import bisect
from typing import Optional, Tuple
import numpy as np
from src.base import BaseIndex


class RMIIndex(BaseIndex):
    def __init__(self, m: int = 1000, buffer_ratio: float = 0.01):
        """
        :param m: Number of Stage-2 leaf models (default 1000).
        :param buffer_ratio: Delta buffer capacity as a fraction of dataset size N (default 1%).
        """
        self._m = max(1, int(m))
        self._buffer_ratio = buffer_ratio
        self._name = f"RMI (M={self._m})"

        # Main storage
        self._keys: np.ndarray = np.empty(0, dtype=np.int64)
        self._values: np.ndarray = np.empty(0, dtype=np.int64)
        self._n = 0

        # Stage 1 model: key -> model_index
        self._w1: float = 0.0
        self._b1: float = 0.0

        # Stage 2 models: vectorized arrays for M models
        self._w2 = np.zeros(self._m, dtype=np.float64)
        self._b2 = np.zeros(self._m, dtype=np.float64)
        self._min_err = np.zeros(self._m, dtype=np.int64)
        self._max_err = np.zeros(self._m, dtype=np.int64)
        self._model_empty = np.ones(self._m, dtype=bool)

        # Delta buffer for dynamic inserts: maintains sorted keys and values
        self._delta_keys: list[int] = []
        self._delta_values: list[int] = []
        self._retrain_threshold: int = 100

        # Retrain tracking statistics
        self.retrain_count: int = 0
        self.total_retrain_time_sec: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    def build(self, sorted_keys: np.ndarray, values: Optional[np.ndarray] = None) -> None:
        """
        Bulk build the 2-stage RMI from sorted unique keys.
        Vectorized with NumPy for high efficiency.
        """
        self._keys = np.asarray(sorted_keys, dtype=np.int64)
        self._n = len(self._keys)
        if values is None:
            self._values = np.arange(self._n, dtype=np.int64)
        else:
            self._values = np.asarray(values, dtype=np.int64)

        # Reset delta buffer
        self._delta_keys.clear()
        self._delta_values.clear()
        self._retrain_threshold = max(10, int(self._n * self._buffer_ratio))

        if self._n == 0:
            return

        # Train Stage 1 and Stage 2 models
        self._train_models()

    def _train_models(self) -> None:
        """
        Vectorized training for Stage 1 and Stage 2 linear models.
        """
        if self._n <= 1:
            self._w1 = 0.0
            self._b1 = 0.0
            self._w2.fill(0.0)
            self._b2.fill(0.0)
            self._min_err.fill(0)
            self._max_err.fill(0)
            self._model_empty.fill(True)
            if self._n == 1:
                self._model_empty[0] = False
            return

        k_min = float(self._keys[0])
        k_max = float(self._keys[-1])

        # Stage 1: map key range [k_min, k_max] -> [0, M - 1]
        if k_max > k_min:
            self._w1 = float(self._m - 1) / (k_max - k_min)
            self._b1 = -self._w1 * k_min
        else:
            self._w1 = 0.0
            self._b1 = 0.0

        # Route all keys through Stage 1 to find model assignments
        # stage1_preds: model index for each key
        raw_m = np.floor(self._keys * self._w1 + self._b1).astype(np.int64)
        model_assignments = np.clip(raw_m, 0, self._m - 1)

        # Find boundaries for each stage-2 model in O(M log N)
        # Since keys are sorted and w1 >= 0, model_assignments is monotonically non-decreasing
        splits = np.searchsorted(model_assignments, np.arange(self._m + 1))

        # Reset stage 2 structures
        self._w2.fill(0.0)
        self._b2.fill(0.0)
        self._min_err.fill(0)
        self._max_err.fill(0)
        self._model_empty.fill(True)

        last_valid_model = -1

        for m_idx in range(self._m):
            start = splits[m_idx]
            end = splits[m_idx + 1]
            count = end - start

            if count == 0:
                # Empty model: will be handled via neighboring model bounds
                self._model_empty[m_idx] = True
                continue

            self._model_empty[m_idx] = False
            last_valid_model = m_idx

            k_slice = self._keys[start:end]
            y_slice = np.arange(start, end, dtype=np.float64)

            if count == 1:
                self._w2[m_idx] = 0.0
                self._b2[m_idx] = float(start)
                self._min_err[m_idx] = 0
                self._max_err[m_idx] = 0
            else:
                k_span = float(k_slice[-1] - k_slice[0])
                if k_span > 0:
                    # Direct endpoint slope or linear fit
                    w = (float(count - 1)) / k_span
                    b = float(start) - w * float(k_slice[0])
                else:
                    w = 0.0
                    b = float(start)

                self._w2[m_idx] = w
                self._b2[m_idx] = b

                # Compute prediction errors on all keys in this model
                preds = np.floor(k_slice * w + b).astype(np.int64)
                actuals = np.arange(start, end, dtype=np.int64)
                errors = actuals - preds

                self._min_err[m_idx] = int(np.min(errors))
                self._max_err[m_idx] = int(np.max(errors))

        # Fill empty models with nearest neighbor parameters to handle edge routing
        if last_valid_model != -1:
            curr_w = self._w2[last_valid_model]
            curr_b = self._b2[last_valid_model]
            curr_min = self._min_err[last_valid_model]
            curr_max = self._max_err[last_valid_model]
            for m_idx in range(self._m - 1, -1, -1):
                if not self._model_empty[m_idx]:
                    curr_w = self._w2[m_idx]
                    curr_b = self._b2[m_idx]
                    curr_min = self._min_err[m_idx]
                    curr_max = self._max_err[m_idx]
                else:
                    self._w2[m_idx] = curr_w
                    self._b2[m_idx] = curr_b
                    self._min_err[m_idx] = curr_min
                    self._max_err[m_idx] = curr_max

    def lookup(self, key: int) -> Optional[int]:
        """
        Learned lookup:
        1. Predict position in main array using 2-stage models.
        2. Binary search within [pred + min_err, pred + max_err].
        3. Check delta buffer for updates/new insertions.
        """
        key = int(key)

        # 1. Check main array
        found_in_main = False
        val_in_main = None

        if self._n > 0:
            if self._n == 1:
                if self._keys[0] == key:
                    found_in_main = True
                    val_in_main = int(self._values[0])
            else:
                # Stage 1 routing
                raw_m = int(np.floor(key * self._w1 + self._b1))
                m_idx = max(0, min(self._m - 1, raw_m))

                # Stage 2 prediction
                pred = int(np.floor(key * self._w2[m_idx] + self._b2[m_idx]))
                low = max(0, pred + self._min_err[m_idx])
                high = min(self._n - 1, pred + self._max_err[m_idx])

                if low <= high:
                    # Binary search only within bounded window
                    idx = int(np.searchsorted(self._keys[low:high + 1], key)) + low
                    if idx < self._n and self._keys[idx] == key:
                        found_in_main = True
                        val_in_main = int(self._values[idx])

        # 2. Check delta buffer (buffer takes precedence for updates)
        if self._delta_keys:
            b_idx = bisect.bisect_left(self._delta_keys, key)
            if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
                return self._delta_values[b_idx]

        if found_in_main:
            return val_in_main

        return None

    def lookup_trace(self, key: int) -> dict:
        """
        Learned lookup with complete internal routing and search trace.
        Guaranteed to return identical result to lookup(key).
        """
        key = int(key)

        trace = {
            "key": key,
            "stage1_prediction": 0.0,
            "chosen_stage2_model_id": 0,
            "predicted_position": 0,
            "error_min": 0,
            "error_max": 0,
            "search_window": (0, 0),
            "binary_search_steps": [],
            "found": False,
            "final_position": None,
            "value": None,
            "found_in_delta": False,
        }

        if self._n == 0:
            if self._delta_keys:
                b_idx = bisect.bisect_left(self._delta_keys, key)
                if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
                    trace["found"] = True
                    trace["value"] = self._delta_values[b_idx]
                    trace["found_in_delta"] = True
            return trace

        if self._n == 1:
            trace["predicted_position"] = 0
            trace["search_window"] = (0, 0)
            trace["binary_search_steps"] = [0]
            if self._keys[0] == key:
                trace["found"] = True
                trace["final_position"] = 0
                trace["value"] = int(self._values[0])
            # Check delta
            if self._delta_keys:
                b_idx = bisect.bisect_left(self._delta_keys, key)
                if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
                    trace["found"] = True
                    trace["value"] = self._delta_values[b_idx]
                    trace["found_in_delta"] = True
            return trace

        # Stage 1 Routing
        raw_m = float(key * self._w1 + self._b1)
        m_idx = int(np.clip(int(np.floor(raw_m)), 0, self._m - 1))
        trace["stage1_prediction"] = raw_m
        trace["chosen_stage2_model_id"] = m_idx

        # Stage 2 Prediction
        pred = int(np.floor(key * self._w2[m_idx] + self._b2[m_idx]))
        min_err = int(self._min_err[m_idx])
        max_err = int(self._max_err[m_idx])
        low = max(0, pred + min_err)
        high = min(self._n - 1, pred + max_err)

        trace["predicted_position"] = pred
        trace["error_min"] = min_err
        trace["error_max"] = max_err
        trace["search_window"] = (low, high)

        # Trace exact binary search steps
        steps = []
        l = low
        r = high
        found_in_main = False
        final_pos = None

        while l <= r:
            mid = (l + r) // 2
            steps.append(int(mid))
            mid_val = self._keys[mid]
            if mid_val == key:
                found_in_main = True
                final_pos = int(mid)
                break
            elif mid_val < key:
                l = mid + 1
            else:
                r = mid - 1

        trace["binary_search_steps"] = steps

        if found_in_main:
            trace["found"] = True
            trace["final_position"] = final_pos
            trace["value"] = int(self._values[final_pos])

        # Check delta buffer (takes precedence for updates)
        if self._delta_keys:
            b_idx = bisect.bisect_left(self._delta_keys, key)
            if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
                trace["found"] = True
                trace["value"] = self._delta_values[b_idx]
                trace["found_in_delta"] = True

        return trace

    def insert(self, key: int, value: int) -> None:
        """
        Insert (key, value) into delta buffer.
        If buffer exceeds threshold, merge and retrain the index.
        """
        key = int(key)
        value = int(value)

        # Insert into sorted delta buffer
        b_idx = bisect.bisect_left(self._delta_keys, key)
        if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
            self._delta_values[b_idx] = value
        else:
            self._delta_keys.insert(b_idx, key)
            self._delta_values.insert(b_idx, value)

        # Check if retraining is needed
        if len(self._delta_keys) >= self._retrain_threshold:
            self._merge_and_retrain()

    def _merge_and_retrain(self) -> None:
        """
        Merges delta buffer into main array and retrains models.
        Tracks the write penalty (retrain time and count).
        """
        t0 = time.perf_counter()
        self.retrain_count += 1

        if not self._delta_keys:
            return

        # Combine main and delta entries
        d_keys = np.asarray(self._delta_keys, dtype=np.int64)
        d_vals = np.asarray(self._delta_values, dtype=np.int64)

        if self._n == 0:
            merged_keys = d_keys
            merged_vals = d_vals
        else:
            # Merge two sorted sequences
            all_keys = np.concatenate([self._keys, d_keys])
            all_vals = np.concatenate([self._values, d_vals])
            
            # Sort and deduplicate (keeping latest delta value)
            sort_order = np.argsort(all_keys, kind="stable")
            sorted_all_keys = all_keys[sort_order]
            sorted_all_vals = all_vals[sort_order]

            # In case of duplicates, retain the last occurrence (the inserted one)
            unique_mask = np.empty(len(sorted_all_keys), dtype=bool)
            unique_mask[:-1] = sorted_all_keys[:-1] != sorted_all_keys[1:]
            unique_mask[-1] = True

            merged_keys = sorted_all_keys[unique_mask]
            merged_vals = sorted_all_vals[unique_mask]

        self._keys = merged_keys
        self._values = merged_vals
        self._n = len(self._keys)

        self._delta_keys.clear()
        self._delta_values.clear()
        self._retrain_threshold = max(10, int(self._n * self._buffer_ratio))

        # Re-fit models
        self._train_models()

        self.total_retrain_time_sec += (time.perf_counter() - t0)

    def memory_bytes(self) -> int:
        """
        Honest analytical estimate of index overhead:
        - Stage 1 parameters (w1, b1: 16 bytes)
        - Stage 2 arrays:
            w2 (M * 8 bytes)
            b2 (M * 8 bytes)
            min_err (M * 8 bytes)
            max_err (M * 8 bytes)
            model_empty (M * 1 byte)
        - Delta buffer overhead: keys list + values list (~16 bytes per entry)
        Excludes the base key-value arrays to measure pure index overhead.
        """
        stage1_bytes = 16
        stage2_bytes = (
            self._w2.nbytes +
            self._b2.nbytes +
            self._min_err.nbytes +
            self._max_err.nbytes +
            self._model_empty.nbytes
        )
        delta_bytes = (len(self._delta_keys) + len(self._delta_values)) * 8
        return stage1_bytes + stage2_bytes + delta_bytes

    def get_error_stats(self) -> Tuple[float, int, float]:
        """
        Returns (mean_error, max_error, mean_window_size) across all trained keys.
        """
        if self._n <= 1:
            return 0.0, 0, 1.0

        raw_m = np.clip(np.floor(self._keys * self._w1 + self._b1).astype(np.int64), 0, self._m - 1)
        preds = np.floor(self._keys * self._w2[raw_m] + self._b2[raw_m]).astype(np.int64)
        actuals = np.arange(self._n, dtype=np.int64)
        errors = np.abs(actuals - preds)

        windows = (self._max_err[raw_m] - self._min_err[raw_m] + 1)

        return float(np.mean(errors)), int(np.max(errors)), float(np.mean(windows))

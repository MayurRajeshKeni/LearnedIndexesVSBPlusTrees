"""
Piecewise Geometric Model Index (PGM-Index) - Simplified Single-Level Implementation

Based on Ferragina & Vinciguerra 2020 ("The PGM-index: a compact and efficient learned index
with provable worst-case bounds").

How PGM Works (Plain English):
------------------------------
1. Unlike RMI which fixes the number of models M upfront, PGM constructs piecewise linear
   segments adaptively while guaranteeing that EVERY key's actual position is within ±ε
   of the predicted position.
2. The Greedy "Shrinking Cone" Segmentation (O(n)):
   - Start a segment at key (k0, pos0).
   - Any line through (k0, pos0) has slope s: pred(k) = pos0 + s * (k - k0).
   - For each subsequent key (ki, posi), staying within ±ε requires:
       (posi - pos0 - ε) / (ki - k0) <= s <= (posi - pos0 + ε) / (ki - k0)
   - As we inspect more keys, the allowable slope range [s_low, s_high] continuously shrinks
     (like a narrowing cone of light).
   - When a key causes s_low > s_high, no single line can cover the segment within ±ε.
     We finalize the segment's slope and start a fresh segment at this key.
3. Fast Bounded Lookup:
   - Segments' first keys are sorted. We locate the responsible segment with binary search
     (O(log(num_segments))), compute the prediction, and then search ONLY the guaranteed
     window [pred - ε, pred + ε] (at most 2ε keys).
4. Inserts:
   - Handled via a sorted delta buffer, triggering a merge-and-rebuild when the buffer exceeds
     the capacity threshold (1% of N), mirroring the RMI design for fair comparison.
5. Note:
   - This implementation is a single-level PGM index, not the full recursive multi-level PGM tree.
"""

import time
import bisect
from typing import Optional, Tuple
import numpy as np
from src.base import BaseIndex


class PGMIndex(BaseIndex):
    def __init__(self, epsilon: int = 64, buffer_ratio: float = 0.01):
        """
        :param epsilon: Maximum guaranteed prediction error bound (default ε = 64).
        :param buffer_ratio: Delta buffer capacity as a fraction of dataset size N (default 1%).
        """
        self._epsilon = max(1, int(epsilon))
        self._buffer_ratio = buffer_ratio
        self._name = f"PGM (ε={self._epsilon})"

        # Main dataset
        self._keys: np.ndarray = np.empty(0, dtype=np.int64)
        self._values: np.ndarray = np.empty(0, dtype=np.int64)
        self._n = 0

        # Segment storage
        self._seg_keys = np.empty(0, dtype=np.int64)      # Starting key of each segment
        self._seg_pos = np.empty(0, dtype=np.int64)       # Starting index position of each segment
        self._seg_slopes = np.empty(0, dtype=np.float64)  # Slope of each segment
        self.num_segments: int = 0

        # Delta buffer for dynamic inserts
        self._delta_keys: list[int] = []
        self._delta_values: list[int] = []
        self._retrain_threshold: int = 100

        # Retrain tracking statistics
        self.retrain_count: int = 0
        self.total_retrain_time_sec: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def epsilon(self) -> int:
        return self._epsilon

    def build(self, sorted_keys: np.ndarray, values: Optional[np.ndarray] = None) -> None:
        """
        Build the piecewise linear segments from sorted unique keys using the O(n) shrinking cone.
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
            self._seg_keys = np.empty(0, dtype=np.int64)
            self._seg_pos = np.empty(0, dtype=np.int64)
            self._seg_slopes = np.empty(0, dtype=np.float64)
            self.num_segments = 0
            return

        self._build_segments()

    def _build_segments(self) -> None:
        """
        Greedy O(n) shrinking cone segmentation algorithm.
        Guarantees that for every key i in segment, |pred(keys[i]) - i| <= epsilon.
        """
        if self._n == 1:
            self._seg_keys = np.array([self._keys[0]], dtype=np.int64)
            self._seg_pos = np.array([0], dtype=np.int64)
            self._seg_slopes = np.array([0.0], dtype=np.float64)
            self.num_segments = 1
            return

        seg_keys_list = []
        seg_pos_list = []
        seg_slopes_list = []

        start_idx = 0
        k0 = float(self._keys[0])
        pos0 = 0.0
        
        # Initial bounds for slope of line passing through (k0, pos0)
        low_slope = 0.0
        high_slope = float("inf")

        prev_low = 0.0
        prev_high = float("inf")

        for i in range(1, self._n):
            ki = float(self._keys[i])
            posi = float(i)
            dx = ki - k0

            if dx <= 0.0:
                continue

            # Compute min and max slope needed to hit (posi +- epsilon)
            s_min = (posi - pos0 - self._epsilon) / dx
            s_max = (posi - pos0 + self._epsilon) / dx

            cand_low = max(low_slope, s_min)
            cand_high = min(high_slope, s_max)

            if cand_low <= cand_high:
                # Still within cone; update slope constraints
                low_slope = cand_low
                high_slope = cand_high
                prev_low = low_slope
                prev_high = high_slope
            else:
                # Cone collapsed: finalize the completed segment
                seg_keys_list.append(self._keys[start_idx])
                seg_pos_list.append(start_idx)
                if prev_high == float("inf"):
                    slope = max(0.0, prev_low)
                else:
                    slope = (prev_low + prev_high) / 2.0
                seg_slopes_list.append(slope)

                # Start fresh segment rooted at key i
                start_idx = i
                k0 = float(self._keys[i])
                pos0 = float(i)
                low_slope = 0.0
                high_slope = float("inf")
                prev_low = 0.0
                prev_high = float("inf")

        # Finalize last segment
        seg_keys_list.append(self._keys[start_idx])
        seg_pos_list.append(start_idx)
        if prev_high == float("inf"):
            slope = max(0.0, prev_low)
        else:
            slope = (prev_low + prev_high) / 2.0
        seg_slopes_list.append(slope)

        self._seg_keys = np.asarray(seg_keys_list, dtype=np.int64)
        self._seg_pos = np.asarray(seg_pos_list, dtype=np.int64)
        self._seg_slopes = np.asarray(seg_slopes_list, dtype=np.float64)
        self.num_segments = len(self._seg_keys)

    def lookup(self, key: int) -> Optional[int]:
        """
        Point lookup:
        1. Find segment via searchsorted on segment first keys.
        2. Predict position: pos0 + slope * (key - k0).
        3. Search bounded window [pred - ε, pred + ε].
        4. Check delta buffer for updates/new items.
        """
        key = int(key)
        found_in_main = False
        val_in_main = None

        if self._n > 0:
            # 1. Binary search over segment starts in O(log(num_segments))
            seg_idx = int(np.searchsorted(self._seg_keys, key, side="right")) - 1
            if seg_idx < 0:
                seg_idx = 0

            k0 = self._seg_keys[seg_idx]
            pos0 = self._seg_pos[seg_idx]
            slope = self._seg_slopes[seg_idx]

            pred = int(np.floor(pos0 + slope * (key - k0)))
            low = max(0, pred - self._epsilon)
            high = min(self._n - 1, pred + self._epsilon)

            if low <= high:
                idx = int(np.searchsorted(self._keys[low:high + 1], key)) + low
                if idx < self._n and self._keys[idx] == key:
                    found_in_main = True
                    val_in_main = int(self._values[idx])

        # 2. Check delta buffer (takes precedence for updates)
        if self._delta_keys:
            b_idx = bisect.bisect_left(self._delta_keys, key)
            if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
                return self._delta_values[b_idx]

        if found_in_main:
            return val_in_main

        return None

    def lookup_trace(self, key: int) -> dict:
        """
        Learned point lookup with complete segment evaluation and search trace.
        Guaranteed to return identical result to lookup(key).
        """
        key = int(key)

        trace = {
            "key": key,
            "segment_id": 0,
            "segment_start_key": 0,
            "slope": 0.0,
            "intercept": 0.0,
            "predicted_position": 0,
            "epsilon": self._epsilon,
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

        # Binary search over segment starts in O(log(num_segments))
        seg_idx = int(np.searchsorted(self._seg_keys, key, side="right")) - 1
        if seg_idx < 0:
            seg_idx = 0

        k0 = int(self._seg_keys[seg_idx])
        pos0 = float(self._seg_pos[seg_idx])
        slope = float(self._seg_slopes[seg_idx])

        pred = int(np.floor(pos0 + slope * (key - k0)))
        low = max(0, pred - self._epsilon)
        high = min(self._n - 1, pred + self._epsilon)

        trace["segment_id"] = seg_idx
        trace["segment_start_key"] = k0
        trace["slope"] = slope
        trace["intercept"] = pos0
        trace["predicted_position"] = pred
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

        # Check delta buffer
        if self._delta_keys:
            b_idx = bisect.bisect_left(self._delta_keys, key)
            if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
                trace["found"] = True
                trace["value"] = self._delta_values[b_idx]
                trace["found_in_delta"] = True

        return trace

    def insert(self, key: int, value: int) -> None:
        """
        Insert into delta buffer. Merge and rebuild when buffer capacity exceeded.
        """
        key = int(key)
        value = int(value)

        b_idx = bisect.bisect_left(self._delta_keys, key)
        if b_idx < len(self._delta_keys) and self._delta_keys[b_idx] == key:
            self._delta_values[b_idx] = value
        else:
            self._delta_keys.insert(b_idx, key)
            self._delta_values.insert(b_idx, value)

        if len(self._delta_keys) >= self._retrain_threshold:
            self._merge_and_rebuild()

    def _merge_and_rebuild(self) -> None:
        """
        Merge delta buffer and re-segment dataset.
        Records retrain timing and count.
        """
        t0 = time.perf_counter()
        self.retrain_count += 1

        if not self._delta_keys:
            return

        d_keys = np.asarray(self._delta_keys, dtype=np.int64)
        d_vals = np.asarray(self._delta_values, dtype=np.int64)

        if self._n == 0:
            merged_keys = d_keys
            merged_vals = d_vals
        else:
            all_keys = np.concatenate([self._keys, d_keys])
            all_vals = np.concatenate([self._values, d_vals])

            sort_order = np.argsort(all_keys, kind="stable")
            sorted_all_keys = all_keys[sort_order]
            sorted_all_vals = all_vals[sort_order]

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

        self._build_segments()

        self.total_retrain_time_sec += (time.perf_counter() - t0)

    def memory_bytes(self) -> int:
        """
        Analytical index memory footprint:
        - Segment metadata:
            seg_keys (num_segments * 8 bytes)
            seg_pos (num_segments * 8 bytes)
            seg_slopes (num_segments * 8 bytes)
        - Delta buffer storage: (len(keys) + len(values)) * 8 bytes
        Excludes raw sorted keys array.
        """
        seg_bytes = self._seg_keys.nbytes + self._seg_pos.nbytes + self._seg_slopes.nbytes
        delta_bytes = (len(self._delta_keys) + len(self._delta_values)) * 8
        return seg_bytes + delta_bytes

    def get_error_stats(self) -> Tuple[float, int, float]:
        """
        Returns (mean_error, max_error, mean_window_size) across all indexed keys.
        """
        if self._n <= 1 or self.num_segments == 0:
            return 0.0, 0, float(min(1, self._n))

        seg_indices = np.searchsorted(self._seg_keys, self._keys, side="right") - 1
        seg_indices = np.clip(seg_indices, 0, self.num_segments - 1)

        k0 = self._seg_keys[seg_indices]
        pos0 = self._seg_pos[seg_indices]
        slopes = self._seg_slopes[seg_indices]

        preds = np.floor(pos0 + slopes * (self._keys - k0)).astype(np.int64)
        actuals = np.arange(self._n, dtype=np.int64)
        errors = np.abs(actuals - preds)

        window_size = float(2 * self._epsilon + 1)
        return float(np.mean(errors)), int(np.max(errors)), window_size

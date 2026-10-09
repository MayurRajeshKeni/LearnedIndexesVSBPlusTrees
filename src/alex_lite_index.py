"""
ALEX-lite: Simplified Updatable Learned Index (Gapped Array)

Note: This is an ALEX-inspired simplification for educational benchmarking,
NOT the original C++ ALEX implementation by Ding et al. 2020.

How ALEX-lite Works (Plain English):
------------------------------------
1. Traditional learned indexes use compact sorted arrays, making direct in-place
   inserts prohibitively expensive (requiring delta buffers).
2. ALEX introduced the "Gapped Array" paradigm: allocate an array with intentional
   empty slots (gaps) interspersed among elements (~30-50% gaps).
3. A linear regression model predicts the target slot position for a given key:
   pred = w * key + b.
4. When a new key is inserted:
   - The index finds the exact sorted insertion slot using model-guided search.
   - If that slot is an empty gap, the key is placed directly (0 shifts).
   - If occupied, elements are shifted locally to the nearest vacant gap (left or right).
5. Node Expansion & Retraining:
   - When the array density exceeds 80% (few gaps remaining, making shifts slower),
     the index doubles capacity, re-distributes elements with fresh gaps, and retrains the model.
"""

import time
from typing import Optional, Tuple
import numpy as np
from src.base import BaseIndex


class AlexLiteIndex(BaseIndex):
    def __init__(self, initial_gap_ratio: float = 0.5, max_density: float = 0.8):
        """
        :param initial_gap_ratio: Ratio of gaps added during build/expansion (0.5 means ~33% gaps).
        :param max_density: Maximum density threshold before expansion (default 0.8 = 80%).
        """
        self._initial_gap_ratio = initial_gap_ratio
        self._max_density = max_density
        self._name = "ALEX-lite"

        self._capacity: int = 0
        self._num_keys: int = 0
        self._keys: np.ndarray = np.empty(0, dtype=np.int64)
        self._values: np.ndarray = np.empty(0, dtype=np.int64)
        self._occupied: np.ndarray = np.empty(0, dtype=bool)

        # Linear model: key -> slot index [0, capacity - 1]
        self._w: float = 0.0
        self._b: float = 0.0

        # Retraining / expansion tracking
        self.retrain_count: int = 0
        self.total_retrain_time_sec: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    def build(self, sorted_keys: np.ndarray, values: Optional[np.ndarray] = None) -> None:
        """
        Bulk build ALEX-lite by distributing sorted keys across a gapped array.
        """
        n = len(sorted_keys)
        self._num_keys = n
        if n == 0:
            self._capacity = 0
            self._keys = np.empty(0, dtype=np.int64)
            self._values = np.empty(0, dtype=np.int64)
            self._occupied = np.empty(0, dtype=bool)
            self._w = 0.0
            self._b = 0.0
            return

        if values is None:
            values = np.arange(n, dtype=np.int64)
        else:
            values = np.asarray(values, dtype=np.int64)

        # Allocate gapped array with ~33-50% gaps
        self._capacity = max(int(np.ceil(n * (1.0 + self._initial_gap_ratio))), n + 2)
        self._keys = np.zeros(self._capacity, dtype=np.int64)
        self._values = np.zeros(self._capacity, dtype=np.int64)
        self._occupied = np.zeros(self._capacity, dtype=bool)

        if n == 1:
            self._keys[0] = int(sorted_keys[0])
            self._values[0] = int(values[0])
            self._occupied[0] = True
            self._w = 0.0
            self._b = 0.0
            return

        # Allocate strictly monotonically increasing slots
        slots = np.zeros(n, dtype=np.int64)
        cur = 0
        for i in range(n):
            target = int(round(i * (self._capacity - 1) / (n - 1)))
            cur = max(cur, target)
            if cur > self._capacity - (n - i):
                cur = self._capacity - (n - i)
            slots[i] = cur
            cur += 1

        # Fit model: keys -> slots
        k_min = float(sorted_keys[0])
        k_max = float(sorted_keys[-1])
        if k_max > k_min:
            self._w = float(self._capacity - 1) / (k_max - k_min)
            self._b = -self._w * k_min
        else:
            self._w = 0.0
            self._b = 0.0

        # Populate gapped array
        for i in range(n):
            s = int(slots[i])
            self._keys[s] = int(sorted_keys[i])
            self._values[s] = int(values[i])
            self._occupied[s] = True

    def _gapped_search(self, key: int) -> Tuple[str, int]:
        """
        Binary search on gapped array that skips unoccupied slots.
        Returns ('found', slot_idx) or ('insert_pos', slot_idx).
        """
        low = 0
        high = self._capacity - 1

        while low <= high:
            mid = (low + high) // 2
            m = mid

            # If mid is not occupied, find nearest occupied slot within [low, high]
            if not self._occupied[m]:
                l_cand = m - 1
                r_cand = m + 1
                found_cand = -1
                while l_cand >= low or r_cand <= high:
                    if r_cand <= high and self._occupied[r_cand]:
                        found_cand = r_cand
                        break
                    if l_cand >= low and self._occupied[l_cand]:
                        found_cand = l_cand
                        break
                    l_cand -= 1
                    r_cand += 1

                if found_cand == -1:
                    # No occupied elements in entire [low, high] window
                    return ('insert_pos', low)
                m = found_cand

            if self._keys[m] == key:
                return ('found', m)
            elif self._keys[m] < key:
                low = m + 1
            else:
                high = m - 1

        return ('insert_pos', min(low, self._capacity))

    def lookup(self, key: int) -> Optional[int]:
        """
        Point lookup: returns value if present, else None.
        """
        if self._num_keys == 0:
            return None

        status, idx = self._gapped_search(int(key))
        if status == 'found':
            return int(self._values[idx])
        return None

    def lookup_trace(self, key: int) -> dict:
        """
        Point lookup trace for ALEX-lite gapped array.
        """
        key = int(key)
        pred_slot = int(np.clip(int(np.floor(key * self._w + self._b)), 0, max(0, self._capacity - 1))) if self._capacity > 0 else 0

        trace = {
            "key": key,
            "predicted_slot": pred_slot,
            "capacity": self._capacity,
            "num_keys": self._num_keys,
            "found": False,
            "final_position": None,
            "value": None,
            "search_steps": []
        }

        if self._num_keys == 0:
            return trace

        # Trace gapped search
        low = 0
        high = self._capacity - 1
        steps = []
        found_idx = None

        while low <= high:
            mid = (low + high) // 2
            m = mid
            if not self._occupied[m]:
                l_cand = m - 1
                r_cand = m + 1
                found_cand = -1
                while l_cand >= low or r_cand <= high:
                    if r_cand <= high and self._occupied[r_cand]:
                        found_cand = r_cand
                        break
                    if l_cand >= low and self._occupied[l_cand]:
                        found_cand = l_cand
                        break
                    l_cand -= 1
                    r_cand += 1
                if found_cand == -1:
                    break
                m = found_cand

            steps.append(int(m))
            if self._keys[m] == key:
                found_idx = m
                break
            elif self._keys[m] < key:
                low = m + 1
            else:
                high = m - 1

        trace["search_steps"] = steps
        if found_idx is not None:
            trace["found"] = True
            trace["final_position"] = found_idx
            trace["value"] = int(self._values[found_idx])

        return trace

    def insert(self, key: int, value: int) -> None:
        """
        Insert (key, value) into gapped array.
        Shifts elements locally to the nearest vacant gap.
        Triggers expansion and retrain when density exceeds max_density.
        """
        key = int(key)
        value = int(value)

        # Check density limit
        if (self._num_keys + 1) / max(1, self._capacity) > self._max_density:
            self._expand_and_retrain()

        if self._capacity == 0:
            self.build(np.array([key], dtype=np.int64), np.array([value], dtype=np.int64))
            return

        status, pos = self._gapped_search(key)
        if status == 'found':
            # Duplicate update
            self._values[pos] = value
            return

        # Insert at `pos`
        if pos < self._capacity and not self._occupied[pos]:
            self._keys[pos] = key
            self._values[pos] = value
            self._occupied[pos] = True
            self._num_keys += 1
            return

        # Find closest gap to the right and left
        gap_right = -1
        for g in range(pos, self._capacity):
            if not self._occupied[g]:
                gap_right = g
                break

        gap_left = -1
        for g in range(pos - 1, -1, -1):
            if not self._occupied[g]:
                gap_left = g
                break

        dist_r = (gap_right - pos) if gap_right != -1 else float("inf")
        dist_l = (pos - 1 - gap_left) if gap_left != -1 else float("inf")

        if dist_r <= dist_l and dist_r != float("inf"):
            # Shift right: move elements [pos, gap_right - 1] to the right
            for i in range(gap_right, pos, -1):
                self._keys[i] = self._keys[i - 1]
                self._values[i] = self._values[i - 1]
                self._occupied[i] = self._occupied[i - 1]
            self._keys[pos] = key
            self._values[pos] = value
            self._occupied[pos] = True
            self._num_keys += 1
        elif dist_l != float("inf"):
            # Shift left: move elements [gap_left + 1, pos - 1] to the left
            for i in range(gap_left, pos - 1):
                self._keys[i] = self._keys[i + 1]
                self._values[i] = self._values[i + 1]
                self._occupied[i] = self._occupied[i + 1]
            insert_slot = pos - 1
            self._keys[insert_slot] = key
            self._values[insert_slot] = value
            self._occupied[insert_slot] = True
            self._num_keys += 1
        else:
            # Array full; expand and retry
            self._expand_and_retrain()
            self.insert(key, value)

    def _expand_and_retrain(self) -> None:
        """
        Expands array capacity and retrains linear model.
        Tracks retraining penalty.
        """
        t0 = time.perf_counter()
        self.retrain_count += 1

        active_keys = self._keys[self._occupied]
        active_vals = self._values[self._occupied]

        self.build(active_keys, active_vals)
        self.total_retrain_time_sec += (time.perf_counter() - t0)

    def memory_bytes(self) -> int:
        """
        Analytical memory footprint of ALEX-lite:
        - Model weights: 16 bytes
        - Gapped array metadata & empty slot overhead
        """
        model_bytes = 16
        occupied_bytes = self._occupied.nbytes
        gap_overhead = (self._capacity - self._num_keys) * 8
        return model_bytes + occupied_bytes + gap_overhead

"""
B+ Tree Index Baseline

Implementation note:
This module utilizes `BTrees.OOBTree` (or `sortedcontainers.SortedDict` as fallback).
In production database engines, B+ Trees are highly optimized, cache-aligned, block-oriented
data structures where inner nodes store routing keys/pointers and leaf nodes store values with
horizontal sibling pointers for sequential scanning. Here, `BTrees.OOBTree` serves as a standard
high-performance in-memory B-Tree library variant standing in for a B+ Tree in Python.
"""

from typing import Optional
import numpy as np
from src.base import BaseIndex

try:
    from BTrees.OOBTree import OOBTree
    HAS_BTREES = True
except ImportError:
    HAS_BTREES = False
    from sortedcontainers import SortedDict

try:
    from pympler import asizeof
    HAS_PYMPLER = True
except ImportError:
    HAS_PYMPLER = False


class BTreeIndex(BaseIndex):
    def __init__(self):
        self._tree = None
        self._name = "B+ Tree"

    @property
    def name(self) -> str:
        return self._name

    def build(self, sorted_keys: np.ndarray, values: Optional[np.ndarray] = None) -> None:
        """
        Bulk build the B-tree.
        If values is None, assigns sequential 0..N-1 index positions.
        """
        n = len(sorted_keys)
        if values is None:
            values = np.arange(n, dtype=np.int64)

        if HAS_BTREES:
            self._tree = OOBTree()
            # Batch populate for fast bulk loading
            self._tree.update(dict(zip(sorted_keys.tolist(), values.tolist())))
        else:
            self._tree = SortedDict(zip(sorted_keys.tolist(), values.tolist()))

    def lookup(self, key: int) -> Optional[int]:
        """
        Logarithmic key lookup: returns value if present, else None.
        """
        if self._tree is None:
            return None
        return self._tree.get(int(key), None)

    def insert(self, key: int, value: int) -> None:
        """
        Insert or update a key-value pair.
        """
        if self._tree is None:
            if HAS_BTREES:
                self._tree = OOBTree()
            else:
                self._tree = SortedDict()
        self._tree[int(key)] = int(value)

    def memory_bytes(self) -> int:
        """
        Honest estimate of index memory footprint.
        Uses pympler.asizeof for accurate recursive heap traversal.
        If pympler is unavailable, applies a standard analytical formula for B-Trees:
        ~48 bytes per entry (keys, values, and tree node pointer overhead).
        """
        if self._tree is None:
            return 0
        if HAS_PYMPLER:
            # pympler traverses all internal node buckets and pointers
            return asizeof.asizeof(self._tree)
        
        # Analytical fallback: ~48-64 bytes per entry for balanced B-tree nodes
        return len(self._tree) * 48

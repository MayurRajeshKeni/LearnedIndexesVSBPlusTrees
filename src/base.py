from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np


class BaseIndex(ABC):
    """
    Abstract base class for all index implementations (B+ Tree, RMI, PGM, ALEX-lite).
    Enforces a uniform key-value interface so comparisons across learned and traditional
    structures are completely fair and reproducible.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the index structure."""
        pass

    @abstractmethod
    def build(self, sorted_keys: np.ndarray, values: Optional[np.ndarray] = None) -> None:
        """
        Bulk build the index from sorted unique keys.
        If values is None, value defaults to the key's index position in the array.
        """
        pass

    @abstractmethod
    def lookup(self, key: int) -> Optional[int]:
        """
        Look up key and return associated value, or None if key is absent.
        """
        pass

    @abstractmethod
    def insert(self, key: int, value: int) -> None:
        """
        Insert (key, value) pair.
        Duplicate policy: If key already exists, updates value to the new value.
        """
        pass

    @abstractmethod
    def memory_bytes(self) -> int:
        """
        Honest estimate of index overhead in bytes.
        Excludes the raw sorted keys/records storage where applicable,
        measuring only the indexing metadata, models, pointers, and delta buffers.
        """
        pass

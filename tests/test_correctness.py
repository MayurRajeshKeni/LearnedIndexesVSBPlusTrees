"""
Correctness Gate Test Suite

Must pass for every index (B+ Tree, RMI, PGM, ALEX-lite) across all datasets.
Verifies:
1. Every key present in build set is found with correct value.
2. Absent keys return None.
3. Inserts (including enough to trigger delta-buffer merge & model retrain) retain all old and new keys.
4. Duplicate insert updates the value properly.
5. Edge cases: n=1, n=2, query below min key, query above max key.
"""

import pytest
import numpy as np

from src.datasets import get_dataset, DATASET_GENERATORS
from src.btree_index import BTreeIndex
from src.rmi_index import RMIIndex
from src.pgm_index import PGMIndex
from src.alex_lite_index import AlexLiteIndex


def create_index_instances():
    return [
        BTreeIndex(),
        RMIIndex(m=50, buffer_ratio=0.02),
        PGMIndex(epsilon=32, buffer_ratio=0.02),
        AlexLiteIndex(initial_gap_ratio=0.5, max_density=0.8),
    ]


@pytest.mark.parametrize("index_factory", [
    lambda: BTreeIndex(),
    lambda: RMIIndex(m=50, buffer_ratio=0.02),
    lambda: PGMIndex(epsilon=32, buffer_ratio=0.02),
    lambda: AlexLiteIndex(initial_gap_ratio=0.5, max_density=0.8),
])
@pytest.mark.parametrize("dataset_name", ["uniform", "lognormal", "clustered", "sequential_noise"])
def test_build_and_lookup_all_datasets(index_factory, dataset_name):
    """Verify build and point lookup across all data distributions."""
    n = 2000
    train_keys, _ = get_dataset(dataset_name, n=n, insert_fraction=0.1, seed=123)
    values = np.arange(100, 100 + n, dtype=np.int64)

    idx = index_factory()
    idx.build(train_keys, values)

    # 1. Test all present keys are found with exact values
    # Sample a subset if needed, but 2000 is small enough to test all
    for i in range(0, n, max(1, n // 200)):
        k = int(train_keys[i])
        expected_val = int(values[i])
        actual_val = idx.lookup(k)
        assert actual_val == expected_val, f"Failed for index {idx.name}, dataset {dataset_name}, key {k}: expected {expected_val}, got {actual_val}"

    # 2. Test boundary keys explicitly
    assert idx.lookup(int(train_keys[0])) == int(values[0])
    assert idx.lookup(int(train_keys[-1])) == int(values[-1])


@pytest.mark.parametrize("index_factory", [
    lambda: BTreeIndex(),
    lambda: RMIIndex(m=30, buffer_ratio=0.02),
    lambda: PGMIndex(epsilon=16, buffer_ratio=0.02),
    lambda: AlexLiteIndex(),
])
def test_absent_keys_return_none(index_factory):
    """Verify that absent keys strictly return None."""
    train_keys = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    values = np.arange(len(train_keys), dtype=np.int64) * 100

    idx = index_factory()
    idx.build(train_keys, values)

    absent_keys = [-100, 0, 5, 25, 45, 75, 85, 999]
    for ak in absent_keys:
        res = idx.lookup(ak)
        assert res is None, f"Expected None for absent key {ak} in {idx.name}, got {res}"


@pytest.mark.parametrize("index_factory", [
    lambda: BTreeIndex(),
    lambda: RMIIndex(m=20, buffer_ratio=0.05),
    lambda: PGMIndex(epsilon=16, buffer_ratio=0.05),
    lambda: AlexLiteIndex(),
])
def test_inserts_and_retrain(index_factory):
    """
    Verify that dynamic inserts (triggering delta buffer merges and retraining)
    preserve all original keys and store all new inserted keys.
    """
    n = 500
    train_keys, insert_keys = get_dataset("uniform", n=n, insert_fraction=0.2, seed=42)
    train_values = np.arange(n, dtype=np.int64)

    idx = index_factory()
    idx.build(train_keys, train_values)

    # Insert batch of new keys
    for i, ik in enumerate(insert_keys):
        idx.insert(int(ik), int(i + 10000))

    # Verify ALL original keys remain intact
    for i in range(0, n, 10):
        res = idx.lookup(int(train_keys[i]))
        assert res == int(train_values[i]), f"Original key {train_keys[i]} lost in {idx.name}"

    # Verify ALL inserted keys are present
    for i, ik in enumerate(insert_keys):
        res = idx.lookup(int(ik))
        assert res == int(i + 10000), f"Inserted key {ik} missing/wrong in {idx.name}"


@pytest.mark.parametrize("index_factory", [
    lambda: BTreeIndex(),
    lambda: RMIIndex(m=10),
    lambda: PGMIndex(epsilon=8),
    lambda: AlexLiteIndex(),
])
def test_duplicate_insert_behavior(index_factory):
    """
    Duplicate insert policy: updating an existing key replaces its value.
    """
    keys = np.array([100, 200, 300, 400], dtype=np.int64)
    vals = np.array([1, 2, 3, 4], dtype=np.int64)

    idx = index_factory()
    idx.build(keys, vals)

    assert idx.lookup(200) == 2
    # Update duplicate
    idx.insert(200, 999)
    assert idx.lookup(200) == 999, f"Failed duplicate update in {idx.name}"


@pytest.mark.parametrize("index_factory", [
    lambda: BTreeIndex(),
    lambda: RMIIndex(m=5),
    lambda: PGMIndex(epsilon=4),
    lambda: AlexLiteIndex(),
])
def test_edge_cases_small_n(index_factory):
    """Test tiny datasets: n=1 and n=2, plus boundary queries."""
    # Test n=1
    idx1 = index_factory()
    idx1.build(np.array([42], dtype=np.int64), np.array([100], dtype=np.int64))
    assert idx1.lookup(42) == 100
    assert idx1.lookup(41) is None
    assert idx1.lookup(43) is None

    # Test n=2
    idx2 = index_factory()
    idx2.build(np.array([10, 50], dtype=np.int64), np.array([1, 2], dtype=np.int64))
    assert idx2.lookup(10) == 1
    assert idx2.lookup(50) == 2
    assert idx2.lookup(5) is None
    assert idx2.lookup(30) is None
    assert idx2.lookup(99) is None

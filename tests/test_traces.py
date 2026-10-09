"""
Tests for Lookup Trace and Visualization B+ Tree

Verifies that:
1. For RMI, PGM, and ALEX-lite, lookup(key) and lookup_trace(key)['value'] return identical results.
2. Trace dictionaries contain all required metadata (bounds, steps, predictions).
3. VizBTree maintains B+ tree properties, correct lookups, splits upon insertion, and valid DOT generation.
"""

import pytest
import numpy as np

from src.datasets import get_dataset
from src.rmi_index import RMIIndex
from src.pgm_index import PGMIndex
from src.alex_lite_index import AlexLiteIndex
from src.viz_btree import VizBTree


def test_rmi_lookup_trace_consistency():
    train_keys, _ = get_dataset("uniform", n=2000, seed=42)
    rmi = RMIIndex(m=20)
    rmi.build(train_keys)

    # Test existing keys
    for k in train_keys[:50]:
        val = rmi.lookup(int(k))
        trace = rmi.lookup_trace(int(k))
        assert trace["found"] is True
        assert trace["value"] == val
        assert trace["final_position"] is not None
        assert trace["chosen_stage2_model_id"] >= 0
        assert len(trace["binary_search_steps"]) > 0
        assert trace["search_window"][0] <= trace["search_window"][1]

    # Test absent keys
    absent_keys = [-999, 0, 99999999]
    for ak in absent_keys:
        assert rmi.lookup(ak) is None
        trace = rmi.lookup_trace(ak)
        assert trace["found"] is False
        assert trace["value"] is None


def test_pgm_lookup_trace_consistency():
    train_keys, _ = get_dataset("uniform", n=2000, seed=42)
    pgm = PGMIndex(epsilon=32)
    pgm.build(train_keys)

    for k in train_keys[:50]:
        val = pgm.lookup(int(k))
        trace = pgm.lookup_trace(int(k))
        assert trace["found"] is True
        assert trace["value"] == val
        assert trace["segment_id"] >= 0
        assert trace["epsilon"] == 32
        assert len(trace["binary_search_steps"]) > 0
        assert trace["search_window"][0] <= trace["search_window"][1]

    absent_keys = [-999, 0, 99999999]
    for ak in absent_keys:
        assert pgm.lookup(ak) is None
        trace = pgm.lookup_trace(ak)
        assert trace["found"] is False
        assert trace["value"] is None


def test_alex_lite_lookup_trace_consistency():
    train_keys, _ = get_dataset("uniform", n=500, seed=42)
    alex = AlexLiteIndex()
    alex.build(train_keys)

    for k in train_keys[:30]:
        val = alex.lookup(int(k))
        trace = alex.lookup_trace(int(k))
        assert trace["found"] is True
        assert trace["value"] == val
        assert trace["predicted_slot"] >= 0
        assert len(trace["search_steps"]) > 0

    absent_keys = [-999, 0, 99999999]
    for ak in absent_keys:
        assert alex.lookup(ak) is None
        trace = alex.lookup_trace(ak)
        assert trace["found"] is False
        assert trace["value"] is None


def test_viz_btree_correctness_and_splits():
    tree = VizBTree(order=4)
    keys = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    tree.build(keys, [k * 10 for k in keys])

    # Verify all keys findable
    for k in keys:
        assert tree.lookup(k) == k * 10
        trace = tree.lookup_trace(k)
        assert trace["found"] is True
        assert trace["value"] == k * 10
        assert len(trace["visited_nodes"]) > 0
        assert trace["visited_nodes"][-1]["is_leaf"] is True

    # Verify absent key
    assert tree.lookup(25) is None
    trace = tree.lookup_trace(25)
    assert trace["found"] is False

    # Insert more keys to trigger multiple node splits
    tree.insert(25, 250)
    tree.insert(35, 350)
    tree.insert(45, 450)
    assert tree.lookup(25) == 250
    assert tree.lookup(35) == 350
    assert tree.lookup(45) == 450
    assert len(tree.split_events) > 0

    # Verify DOT generation
    visited_ids = [n["node_id"] for n in trace["visited_nodes"]]
    dot = tree.to_dot(highlight_path=visited_ids)
    assert "digraph BPlusTree" in dot
    assert "rankdir=TB" in dot
    assert f"node_{visited_ids[0]}" in dot

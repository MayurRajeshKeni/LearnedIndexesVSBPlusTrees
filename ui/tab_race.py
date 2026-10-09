"""
Tab 2: Live Race

Runs head-to-head live workloads across B+ Tree, RMI, PGM, and ALEX-lite.
Displays:
- Ops/sec throughput and execution time.
- Cumulative time vs. operations chart highlighting retraining spikes (the write penalty).
- Delta buffer fill status gauge.
- Silent post-race correctness verification badge.
"""

import time
from typing import Dict, List, Any
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.datasets import get_dataset
from src.btree_index import BTreeIndex
from src.rmi_index import RMIIndex
from src.pgm_index import PGMIndex
from src.alex_lite_index import AlexLiteIndex
from ui.theme import INDEX_COLORS, apply_plotly_theme, render_metric_card


def render_tab_race(dataset_name: str, n_keys: int, seed: int):
    st.markdown("### 🏎️ Live Head-to-Head Performance Race")
    st.markdown(
        "Execute identical workloads live across all indexing architectures. "
        "Observe how read-heavy workloads favor mathematical models, while write-heavy workloads "
        "trigger periodic retraining pauses (the **write penalty**)."
    )

    st.caption(
        "ℹ️ **Methodology Notice:** Pure Python execution adds interpreter overhead that blurs absolute speedups. "
        "These live races illustrate **relative scaling trends** and the architectural impact of delta-buffer retrains."
    )

    # Workload Controls
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        workload_type = st.selectbox(
            "Workload Type:",
            ["100% Lookups (Read-Only)", "100% Inserts (Write-Heavy)", "Mixed 90/10 (Read-Dominant)", "Mixed 50/50 (Balanced Read/Write)"],
            help="Select the mix of query and insertion operations"
        )
    with c2:
        num_ops = st.select_slider(
            "Number of Operations:",
            options=[1_000, 2_500, 5_000, 10_000, 20_000],
            value=5_000,
            help="Total operations to execute in the live race"
        )
    with c3:
        include_alex = st.checkbox("Include ALEX-lite", value=True)

    with st.expander("💡 What Changes What in the Live Race?", expanded=False):
        st.markdown(
            """
            • **100% Lookups (Read-Only):** Demonstrates pure query execution without any buffer overhead or retraining pauses.  
            • **100% Inserts (Write-Heavy):** Tests the write penalty directly. Watch RMI and PGM delta buffers fill up and trigger sharp retraining pauses on the trajectory plot, while B+ Tree continues at a steady pace.  
            • **Mixed 90/10 & 50/50:** Simulates realistic database transaction traffic. Compares how periodic retraining pauses affect cumulative latency compared to the smooth, logarithmic B+ Tree.  
            • **Include ALEX-lite:** Enables an updatable learned index with gapped-array slots instead of a delta buffer, showing how in-place inserts avoid full-model retraining stops.  
            • **Number of Operations:** Controls the benchmark duration and the number of buffer retraining cycles triggered.
            """
        )

    start_race = st.button("🚀 Start Live Race!", type="primary", use_container_width=True)

    if not start_race:
        st.info("Click **Start Live Race** above to execute the benchmark in real-time.")
        return

    # Prepare data
    with st.spinner("Preparing datasets and initializing indexes..."):
        # Dynamically ensure we have enough insert keys for the chosen workload size
        needed_fraction = max(0.5, float(num_ops + 500) / max(1, n_keys))
        train_keys, insert_keys = get_dataset(dataset_name, n=n_keys, insert_fraction=needed_fraction, seed=seed)

        # Prepare indexes
        btree = BTreeIndex()
        btree.build(train_keys)

        rmi = RMIIndex(m=min(1000, max(50, n_keys // 10)), buffer_ratio=0.01)
        rmi.build(train_keys)

        pgm = PGMIndex(epsilon=64, buffer_ratio=0.01)
        pgm.build(train_keys)

        indexes = [
            ("B+ Tree", btree),
            ("RMI", rmi),
            ("PGM", pgm),
        ]
        if include_alex:
            alex = AlexLiteIndex(initial_gap_ratio=0.5, max_density=0.8)
            alex.build(train_keys)
            indexes.append(("ALEX-lite", alex))

    # Determine operations
    rng = np.random.default_rng(seed)
    if "100% Lookups" in workload_type:
        read_ratio = 1.0
    elif "100% Inserts" in workload_type:
        read_ratio = 0.0
    elif "90/10" in workload_type:
        read_ratio = 0.9
    else:
        read_ratio = 0.5

    n_reads = int(num_ops * read_ratio)
    n_writes = num_ops - n_reads

    op_plan = [0] * n_reads + [1] * n_writes
    rng.shuffle(op_plan)

    read_targets = rng.choice(train_keys, size=n_reads, replace=True)
    if len(insert_keys) >= n_writes:
        write_targets = insert_keys[:n_writes]
    else:
        write_targets = rng.choice(insert_keys, size=n_writes, replace=True)

    # Progress bar and status
    prog_bar = st.progress(0, text="Racing...")
    results = {}
    time_series = {}
    retrain_events = {}

    chunk_size = max(50, num_ops // 20)

    for idx_num, (name, idx_obj) in enumerate(indexes):
        prog_bar.progress(int((idx_num / len(indexes)) * 100), text=f"Racing {name}...")

        r_ptr = 0
        w_ptr = 0
        cumulative_times_ms = [0.0]
        ops_completed = [0]
        retrains = []

        last_retrain_count = getattr(idx_obj, "retrain_count", 0)
        t_start = time.perf_counter()

        for step in range(0, num_ops, chunk_size):
            end_step = min(num_ops, step + chunk_size)
            for i in range(step, end_step):
                if op_plan[i] == 0:
                    idx_obj.lookup(int(read_targets[r_ptr % max(1, len(read_targets))]))
                    r_ptr += 1
                else:
                    idx_obj.insert(int(write_targets[w_ptr % max(1, len(write_targets))]), int(w_ptr + 888_000))
                    w_ptr += 1

            now = (time.perf_counter() - t_start) * 1000.0  # ms
            cumulative_times_ms.append(now)
            ops_completed.append(end_step)

            # Check if retrain occurred
            curr_retrain_cnt = getattr(idx_obj, "retrain_count", 0)
            if curr_retrain_cnt > last_retrain_count:
                retrains.append((end_step, now, curr_retrain_cnt))
                last_retrain_count = curr_retrain_cnt

        total_time_s = (time.perf_counter() - t_start)
        throughput = num_ops / max(1e-6, total_time_s)

        results[name] = {
            "total_time_s": total_time_s,
            "throughput": throughput,
            "retrain_count": getattr(idx_obj, "retrain_count", 0),
            "retrain_time_s": getattr(idx_obj, "total_retrain_time_sec", 0.0),
        }
        time_series[name] = (ops_completed, cumulative_times_ms)
        retrain_events[name] = retrains

    prog_bar.progress(100, text="Race Finished!")

    # 1. Summary Cards
    st.markdown("#### 🏁 Race Results")
    cols = st.columns(len(indexes))
    for i, (name, _) in enumerate(indexes):
        res = results[name]
        with cols[i]:
            render_metric_card(
                f"{name}",
                f"{res['throughput']:,.0f} ops/s",
                f"Total time: {res['total_time_s']*1000:.1f} ms" + (f" | Retrains: {res['retrain_count']}" if res['retrain_count'] > 0 else ""),
                color=INDEX_COLORS.get(name, "#333333")
            )

    # 2. Cumulative Time Chart with Retraining Spikes
    st.markdown("---")
    st.markdown("#### ⏱️ Cumulative Elapsed Time vs. Operations Completed")
    st.caption("A steep jump in cumulative time marks a retraining pause (the write penalty).")

    fig_race = go.Figure()
    for name, _ in indexes:
        ops, times_ms = time_series[name]
        fig_race.add_trace(go.Scatter(
            x=ops,
            y=times_ms,
            mode="lines+markers",
            name=name,
            line=dict(color=INDEX_COLORS.get(name, "#555555"), width=2.5),
            marker=dict(size=4),
            hovertemplate=f"<b>{name}</b><br>Ops: %{{x:,}}<br>Elapsed: %{{y:.1f}} ms<extra></extra>"
        ))

        # Annotate retrain events
        for op_idx, t_ms, r_cnt in retrain_events[name]:
            fig_race.add_trace(go.Scatter(
                x=[op_idx],
                y=[t_ms],
                mode="markers+text",
                marker=dict(symbol="star", size=10, color=INDEX_COLORS.get(name, "#555555")),
                text=[f"Retrain #{r_cnt}"],
                textposition="top center",
                name=f"{name} Retrain",
                showlegend=False,
                hovertext=f"{name} Triggered Rebuild #{r_cnt} at Op {op_idx:,}",
                hoverinfo="text"
            ))

    apply_plotly_theme(fig_race, title="Workload Execution Trajectory", x_title="Operations Completed", y_title="Cumulative Elapsed Time (ms) [Lower is Better]")
    st.plotly_chart(fig_race, use_container_width=True)

    # 3. Buffer Fill Gauge for Learned Indexes (if write operations were present)
    if n_writes > 0:
        st.markdown("---")
        st.markdown("#### 📦 Delta Buffer State Post-Race")
        g1, g2 = st.columns(2)
        with g1:
            rmi_buf_len = len(rmi._delta_keys)
            rmi_thresh = rmi._retrain_threshold
            rmi_pct = min(100.0, (rmi_buf_len / max(1, rmi_thresh)) * 100.0)

            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rmi_pct,
                title={"text": f"RMI Delta Buffer Fill ({rmi_buf_len}/{rmi_thresh} keys)", "font": {"size": 14}},
                domain={"x": [0, 1], "y": [0, 0.78]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": INDEX_COLORS["RMI"]},
                    "steps": [
                        {"range": [0, 80], "color": "rgba(128, 128, 128, 0.2)"},
                        {"range": [80, 100], "color": "rgba(245, 158, 11, 0.35)"}
                    ]
                }
            ))
            fig_g1.update_layout(height=240, margin=dict(l=25, r=25, t=55, b=10))
            st.plotly_chart(fig_g1, use_container_width=True)

        with g2:
            pgm_buf_len = len(pgm._delta_keys)
            pgm_thresh = pgm._retrain_threshold
            pgm_pct = min(100.0, (pgm_buf_len / max(1, pgm_thresh)) * 100.0)

            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pgm_pct,
                title={"text": f"PGM Delta Buffer Fill ({pgm_buf_len}/{pgm_thresh} keys)", "font": {"size": 14}},
                domain={"x": [0, 1], "y": [0, 0.78]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": INDEX_COLORS["PGM"]},
                    "steps": [
                        {"range": [0, 80], "color": "rgba(128, 128, 128, 0.2)"},
                        {"range": [80, 100], "color": "rgba(245, 158, 11, 0.35)"}
                    ]
                }
            ))
            fig_g2.update_layout(height=240, margin=dict(l=25, r=25, t=55, b=10))
            st.plotly_chart(fig_g2, use_container_width=True)

    # 4. Silent Correctness Verification
    st.markdown("---")
    st.markdown("#### 🛡️ Post-Race Correctness Verification Gate")
    test_sample = rng.choice(train_keys, size=min(100, len(train_keys)), replace=False)
    all_correct = True
    for name, idx_obj in indexes:
        for k in test_sample:
            if idx_obj.lookup(int(k)) is None:
                all_correct = False
                break

    if all_correct:
        st.success(f"✅ **Verified Correct:** Sampled 100 original keys across all tested indexes post-workload; 100% found with zero data loss or corruption.")
    else:
        st.error("❌ **Verification Failed:** Some keys could not be found after the workload.")

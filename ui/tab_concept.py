"""
Tab 1: How It Works (Visual Concept & Trajectory Comparison)

Side-by-side comparison of:
- LEFT: B+ Tree pointer-based tree traversal with search path highlighted in Graphviz.
- RIGHT: Learned Index CDF approximation with model predictions, error bands, and search window.
"""

from typing import Dict, Any, List
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.datasets import get_dataset
from src.rmi_index import RMIIndex
from src.pgm_index import PGMIndex
from src.viz_btree import VizBTree
from ui.theme import INDEX_COLORS, apply_plotly_theme, render_metric_card, render_info_cloud


def render_tab_concept(
    dataset_name: str,
    n_keys: int,
    rmi_m: int,
    pgm_eps: int,
    seed: int
):
    st.markdown("### 🧠 How It Works: Pointer Chasing vs. Function Approximation")
    st.markdown(
        "Compare how a traditional **B+ Tree** and a **Learned Index** find the exact same key. "
        "The B+ Tree chases pointers down balanced tree nodes, while the Learned Index calculates the position directly."
    )

    concept_clouds = [
        {
            "knob": "Dataset Distribution",
            "category": "Data Shape",
            "effect": "Alters the Blue CDF Line & Model Accuracy",
            "detail": "Uniform creates a flat diagonal line (low error). Clustered creates step cliffs that widen error bounds."
        },
        {
            "knob": "Dataset Size (N)",
            "category": "Volume",
            "effect": "Scales Array Length & B+ Tree Depth",
            "detail": "B+ Tree height grows with O(log N). The Learned Index keeps calculating a direct 1-step coordinate."
        },
        {
            "knob": "RMI Models (M)",
            "category": "Resolution",
            "effect": "Splits CDF Into Finer Local Linear Models",
            "detail": "Higher M narrows the shaded error band [min_err, max_err], shrinking the final binary search window."
        },
        {
            "knob": "PGM Error Bound (ε)",
            "category": "Tolerance",
            "effect": "Controls Shaded Error Corridor ±ε",
            "detail": "Smaller ε creates more linear segments (more memory) for tighter search; larger ε compresses into fewer segments."
        },
        {
            "knob": "Query Mode",
            "category": "Lookup",
            "effect": "Traces Present Key vs. Absent Verification",
            "detail": "See how learned bounds pinpoint existing keys, or quickly reject absent keys without scanning the whole dataset."
        }
    ]
    render_info_cloud("What Changes What? (Concept Explorer Guide)", concept_clouds, icon="💡")

    # 1. Dataset generation
    train_keys, _ = get_dataset(dataset_name, n=n_keys, seed=seed)

    # Key selector controls
    st.markdown("#### 🎯 Select a Query Key to Trace")
    c_sel1, c_sel2, c_sel3 = st.columns([3, 1, 1])

    # Session state for key selection
    if "selected_key_idx" not in st.session_state or st.session_state.selected_key_idx >= len(train_keys):
        st.session_state.selected_key_idx = len(train_keys) // 2

    with c_sel2:
        if st.button("🎲 Random Present Key", use_container_width=True):
            st.session_state.selected_key_idx = int(np.random.randint(0, len(train_keys)))
            st.session_state.absent_key_mode = False

    with c_sel3:
        if st.button("🚫 Key That Does Not Exist", use_container_width=True):
            st.session_state.absent_key_mode = True

    absent_mode = st.session_state.get("absent_key_mode", False)

    with c_sel1:
        if not absent_mode:
            key_idx = st.slider(
                "Pick key by array rank (index position):",
                min_value=0,
                max_value=len(train_keys) - 1,
                value=st.session_state.selected_key_idx,
                help="Slide to pick any key in the dataset"
            )
            st.session_state.selected_key_idx = key_idx
            target_key = int(train_keys[key_idx])
            st.caption(f"Selected Key: **{target_key:,}** (Rank: {key_idx:,} of {n_keys:,})")
        else:
            # Pick a non-existent key between existing keys
            mid_val = int(train_keys[st.session_state.selected_key_idx])
            target_key = mid_val + 1
            while target_key in train_keys:
                target_key += 1
            st.caption(f"Testing Absent Key: **{target_key:,}** (Does NOT exist in index)")

    # Learned index model selector
    learned_choice = st.radio(
        "Learned Index Model to Compare:",
        ["PGM (Piecewise Linear)", "RMI (2-Stage Hierarchy)"],
        horizontal=True
    )
    is_pgm = "PGM" in learned_choice

    # 2. Build indexes and run traces
    # For VizBTree, cap keys if N > 150 for diagram readability
    viz_n = min(120, n_keys)
    if n_keys <= 120:
        viz_keys = train_keys
    else:
        # Sample subset around target key so visualization remains readable
        sample_indices = np.linspace(0, n_keys - 1, viz_n, dtype=int)
        viz_keys = np.unique(np.sort(np.append(train_keys[sample_indices], target_key)))

    viz_tree = VizBTree(order=4)
    viz_tree.build(viz_keys.tolist())
    btree_trace = viz_tree.lookup_trace(target_key)

    if is_pgm:
        model = PGMIndex(epsilon=pgm_eps)
        model.build(train_keys)
        learned_trace = model.lookup_trace(target_key)
        model_name = f"PGM (ε={pgm_eps})"
    else:
        model = RMIIndex(m=rmi_m)
        model.build(train_keys)
        learned_trace = model.lookup_trace(target_key)
        model_name = f"RMI (M={rmi_m})"

    # 3. Two-Column Visual Layout
    col_left, col_right = st.columns(2)

    # --- LEFT: B+ TREE ---
    with col_left:
        st.markdown(f"#### 🌲 B+ Tree Search Trajectory (`BTrees`)")
        st.caption("Pointer traversal down internal routing nodes to leaf data page.")

        visited_node_ids = [n["node_id"] for n in btree_trace["visited_nodes"]]
        dot_str = viz_tree.to_dot(highlight_path=visited_node_ids, max_nodes=50)

        # Graphviz Diagram
        st.graphviz_chart(dot_str, use_container_width=True)

        # Plain English summary
        num_nodes = len(btree_trace["visited_nodes"])
        levels = btree_trace["max_level"] + 1
        comparisons = btree_trace["comparisons"]

        status_text = "Found" if btree_trace["found"] else "Not Found"
        sample_note = f" *(Diagram displays representative sample of {len(viz_keys)} keys for visual clarity)*" if n_keys > 120 else ""
        st.info(
            f"**Traversal Summary:** Visited **{num_nodes} nodes** across **{levels} levels**; "
            f"compared **{comparisons} keys**. Result: `{status_text}`.{sample_note}"
        )

    # --- RIGHT: LEARNED INDEX ---
    with col_right:
        st.markdown(f"#### 📈 {model_name} Function Approximation")
        st.caption("Predicts key position via regression, then binary searches bounded window.")

        # Plot CDF and model prediction
        fig = go.Figure()

        # 1. Actual CDF curve
        step_sample = max(1, len(train_keys) // 2000)
        sample_keys = train_keys[::step_sample]
        sample_ranks = np.arange(len(train_keys))[::step_sample]

        fig.add_trace(go.Scatter(
            x=sample_keys,
            y=sample_ranks,
            mode="lines",
            name="True CDF (Data)",
            line=dict(color="#1f77b4", width=2.5),
            hovertemplate="Key: %{x:,}<br>Actual Position: %{y:,}<extra></extra>"
        ))

        # 2. Model Prediction Line & Error Band
        pred_pos = learned_trace["predicted_position"]
        search_lo, search_hi = learned_trace["search_window"]
        win_size = search_hi - search_lo + 1

        if is_pgm:
            # Segment line
            k0 = learned_trace["segment_start_key"]
            pos0 = learned_trace["intercept"]
            slope = learned_trace["slope"]
            eps = learned_trace["epsilon"]

            # Plot segment line around query
            span_k = np.linspace(max(train_keys[0], target_key - 5000), min(train_keys[-1], target_key + 5000), 50)
            pred_curve = pos0 + slope * (span_k - k0)

            fig.add_trace(go.Scatter(
                x=span_k,
                y=pred_curve,
                mode="lines",
                name="PGM Segment Model",
                line=dict(color=INDEX_COLORS["PGM"], width=2, dash="dot"),
                hovertemplate="Predicted Pos: %{y:.1f}<extra></extra>"
            ))
            # Error band
            fig.add_trace(go.Scatter(
                x=list(span_k) + list(span_k[::-1]),
                y=list(pred_curve + eps) + list((pred_curve - eps)[::-1]),
                fill="toself",
                fillcolor="rgba(42, 157, 143, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="±ε Error Band"
            ))
        else:
            # RMI Model
            m_id = learned_trace["chosen_stage2_model_id"]
            w = model._w2[m_id]
            b = model._b2[m_id]
            min_err = learned_trace["error_min"]
            max_err = learned_trace["error_max"]

            span_k = np.linspace(max(train_keys[0], target_key - 5000), min(train_keys[-1], target_key + 5000), 50)
            pred_curve = span_k * w + b

            fig.add_trace(go.Scatter(
                x=span_k,
                y=pred_curve,
                mode="lines",
                name=f"RMI Model #{m_id}",
                line=dict(color=INDEX_COLORS["RMI"], width=2, dash="dot"),
                hovertemplate="Predicted Pos: %{y:.1f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=list(span_k) + list(span_k[::-1]),
                y=list(pred_curve + max_err) + list((pred_curve + min_err)[::-1]),
                fill="toself",
                fillcolor="rgba(226, 109, 92, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="[min_err, max_err] Band"
            ))

        # 3. Highlight Selected Key Query
        fig.add_trace(go.Scatter(
            x=[target_key],
            y=[pred_pos],
            mode="markers",
            marker=dict(size=11, color="#e63946", symbol="diamond"),
            name="Model Prediction",
            hovertemplate=f"Key: {target_key:,}<br>Predicted Pos: {pred_pos:,}<extra></extra>"
        ))

        if learned_trace["found"]:
            final_p = learned_trace["final_position"]
            fig.add_trace(go.Scatter(
                x=[target_key],
                y=[final_p],
                mode="markers",
                marker=dict(size=9, color="#2a9d8f", symbol="circle"),
                name="True Position",
                hovertemplate=f"Key: {target_key:,}<br>Actual Pos: {final_p:,}<extra></extra>"
            ))

        apply_plotly_theme(fig, title="", x_title="Key Value", y_title="Array Position (0 to N-1)")
        st.plotly_chart(fig, use_container_width=True)

        # Plain English summary
        true_pos_str = f"{learned_trace['final_position']:,}" if learned_trace['found'] else "Absent"
        st.info(
            f"**Prediction Summary:** The model predicted position **{pred_pos:,}**. "
            f"The true position was **{true_pos_str}**. "
            f"Bounded search window: `[{search_lo:,}, {search_hi:,}]` "
            f"(searched **{win_size:,} slots** instead of all {n_keys:,}!)."
        )

    # 4. Side-by-Side Comparison Metrics
    st.markdown("---")
    st.markdown("#### ⚖️ Side-by-Side Trajectory Comparison")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            "B+ Tree Pointer Hops",
            f"{num_nodes} nodes",
            f"{comparisons} key comparisons",
            color=INDEX_COLORS["B+ Tree"]
        )
    with m2:
        render_metric_card(
            f"{model_name} Binary Probes",
            f"{len(learned_trace['binary_search_steps'])} probes",
            f"Searched {win_size} keys",
            color=INDEX_COLORS["PGM"] if is_pgm else INDEX_COLORS["RMI"]
        )
    with m3:
        # Approximate memory
        btree_mem_est = n_keys * 48  # approx bytes
        render_metric_card(
            "B+ Tree Index Overhead",
            f"~{btree_mem_est / 1024:.1f} KB",
            "Pointer buckets (approx)",
            color=INDEX_COLORS["B+ Tree"]
        )
    with m4:
        learned_mem = model.memory_bytes()
        render_metric_card(
            f"{model_name} Overhead",
            f"~{learned_mem / 1024:.2f} KB" if learned_mem >= 1024 else f"{learned_mem} bytes",
            "Regression weights & bounds",
            color=INDEX_COLORS["PGM"] if is_pgm else INDEX_COLORS["RMI"]
        )

    # 5. Zoomed-In Search Window Toggle
    st.markdown("---")
    show_zoom = st.toggle("🔍 Show the Zoomed-In Search Window (Binary Search Probes)", value=False)
    if show_zoom:
        st.markdown("##### Zoomed-In Search Interval & Binary Search Steps")
        steps = learned_trace["binary_search_steps"]

        # Window keys
        w_start = max(0, search_lo - 2)
        w_end = min(len(train_keys) - 1, search_hi + 2)
        win_keys = train_keys[w_start:w_end + 1]
        win_positions = np.arange(w_start, w_end + 1)

        step_map = {step: idx + 1 for idx, step in enumerate(steps)}

        z_fig = go.Figure()
        # Bar representation of slots
        colors = []
        labels = []
        for p in win_positions:
            if p in step_map:
                if learned_trace["found"] and p == learned_trace["final_position"]:
                    colors.append("#2a9d8f")  # Green Match!
                    labels.append(f"Probe #{step_map[p]} (MATCH)")
                else:
                    colors.append("#e76f51")  # Orange Probe
                    labels.append(f"Probe #{step_map[p]}")
            elif search_lo <= p <= search_hi:
                colors.append("#a8dadc")  # Inside window
                labels.append("Inside Window")
            else:
                colors.append("#e9ecef")  # Outside window
                labels.append("Outside Window")

        z_fig.add_trace(go.Bar(
            x=win_positions,
            y=[1] * len(win_positions),
            marker_color=colors,
            text=labels,
            hovertext=[f"Pos: {p}<br>Key: {k:,}<br>{lbl}" for p, k, lbl in zip(win_positions, win_keys, labels)],
            hoverinfo="text",
            showlegend=False
        ))
        apply_plotly_theme(
            z_fig,
            title=f"Bounded Window [{search_lo:,} to {search_hi:,}] - Evaluated in {len(steps)} Probes",
            x_title="Array Index Position",
            y_title=""
        )
        z_fig.update_yaxes(showticklabels=False)
        st.plotly_chart(z_fig, use_container_width=True)

    # 6. Beginner-Friendly Explainer Expander
    with st.expander("📖 Explain this in simple words", expanded=True):
        if dataset_name == "uniform":
            st.markdown(
                """
                1. **Data Shape:** Because the numbers in the **Uniform** dataset are spaced evenly, their CDF forms an almost perfect straight diagonal line.
                2. **Why the Model Wins:** A simple linear equation ($y = m \cdot x + c$) can approximate this line with near-zero error. The prediction is right next to the real target.
                3. **Tiny Search:** Because the error is tiny, the index only had to check a handful of positions.
                4. **Memory Magic:** Instead of storing thousands of pointers across multiple tree levels, the learned index just remembers a couple of numbers (the slope and intercept).
                """
            )
        elif dataset_name == "clustered":
            st.markdown(
                """
                1. **Data Shape:** The **Clustered** dataset has dense groups of numbers separated by huge empty gaps. The CDF looks like a staircase with steep cliffs.
                2. **Why Models Struggle:** Straight lines cannot easily trace sharp staircases. In the gaps, a single line overshoots or undershoots, creating wider error bands.
                3. **Wider Binary Search:** When the error band is wider, the binary search must check more slots before finding the key.
                4. **Why B+ Trees Stay Steady:** The B+ Tree doesn't care about curves; it simply balances keys across tree nodes, keeping search depth identical regardless of gaps.
                """
            )
        else:
            st.markdown(
                """
                1. **Data Shape:** In the **Lognormal** dataset, values are tightly packed at the start and spread thinly with a long tail.
                2. **Local Adaptation:** By dividing the range into smaller sub-models (RMI) or adaptive segments (PGM), learned indexes fit the curve piecewise.
                3. **Predictive Speed:** Once the correct model piece is chosen, computation in CPU registers immediately narrows down the location.
                4. **Write Trade-off:** While reads are fast, adding new numbers requires updating delta buffers and eventually re-fitting the curve equations.
                """
            )

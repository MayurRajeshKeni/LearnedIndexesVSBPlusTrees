"""
Tab 4: Data Explorer

Visualizes the Cumulative Distribution Function (CDF) and key gaps distribution.
Shows why data predictability determines learned index efficiency.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.datasets import get_dataset
from src.rmi_index import RMIIndex
from src.pgm_index import PGMIndex
from ui.theme import INDEX_COLORS, apply_plotly_theme, DATASET_DESCRIPTIONS, render_metric_card, render_info_cloud


def render_tab_data(dataset_name: str, n_keys: int, seed: int):
    st.markdown("### 🔍 Dataset Distribution & Model Adaptability Explorer")
    st.markdown(
        "Learned indexes treat data as a Cumulative Distribution Function (CDF). "
        "A smooth, continuous curve can be approximated with few models; sharp steps, bursts, "
        "and gaps force wider error bounds or numerous segments."
    )

    # Load dataset
    train_keys, _ = get_dataset(dataset_name, n=n_keys, seed=seed)

    # 1. Dataset stats & description
    st.info(f"**{dataset_name.replace('_', ' ').title()} Dataset:** {DATASET_DESCRIPTIONS.get(dataset_name, '')}")

    data_clouds = [
        {
            "knob": "CDF Curvature",
            "category": "Math",
            "effect": "Dictates Linear Regression Residuals",
            "detail": "A straight CDF yields near-zero prediction errors (<= 5 keys). Curvature forces wider error bounds."
        },
        {
            "knob": "Key Distribution Gaps",
            "category": "Sparsity",
            "effect": "Causes Model Overshoot / Undershoot",
            "detail": "Large gaps between dense clusters create steep staircases that challenge single linear regressions."
        },
        {
            "knob": "Uniform & Sequential",
            "category": "Optimal",
            "effect": "Best Case for Learned Indexes",
            "detail": "Models achieve 90%+ memory savings and tiny search windows because data fits a linear slope."
        },
        {
            "knob": "Clustered & Lognormal",
            "category": "Hostile",
            "effect": "Challenges Single Linear Models",
            "detail": "Requires hierarchical RMI sub-models or fine PGM segments to adapt to non-linear regional slopes."
        }
    ]
    render_info_cloud("What Changes What? (CDF Shape vs. Index Difficulty)", data_clouds, icon="🔍")

    col1, col2 = st.columns(2)

    # Plot 1: CDF
    with col1:
        st.markdown("#### Cumulative Distribution Function (CDF)")
        ranks = np.linspace(0, 1, len(train_keys))

        # Sample for plotting if N is large
        if len(train_keys) > 5000:
            sample_idx = np.linspace(0, len(train_keys) - 1, 5000, dtype=int)
            plot_keys = train_keys[sample_idx]
            plot_ranks = ranks[sample_idx]
        else:
            plot_keys = train_keys
            plot_ranks = ranks

        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=plot_keys,
            y=plot_ranks,
            mode="lines",
            name="Empirical CDF",
            line=dict(color="#1f77b4", width=2.5),
            hovertemplate="Key: %{x:,}<br>Rank (Pos/N): %{y:.4f}<extra></extra>"
        ))
        # Ideal linear reference
        fig_cdf.add_trace(go.Scatter(
            x=[plot_keys[0], plot_keys[-1]],
            y=[0.0, 1.0],
            mode="lines",
            name="Ideal Linear CDF",
            line=dict(color="#7f7f7f", dash="dash", width=1.5),
            hoverinfo="skip"
        ))
        apply_plotly_theme(fig_cdf, title=f"CDF of {dataset_name.title()}", x_title="Key Value", y_title="Normalized Rank (0 to 1)")
        st.plotly_chart(fig_cdf, use_container_width=True)

    # Plot 2: Histogram of Key Gaps
    with col2:
        st.markdown("#### Key Gaps Distribution ($\Delta k = k_{i+1} - k_i$)")
        gaps = np.diff(train_keys)
        # Cap outliers for histogram readability
        p99_gap = np.percentile(gaps, 99)
        filtered_gaps = gaps[gaps <= max(1.0, p99_gap * 1.5)]

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Histogram(
            x=filtered_gaps,
            nbinsx=40,
            marker_color="#2b5c8f",
            opacity=0.8,
            hovertemplate="Gap Range: %{x}<br>Frequency: %{y:,}<extra></extra>"
        ))
        apply_plotly_theme(fig_gap, title="Distribution of Gaps Between Consecutive Keys", x_title="Gap Size (Difference Between Keys)", y_title="Frequency")
        st.plotly_chart(fig_gap, use_container_width=True)

    # 2. Train quick RMI and PGM to compute empirical error stats for this dataset
    st.markdown("---")
    st.markdown("#### Model Error & Compression Metrics for this Dataset")

    rmi = RMIIndex(m=min(1000, max(50, n_keys // 10)))
    rmi.build(train_keys)
    rmi_mean_err, rmi_max_err, rmi_win = rmi.get_error_stats()

    pgm = PGMIndex(epsilon=64)
    pgm.build(train_keys)
    pgm_mean_err, pgm_max_err, pgm_win = pgm.get_error_stats()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        render_metric_card("PGM Segments Created", f"{pgm.num_segments:,}", "Fewer segments = higher compression", color=INDEX_COLORS["PGM"])
    with m_col2:
        render_metric_card("PGM Max Search Window", f"{int(pgm_win)} keys", "Guaranteed <= 2ε + 1 keys", color=INDEX_COLORS["PGM"])
    with m_col3:
        render_metric_card("RMI Mean Error", f"{rmi_mean_err:.1f} keys", f"Max error: {rmi_max_err} keys", color=INDEX_COLORS["RMI"])
    with m_col4:
        render_metric_card("RMI Avg Search Window", f"{rmi_win:.1f} keys", f"Over {rmi._m} stage-2 models", color=INDEX_COLORS["RMI"])

    # Computed Verdict
    st.markdown("#### Computed Suitability Verdict")
    if dataset_name in ["uniform", "sequential_noise"]:
        verdict = "🟢 <b>Learned Index Highly Recommended:</b> The CDF is near-linear with regular key gaps. Mathematical models achieve low prediction error (&lt; 10 keys) and superior memory compression."
    elif dataset_name == "lognormal":
        verdict = "🟡 <b>Learned Index Moderate / Mixed:</b> The distribution has a dense head and long right tail. Multi-stage RMI handles it well, but segment counts increase in high-density regions."
    else:  # clustered
        verdict = "🔴 <b>Traditional B+ Tree Recommended:</b> Dense clusters separated by large empty voids create a step-like CDF. Linear models suffer large local prediction errors, increasing search window widths."

    st.markdown(
        f"""
        <div style="background-color: var(--secondary-background-color, rgba(66, 153, 225, 0.1)); border-left: 4px solid #3182ce; padding: 12px 18px; border-radius: 6px;">
            {verdict}
        </div>
        """,
        unsafe_allow_html=True
    )

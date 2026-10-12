"""
Shared UI Theme and Styling Utilities

Maintains consistent colors, Plotly themes, and badges across all tabs.
"""

from typing import Dict, Any
import plotly.graph_objects as go
import streamlit as st

# Canonical color palette per index structure
INDEX_COLORS: Dict[str, str] = {
    "B+ Tree": "#2b5c8f",     # Classic Database Blue
    "RMI": "#e26d5c",         # Coral / Flame
    "PGM": "#2a9d8f",         # Emerald / Teal
    "ALEX-lite": "#9b5de5",    # Violet / Purple
}

DATASET_DESCRIPTIONS: Dict[str, str] = {
    "uniform": "Keys spread evenly across the domain. CDF is virtually a straight diagonal line. Ideal for linear models.",
    "lognormal": "Skewed distribution with heavy right tail. Dense low keys, sparse high keys. Tests adaptability to non-linear slopes.",
    "clustered": "Dense bursts separated by vast empty gaps (like server timestamps). Step-function CDF presents a challenge for linear regressions.",
    "sequential_noise": "Auto-incrementing sequence with local jitter. Nearly linear CDF with mild fluctuations.",
}


def apply_plotly_theme(
    fig: go.Figure,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    legend_below: bool = False
) -> go.Figure:
    """Applies clean, responsive theme supporting light and dark UI modes without title/legend collisions."""
    layout_args: Dict[str, Any] = {
        "template": "plotly_white",
        "xaxis_title": x_title,
        "yaxis_title": y_title,
        "hoverlabel": dict(font_size=12, font_family="Helvetica")
    }

    if title:
        layout_args["title"] = {"text": title, "font": {"size": 15, "weight": 700}}
        layout_args["margin"] = dict(l=40, r=30, t=55, b=55)
        # Place legend below to ensure zero collision with the title
        layout_args["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        )
    else:
        layout_args["margin"] = dict(l=40, r=30, t=35, b=40)
        # Without title, legend has full unobstructed top row
        layout_args["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )

    if legend_below:
        layout_args["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        )
        layout_args["margin"] = dict(l=40, r=30, t=layout_args.get("margin", {}).get("t", 40), b=55)

    fig.update_layout(**layout_args)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
    return fig


import textwrap


def render_html(html_str: str) -> None:
    """Safely renders HTML in Streamlit without Markdown indentation/code block corruption."""
    clean_html = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtext: str = "", color: str = "#2b5c8f") -> None:
    """Renders a styled metric card using native Streamlit bordered container."""
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"### {value}")
        if subtext:
            st.caption(subtext)


def render_info_cloud(title: str, items: list, icon: str = "💡") -> None:
    """Renders a clean, native Streamlit grid container explaining 'What Changes What'."""
    with st.expander(f"{icon} **{title}**", expanded=True):
        n_cols = min(3, len(items)) if items else 1
        cols = st.columns(n_cols)
        for i, it in enumerate(items):
            knob = it.get("knob", "")
            effect = it.get("effect", "")
            detail = it.get("detail", "")
            category = it.get("category", "Knob")
            with cols[i % n_cols]:
                with st.container(border=True):
                    st.caption(f"🏷️ `{category}`")
                    st.markdown(f"**{knob}**")
                    st.markdown(f"**➔ {effect}**")
                    st.caption(detail)



def render_glossary_cloud() -> None:
    """Renders a comprehensive glossary cloud with key terminology, acronyms, and literature citations."""
    glossary_items = [
        {
            "term": "RMI",
            "name": "Recursive Model Index",
            "citation": "Kraska et al., SIGMOD 2018",
            "desc": "Hierarchical network of linear models replacing internal B+ Tree nodes to predict key locations via regression."
        },
        {
            "term": "PGM",
            "name": "Piecewise Geometric Model Index",
            "citation": "Ferragina & Vinciguerra, PVLDB 2020",
            "desc": "Optimal streaming segmentation guaranteeing that maximum prediction error never exceeds a strict tolerance (ε)."
        },
        {
            "term": "ALEX",
            "name": "Adaptive Learned Extensible Index",
            "citation": "Ding et al., SIGMOD 2020",
            "desc": "Dynamic in-memory learned index using gapped arrays to absorb new insertions in-place without full-model retraining stops."
        },
        {
            "term": "B+ Tree",
            "name": "Self-Balancing Search Tree",
            "citation": "Bayer & McCreight, 1972",
            "desc": "Traditional database index organizing keys in multi-way balanced leaf pages with deterministic O(log_B N) pointer traversal."
        },
        {
            "term": "CDF",
            "name": "Cumulative Distribution Function",
            "citation": "Statistics",
            "desc": "F(x) = P(X ≤ x). A learned index trains a regression model to approximate the empirical inverse CDF of the dataset."
        },
        {
            "term": "SOSD",
            "name": "Search on Structured Data",
            "citation": "Marcus et al., 2020",
            "desc": "The open-source standardized benchmark platform for rigorously comparing learned indexes against traditional B-trees."
        },
        {
            "term": "Delta Buffer",
            "name": "Write Staging Buffer",
            "citation": "Architecture",
            "desc": "Appends new keys in a fast auxiliary memory buffer to delay expensive full retraining cycles for static learned models."
        },
        {
            "term": "Write Penalty",
            "name": "Retraining Pause Penalty",
            "citation": "Trade-off",
            "desc": "The dramatic throughput drop that static learned indexes suffer when write buffers fill and models must be re-fitted."
        }
    ]

    with st.expander("📚 **Comprehensive Literature Glossary & Acronyms**", expanded=False):
        cols = st.columns(3)
        for i, it in enumerate(glossary_items):
            with cols[i % 3]:
                with st.container(border=True):
                    st.caption(it["citation"])
                    st.markdown(f"**{it['term']}** — *{it['name']}*")
                    st.caption(it["desc"])


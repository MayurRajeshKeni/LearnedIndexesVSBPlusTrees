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


def render_metric_card(title: str, value: str, subtext: str = "", color: str = "#2b5c8f") -> None:
    """Renders a styled metric card that adapts seamlessly to both light and dark themes."""
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {color};
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1));
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 10px;
        ">
            <div style="font-size: 0.85rem; opacity: 0.8; font-weight: 600;">{title}</div>
            <div style="font-size: 1.45rem; font-weight: 700; margin: 2px 0;">{value}</div>
            <div style="font-size: 0.75rem; opacity: 0.65;">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_info_cloud(title: str, items: list, icon: str = "☁️") -> None:
    """Renders a styled interactive Info Cloud container explaining 'What Changes What'."""
    cards_html = []
    colors = ["#2b5c8f", "#e26d5c", "#2a9d8f", "#9b5de5", "#f4a261", "#457b9d"]
    for i, it in enumerate(items):
        color = colors[i % len(colors)]
        knob = it.get("knob", "")
        effect = it.get("effect", "")
        detail = it.get("detail", "")
        category = it.get("category", "Knob")
        cards_html.append(f"""
        <div style="
            background: rgba(128, 128, 128, 0.07);
            border-left: 3.5px solid {color};
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.86rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; color: {color};">{knob}</span>
                <span style="font-size: 0.72rem; opacity: 0.8; background: rgba(128,128,128,0.18); padding: 1px 6px; border-radius: 4px;">{category}</span>
            </div>
            <div style="font-weight: 600; margin-bottom: 4px; font-size: 0.83rem;">➔ {effect}</div>
            <div style="font-size: 0.78rem; opacity: 0.8; line-height: 1.35;">{detail}</div>
        </div>
        """)

    grid_content = "".join(cards_html)
    full_html = f"""
    <div style="
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
    ">
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 0.96rem; margin-bottom: 12px;">
            <span>{icon}</span> <span>{title}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
            {grid_content}
        </div>
    </div>
    """
    st.markdown(full_html, unsafe_allow_html=True)


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

    cards_html = []
    colors = ["#2b5c8f", "#e26d5c", "#2a9d8f", "#9b5de5", "#f4a261", "#457b9d", "#6c757d", "#e63946"]
    for i, it in enumerate(glossary_items):
        color = colors[i % len(colors)]
        cards_html.append(f"""
        <div style="
            background: rgba(128, 128, 128, 0.07);
            border-left: 3.5px solid {color};
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.86rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2px;">
                <span style="font-weight: 700; color: {color}; font-size: 0.95rem;">{it['term']}</span>
                <span style="font-size: 0.72rem; opacity: 0.75; font-style: italic;">{it['citation']}</span>
            </div>
            <div style="font-weight: 600; font-size: 0.82rem; margin-bottom: 4px; opacity: 0.9;">{it['name']}</div>
            <div style="font-size: 0.78rem; opacity: 0.8; line-height: 1.35;">{it['desc']}</div>
        </div>
        """)

    grid_content = "".join(cards_html)
    full_html = f"""
    <div style="
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
    ">
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 0.98rem; margin-bottom: 12px;">
            <span>📚</span> <span>Comprehensive Literature Glossary & Acronyms</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px;">
            {grid_content}
        </div>
    </div>
    """
    st.markdown(full_html, unsafe_allow_html=True)

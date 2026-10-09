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

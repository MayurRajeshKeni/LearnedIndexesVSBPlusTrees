"""
Learned Indexes vs. B+ Trees: Interactive Educational Demonstration

Run with:
    streamlit run app.py
"""

import sys
import os
import streamlit as st

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.tab_concept import render_tab_concept
from ui.tab_race import render_tab_race
from ui.tab_dashboard import render_tab_dashboard
from ui.tab_data import render_tab_data
from ui.theme import DATASET_DESCRIPTIONS

st.set_page_config(
    page_title="Learned Indexes vs B+ Trees",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clean, professional presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0px;
        color: var(--text-color, inherit);
    }
    .sub-header {
        font-size: 1.05rem;
        opacity: 0.8;
        margin-bottom: 16px;
    }
    .method-badge {
        background-color: var(--secondary-background-color, rgba(49, 130, 206, 0.1));
        border-left: 4px solid #3182ce;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Banner
st.markdown('<div class="main-header">🌲 Learned Indexes vs. B+ Trees</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'An interactive comparison: <i>When do learned indexes beat B+ Trees (lookup speed, memory), '
    'and when do they lose (inserts, write-heavy workloads, hard distributions)?</i>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="method-badge">'
    '<b>Methodological Note:</b> Python runtime overhead blurs absolute speed differences. '
    'Observe <b>relative algorithmic trends</b>, memory compression, and retraining penalties, '
    'rather than claiming production C++ speedups.'
    '</div>',
    unsafe_allow_html=True
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Controls apply to **How It Works**, **Live Race**, and **Data Explorer** tabs.")

    dataset_name = st.selectbox(
        "Dataset Distribution:",
        options=["uniform", "lognormal", "clustered", "sequential_noise"],
        index=0,
        help="Select synthetic key distribution profile"
    )

    n_keys = st.slider(
        "Interactive Dataset Size (N):",
        min_value=100,
        max_value=20_000,
        value=2_000,
        step=100,
        help="Smaller sizes keep diagrams fast and visually legible"
    )

    st.markdown("---")
    st.subheader("Model Hyperparameters")

    rmi_m = st.slider(
        "RMI Stage-2 Models (M):",
        min_value=10,
        max_value=1_000,
        value=100,
        step=10,
        help="Number of sub-models in RMI Stage 2 (more models = narrower bounds)"
    )

    pgm_eps = st.slider(
        "PGM Error Bound (ε):",
        min_value=8,
        max_value=256,
        value=64,
        step=8,
        help="Max guaranteed prediction error for PGM (smaller ε = tighter search, more segments)"
    )

    seed = st.number_input("Random Seed:", value=42, step=1, help="Fixed seed for deterministic reproducibility")

    st.markdown("---")
    with st.expander("💡 What Changes What? (Knobs Guide)", expanded=False):
        st.markdown(
            """
            • **Dataset Distribution:** Dictates the shape of the CDF curve. Smooth diagonal lines (*Uniform*, *Sequential*) allow models to predict positions with near-zero error. Jagged curves with flat gaps (*Clustered*) cause large prediction errors. *B+ Tree is immune to distribution shape.*  
            • **Dataset Size (N):** Scales data volume. B+ Tree depth grows logarithmically ($O(\\log N)$). Learned models continue predicting direct array offsets.  
            • **RMI Models (M):** Higher $M$ divides data into finer slices $\\to$ smaller error bounds & smaller search windows, but requires more memory to store $M$ linear equations.  
            • **PGM Error (ε):** Smaller $\\epsilon$ guarantees a tighter search window ($2\\epsilon + 1$), but creates *more linear segments* (higher memory). Larger $\\epsilon$ maximizes compression into fewer segments at the cost of wider binary searches.  
            • **Random Seed:** Keeps dataset generation deterministic and reproducible across runs.
            """
        )

    with st.expander("📚 Acronyms & Literature Glossary", expanded=False):
        st.markdown(
            """
            • **RMI:** Recursive Model Index *[Kraska 2018]*  
            • **ALEX:** Updatable Adaptive Learned Extensible Index *(The name is a stylized combination of "Adaptive Learned" and "Extensible")* *[Ding 2020]*  
            • **PGM:** Piecewise Geometric Model *[Ferragina 2020]*  
            • **SOSD:** Search on Structured Data *[Marcus 2020]*  
            • **CDF:** Cumulative Distribution Function
            """
        )
    st.caption("Database Systems Benchmark")

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 1. How It Works",
    "🏎️ 2. Live Race",
    "📊 3. Benchmark Dashboard",
    "🔍 4. Data Explorer"
])

with tab1:
    render_tab_concept(dataset_name, n_keys, rmi_m, pgm_eps, seed)

with tab2:
    render_tab_race(dataset_name, n_keys, seed)

with tab3:
    render_tab_dashboard()

with tab4:
    render_tab_data(dataset_name, n_keys, seed)

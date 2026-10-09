"""
Plotting and Visualization Module

Generates publication-quality figures:
1. Lookup Latency comparison across datasets (mean and p99)
2. Memory Footprint comparison (log scale)
3. Insert Throughput and write penalties
4. Mixed Workload Throughput (100/0, 90/10, 50/50 read/write)
5. Sensitivity curves (RMI vs M, PGM vs ε)
6. Combined CDF comparison of datasets
"""

import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDEX_COLORS: Dict[str, str] = {
    "B+ Tree": "#2b5c8f",    # Classic Database Blue
    "RMI": "#e26d5c",        # Vibrant Coral
    "PGM": "#2a9d8f",        # Teal / Emerald
    "ALEX-lite": "#9b5de5",   # Violet / Purple
}


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        "figure.dpi": 200,
    })


def plot_lookup_latency(df: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart of lookup latency by index across datasets."""
    setup_plot_style()
    lat_df = df[
        (df["dataset"] != "sensitivity") &
        (df["metric"].isin(["lookup_mean_ns", "lookup_p99_ns"]))
    ]
    if lat_df.empty:
        return

    pivot_mean = lat_df[lat_df["metric"] == "lookup_mean_ns"].pivot(
        index="dataset", columns="index", values="value"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(pivot_mean.index))
    width = 0.18
    indexes = list(pivot_mean.columns)

    for i, idx in enumerate(indexes):
        color = INDEX_COLORS.get(idx, "#555555")
        vals = pivot_mean[idx].values
        rects = ax.bar(x + i * width, vals, width, label=idx, color=color, edgecolor="black", linewidth=0.5, alpha=0.9)
        # Value labels
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{int(h)}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_ylabel("Lookup Latency (ns) [Lower is Better]")
    ax.set_title("Point Lookup Latency by Dataset (Mean)")
    ax.set_xticks(x + width * (len(indexes) - 1) / 2)
    ax.set_xticklabels([d.replace("_", " ").title() for d in pivot_mean.index])
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Index Type", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_memory_footprint(df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of index memory footprint on log scale."""
    setup_plot_style()
    mem_df = df[
        (df["dataset"] != "sensitivity") &
        (df["metric"] == "memory_bytes")
    ]
    if mem_df.empty:
        return

    # Take uniform dataset as representative for memory
    uniform_mem = mem_df[mem_df["dataset"] == "uniform"].copy()
    if uniform_mem.empty:
        uniform_mem = mem_df.drop_duplicates(subset=["index"])

    uniform_mem["memory_kb"] = uniform_mem["value"] / 1024.0

    fig, ax = plt.subplots(figsize=(8, 5))
    indexes = uniform_mem["index"].tolist()
    kbs = uniform_mem["memory_kb"].tolist()
    colors = [INDEX_COLORS.get(idx, "#555555") for idx in indexes]

    bars = ax.bar(indexes, kbs, color=colors, edgecolor="black", linewidth=0.6, width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Index Overhead (KB, Log Scale) [Lower is Better]")
    ax.set_title("Index Memory Footprint Comparison (N=200,000 Keys)")
    ax.grid(axis="y", linestyle=":", alpha=0.7, which="both")

    for bar in bars:
        yval = bar.get_height()
        if yval >= 1024:
            txt = f"{yval / 1024.0:.1f} MB"
        else:
            txt = f"{yval:.1f} KB"
        ax.annotate(txt,
                    xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_insert_throughput(df: pd.DataFrame, output_path: str) -> None:
    """Insert throughput bar chart comparing B-tree to learned indexes."""
    setup_plot_style()
    ins_df = df[
        (df["dataset"] != "sensitivity") &
        (df["metric"] == "insert_throughput_ops_sec")
    ]
    if ins_df.empty:
        return

    pivot_ins = ins_df.pivot(index="dataset", columns="index", values="value")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(pivot_ins.index))
    width = 0.18
    indexes = list(pivot_ins.columns)

    for i, idx in enumerate(indexes):
        color = INDEX_COLORS.get(idx, "#555555")
        vals = pivot_ins[idx].values
        ax.bar(x + i * width, vals, width, label=idx, color=color, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Insert Throughput (ops/sec) [Higher is Better]")
    ax.set_title("Batch Insert Throughput Across Datasets")
    ax.set_xticks(x + width * (len(indexes) - 1) / 2)
    ax.set_xticklabels([d.replace("_", " ").title() for d in pivot_ins.index])
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Index Type", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_mixed_workloads(df: pd.DataFrame, output_path: str) -> None:
    """Throughput across read/write ratios: 100/0, 90/10, 50/50."""
    setup_plot_style()
    mixed_metrics = ["mixed_100_0_ops_sec", "mixed_90_10_ops_sec", "mixed_50_50_ops_sec"]
    m_df = df[
        (df["dataset"] == "uniform") &
        (df["metric"].isin(mixed_metrics))
    ]
    if m_df.empty:
        return

    workload_labels = ["100% Read\n0% Write", "90% Read\n10% Write", "50% Read\n50% Write"]
    metric_map = {m: l for m, l in zip(mixed_metrics, workload_labels)}
    m_df = m_df.copy()
    m_df["workload"] = m_df["metric"].map(metric_map)

    pivot_m = m_df.pivot(index="workload", columns="index", values="value").reindex(workload_labels)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(workload_labels))
    width = 0.18
    indexes = list(pivot_m.columns)

    for i, idx in enumerate(indexes):
        color = INDEX_COLORS.get(idx, "#555555")
        vals = pivot_m[idx].values
        ax.bar(x + i * width, vals, width, label=idx, color=color, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Total Throughput (ops/sec) [Higher is Better]")
    ax.set_title("Mixed Workload Throughput (Uniform Dataset)")
    ax.set_xticks(x + width * (len(indexes) - 1) / 2)
    ax.set_xticklabels(workload_labels)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Index Type", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_sensitivity_curves(df: pd.DataFrame, output_dir: str) -> None:
    """Sensitivity analysis plots: RMI vs M and PGM vs ε."""
    setup_plot_style()
    sens_df = df[df["dataset"] == "sensitivity"]
    if sens_df.empty:
        return

    # 1. RMI Sensitivity
    rmi_df = sens_df[sens_df["index"] == "RMI_M_sweep"]
    if not rmi_df.empty:
        p_rmi = rmi_df.pivot(index="config", columns="metric", values="value")
        # Ensure ordered by M
        if "M" in p_rmi.columns:
            p_rmi = p_rmi.sort_values(by="M")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        m_vals = p_rmi["M"].values
        lat_vals = p_rmi["lookup_mean_ns"].values
        mem_kbs = p_rmi["memory_bytes"].values / 1024.0

        ax1.plot(m_vals, lat_vals, marker="o", color=INDEX_COLORS["RMI"], linewidth=2)
        ax1.set_xscale("log")
        ax1.set_xlabel("Number of Stage-2 Models (M)")
        ax1.set_ylabel("Lookup Latency (ns)")
        ax1.set_title("RMI: Latency vs. Model Count (M)")
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.plot(m_vals, mem_kbs, marker="s", color="#b5179e", linewidth=2)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Number of Stage-2 Models (M)")
        ax2.set_ylabel("Index Memory (KB)")
        ax2.set_title("RMI: Memory vs. Model Count (M)")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sensitivity_rmi.png"))
        plt.close(fig)

    # 2. PGM Sensitivity
    pgm_df = sens_df[sens_df["index"] == "PGM_eps_sweep"]
    if not pgm_df.empty:
        p_pgm = pgm_df.pivot(index="config", columns="metric", values="value")
        if "epsilon" in p_pgm.columns:
            p_pgm = p_pgm.sort_values(by="epsilon")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        eps_vals = p_pgm["epsilon"].values
        lat_vals = p_pgm["lookup_mean_ns"].values
        seg_vals = p_pgm["num_segments"].values

        ax1.plot(eps_vals, lat_vals, marker="o", color=INDEX_COLORS["PGM"], linewidth=2)
        ax1.set_xlabel("Error Bound (ε)")
        ax1.set_ylabel("Lookup Latency (ns)")
        ax1.set_title("PGM: Latency vs. Error Bound (ε)")
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.plot(eps_vals, seg_vals, marker="^", color="#264653", linewidth=2)
        ax2.set_xlabel("Error Bound (ε)")
        ax2.set_ylabel("Segment Count (Compression)")
        ax2.set_title("PGM: Segments Created vs. Error Bound (ε)")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sensitivity_pgm.png"))
        plt.close(fig)


def generate_all_plots(csv_path: str, output_dir: str = "results") -> None:
    """Reads benchmark CSV and generates all required visualization artifacts."""
    if not os.path.exists(csv_path):
        print(f"[!] Results file {csv_path} not found. Skipping plot generation.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    print("[*] Generating benchmark charts...")
    plot_lookup_latency(df, os.path.join(output_dir, "lookup_latency.png"))
    plot_memory_footprint(df, os.path.join(output_dir, "memory_footprint.png"))
    plot_insert_throughput(df, os.path.join(output_dir, "insert_throughput.png"))
    plot_mixed_workloads(df, os.path.join(output_dir, "mixed_workloads.png"))
    plot_sensitivity_curves(df, output_dir)
    print("[+] All charts successfully generated and saved to results/")

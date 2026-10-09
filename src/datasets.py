import os
from typing import Tuple, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt


def generate_uniform(n_total: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform distribution: keys are spread evenly across the domain.
    The CDF is virtually a straight diagonal line. This is the optimal case for linear models.
    """
    # Sample from a wide domain to minimize collision and maintain realistic sparse integer IDs
    domain_max = max(int(n_total * 10), 1_000_000)
    # Generate unique keys
    keys = rng.choice(domain_max, size=n_total, replace=False).astype(np.int64)
    return keys


def generate_lognormal(n_total: int, rng: np.random.Generator) -> np.ndarray:
    """
    Lognormal distribution: heavily skewed, long right tail (inspired by SOSD benchmark).
    Challenges single linear regressions due to extreme density in the head and sparsity in the tail.
    """
    # Draw from lognormal, scale to integers
    raw = rng.lognormal(mean=2.0, sigma=1.5, size=n_total * 2)
    # Scale to integer domain
    scaled = (raw * 100_000).astype(np.int64)
    unique_keys = np.unique(scaled)
    while len(unique_keys) < n_total:
        more = (rng.lognormal(mean=2.0, sigma=1.5, size=n_total) * 100_000).astype(np.int64)
        unique_keys = np.unique(np.concatenate([unique_keys, more]))
    return unique_keys[:n_total]


def generate_clustered(n_total: int, rng: np.random.Generator) -> np.ndarray:
    """
    Clustered / Weblog-like distribution:
    Simulates bursts of activity (timestamps/server logs) with dense clusters separated by large gaps.
    Presents severe non-linearities and step-function-like behavior in the CDF.
    """
    num_clusters = 10
    cluster_centers = np.linspace(1_000_000, 100_000_000, num_clusters)
    cluster_std = 25_000

    keys_per_cluster = int(np.ceil((n_total * 2) / num_clusters))
    samples = []
    for center in cluster_centers:
        cluster_data = rng.normal(loc=center, scale=cluster_std, size=keys_per_cluster)
        samples.append(np.clip(cluster_data, 0, None).astype(np.int64))
    
    all_keys = np.concatenate(samples)
    unique_keys = np.unique(all_keys)
    if len(unique_keys) < n_total:
        extra = rng.choice(100_000_000, size=n_total - len(unique_keys), replace=False).astype(np.int64)
        unique_keys = np.unique(np.concatenate([unique_keys, extra]))
    return unique_keys[:n_total]


def generate_sequential_noise(n_total: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sequential with Noise:
    Represents auto-incrementing surrogate keys with occasional skips, deletions, or jitter.
    Nearly linear with mild local fluctuations.
    """
    base = np.arange(n_total * 2, dtype=np.int64) * 10
    noise = rng.integers(-3, 4, size=n_total * 2, dtype=np.int64)
    raw = base + noise
    # Enforce strictly monotonic unique keys
    unique_keys = np.unique(raw)
    if len(unique_keys) < n_total:
        unique_keys = np.arange(n_total, dtype=np.int64) * 10
    return unique_keys[:n_total]


DATASET_GENERATORS: Dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "uniform": generate_uniform,
    "lognormal": generate_lognormal,
    "clustered": generate_clustered,
    "sequential_noise": generate_sequential_noise,
}


def get_dataset(
    name: str,
    n: int,
    insert_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sorted unique train keys of size N, and a disjoint pool of insert keys of size int(N * insert_fraction).
    Both pools come from the exact same underlying distribution.
    """
    if name not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_GENERATORS.keys())}")

    rng = np.random.default_rng(seed)
    n_insert = max(1, int(n * insert_fraction))
    total_needed = n + n_insert

    raw_pool = DATASET_GENERATORS[name](total_needed, rng)
    # Shuffle to ensure unbiased split
    rng.shuffle(raw_pool)

    train_keys = np.sort(raw_pool[:n])
    insert_keys = raw_pool[n:n + n_insert]  # Kept in arbitrary/insertion arrival order

    return train_keys, insert_keys


def plot_dataset_cdf(keys: np.ndarray, name: str, output_path: str) -> None:
    """
    Plots the Cumulative Distribution Function (CDF) of keys.
    Visualizing the CDF is fundamental to understanding learned indexing:
    A linear index essentially learns to invert or approximate this exact curve.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(keys)
    # Subsample if N is large for plotting efficiency
    if n > 100_000:
        indices = np.linspace(0, n - 1, 100_000, dtype=np.int64)
        sample_keys = keys[indices]
        sample_ranks = indices / (n - 1)
    else:
        sample_keys = keys
        sample_ranks = np.linspace(0, 1, n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sample_keys, sample_ranks, label=f"CDF of {name}", color="#1f77b4", linewidth=2)

    # Reference diagonal line
    ax.plot(
        [sample_keys[0], sample_keys[-1]],
        [0.0, 1.0],
        linestyle="--",
        color="#7f7f7f",
        alpha=0.7,
        label="Ideal Linear CDF"
    )

    ax.set_title(f"Cumulative Distribution Function (CDF) - {name.capitalize()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Key Value", fontsize=11)
    ax.set_ylabel("Empirical CDF (Normalized Rank)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

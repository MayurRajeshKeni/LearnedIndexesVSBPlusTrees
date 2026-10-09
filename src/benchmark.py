"""
Benchmark Harness for Learned Indexes vs. B+ Trees

Measures:
1. Build time (seconds)
2. Lookup latency: warm-up, 3 repetitions, mean, p50, p99 (nanoseconds)
3. Negative lookup latency (keys not present)
4. Insert throughput (ops/sec), retrain count, retrain time
5. Memory footprint (bytes)
6. Prediction error statistics (mean error, max error, search window)
7. Mixed workloads (100/0, 90/10, 50/50 read/write)
8. Sensitivity sweeps (RMI vs M, PGM vs ε)
"""

import time
import os
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from src.datasets import get_dataset, DATASET_GENERATORS, plot_dataset_cdf
from src.btree_index import BTreeIndex
from src.rmi_index import RMIIndex
from src.pgm_index import PGMIndex
from src.alex_lite_index import AlexLiteIndex


class BenchmarkRunner:
    def __init__(
        self,
        n: int = 200_000,
        n_lookups: int = 50_000,
        insert_ratio: float = 0.1,
        seed: int = 42,
        output_dir: str = "results"
    ):
        self.n = n
        self.n_lookups = min(n_lookups, n)
        self.insert_ratio = insert_ratio
        self.seed = seed
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.records: List[Dict[str, Any]] = []

    def record(self, dataset: str, index_name: str, config: str, metric: str, value: Any) -> None:
        self.records.append({
            "dataset": dataset,
            "index": index_name,
            "config": config,
            "metric": metric,
            "value": value
        })

    def run_all(self, datasets: List[str] = None, include_alex: bool = True) -> pd.DataFrame:
        if datasets is None:
            datasets = ["uniform", "lognormal", "clustered", "sequential_noise"]

        print(f"=== Starting Benchmark Suite (N={self.n}, Lookups={self.n_lookups}, Seed={self.seed}) ===")

        # 1. Generate datasets & CDF plots
        cached_datasets = {}
        for dname in datasets:
            print(f"[*] Generating dataset: {dname}...")
            train_keys, insert_keys = get_dataset(dname, n=self.n, insert_fraction=self.insert_ratio, seed=self.seed)
            cached_datasets[dname] = (train_keys, insert_keys)
            cdf_path = os.path.join(self.output_dir, f"cdf_{dname}.png")
            plot_dataset_cdf(train_keys, dname, cdf_path)
            print(f"    Saved CDF plot to {cdf_path}")

        # 2. Main comparative benchmark across all datasets
        for dname in datasets:
            print(f"\n==========================================")
            print(f"Evaluating Dataset: {dname.upper()}")
            print(f"==========================================")
            train_keys, insert_keys = cached_datasets[dname]

            # Construct indexes for standard comparison
            indexes = [
                ("B+ Tree", "default", lambda: BTreeIndex()),
                ("RMI", "M=1000", lambda: RMIIndex(m=1000, buffer_ratio=0.01)),
                ("PGM", "eps=64", lambda: PGMIndex(epsilon=64, buffer_ratio=0.01)),
            ]
            if include_alex:
                indexes.append(("ALEX-lite", "default", lambda: AlexLiteIndex(initial_gap_ratio=0.5, max_density=0.8)))

            for idx_label, cfg_label, factory in indexes:
                self._evaluate_index(dname, idx_label, cfg_label, factory, train_keys, insert_keys)

        # 3. Sensitivity Sweeps
        print("\n==========================================")
        print("Running Sensitivity Analysis (Uniform)")
        print("==========================================")
        uniform_train, uniform_insert = cached_datasets["uniform"]
        self._run_sensitivity_rmi(uniform_train, uniform_insert)
        self._run_sensitivity_pgm(uniform_train, uniform_insert)

        df = pd.DataFrame(self.records)
        csv_path = os.path.join(self.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[+] Benchmark finished! Results saved to {csv_path}")
        return df

    def _evaluate_index(
        self,
        dataset: str,
        index_label: str,
        config: str,
        factory: Any,
        train_keys: np.ndarray,
        insert_keys: np.ndarray
    ) -> None:
        print(f"\n--> Index: {index_label} ({config}) on {dataset}")
        idx = factory()

        # 1. Build Timing
        t0 = time.perf_counter()
        idx.build(train_keys)
        build_time = time.perf_counter() - t0
        self.record(dataset, index_label, config, "build_time_s", build_time)
        print(f"    Build time: {build_time:.4f} s")

        # 2. Memory Footprint
        mem_bytes = idx.memory_bytes()
        self.record(dataset, index_label, config, "memory_bytes", mem_bytes)
        print(f"    Memory footprint: {mem_bytes:,} bytes ({mem_bytes / (1024 * 1024):.2f} MB)")

        # 3. Model Error Stats (if supported)
        if hasattr(idx, "get_error_stats"):
            mean_err, max_err, win_sz = idx.get_error_stats()
            self.record(dataset, index_label, config, "mean_error", mean_err)
            self.record(dataset, index_label, config, "max_error", max_err)
            self.record(dataset, index_label, config, "search_window_size", win_sz)
            print(f"    Error stats: mean={mean_err:.2f}, max={max_err}, window={win_sz:.1f}")

        if hasattr(idx, "num_segments"):
            self.record(dataset, index_label, config, "num_segments", idx.num_segments)
            print(f"    PGM segments: {idx.num_segments}")

        # 4. Lookup Latency (warm-up + 3 repetitions)
        rng = np.random.default_rng(self.seed + 1)
        query_keys = rng.choice(train_keys, size=self.n_lookups, replace=True)

        # Warm-up pass
        warmup_keys = query_keys[:min(5_000, len(query_keys))]
        for qk in warmup_keys:
            idx.lookup(int(qk))

        # 3 repetitions
        rep_means = []
        rep_p50s = []
        rep_p99s = []
        for rep in range(3):
            latencies_ns = []
            for qk in query_keys:
                t_start = time.perf_counter_ns()
                idx.lookup(int(qk))
                t_end = time.perf_counter_ns()
                latencies_ns.append(t_end - t_start)
            lat_arr = np.array(latencies_ns)
            rep_means.append(np.mean(lat_arr))
            rep_p50s.append(np.percentile(lat_arr, 50))
            rep_p99s.append(np.percentile(lat_arr, 99))

        median_mean = float(np.median(rep_means))
        median_p50 = float(np.median(rep_p50s))
        median_p99 = float(np.median(rep_p99s))

        self.record(dataset, index_label, config, "lookup_mean_ns", median_mean)
        self.record(dataset, index_label, config, "lookup_p50_ns", median_p50)
        self.record(dataset, index_label, config, "lookup_p99_ns", median_p99)
        print(f"    Lookup latency (ns): mean={median_mean:.1f}, p50={median_p50:.1f}, p99={median_p99:.1f}")

        # 5. Negative Lookups (absent keys)
        # Generate keys outside / interleaved that do not exist
        # Sample random integers not in train_keys
        k_min = int(train_keys[0])
        k_max = int(train_keys[-1])
        absent_candidates = rng.integers(max(0, k_min - 10000), k_max + 10000, size=5000, dtype=np.int64)
        absent_keys = [int(k) for k in absent_candidates if idx.lookup(int(k)) is None][:2000]
        if absent_keys:
            neg_latencies = []
            for ak in absent_keys:
                t_start = time.perf_counter_ns()
                idx.lookup(ak)
                t_end = time.perf_counter_ns()
                neg_latencies.append(t_end - t_start)
            neg_mean = float(np.mean(neg_latencies))
            self.record(dataset, index_label, config, "negative_lookup_mean_ns", neg_mean)
            print(f"    Negative lookup mean: {neg_mean:.1f} ns")

        # 6. Insert Throughput & Retraining overhead
        # Fresh instance for clean insert benchmark
        idx_insert = factory()
        idx_insert.build(train_keys)

        t0 = time.perf_counter()
        for i, ik in enumerate(insert_keys):
            idx_insert.insert(int(ik), int(i + 10_000_000))
        insert_time = time.perf_counter() - t0
        insert_ops_sec = len(insert_keys) / max(1e-6, insert_time)

        self.record(dataset, index_label, config, "insert_throughput_ops_sec", insert_ops_sec)
        retrain_cnt = getattr(idx_insert, "retrain_count", 0)
        retrain_sec = getattr(idx_insert, "total_retrain_time_sec", 0.0)
        self.record(dataset, index_label, config, "retrain_count", retrain_cnt)
        self.record(dataset, index_label, config, "retrain_time_s", retrain_sec)
        print(f"    Insert throughput: {insert_ops_sec:.1f} ops/sec (retrains={retrain_cnt}, retrain_time={retrain_sec:.3f}s)")

        # 7. Mixed Workloads (100/0, 90/10, 50/50 read/write)
        for read_pct, write_pct in [(100, 0), (90, 10), (50, 50)]:
            self._benchmark_mixed_workload(
                dataset, index_label, config, factory, train_keys, insert_keys, read_pct, write_pct
            )

    def _benchmark_mixed_workload(
        self,
        dataset: str,
        index_label: str,
        config: str,
        factory: Any,
        train_keys: np.ndarray,
        insert_keys: np.ndarray,
        read_pct: int,
        write_pct: int
    ) -> None:
        num_ops = min(10_000, len(insert_keys))
        rng = np.random.default_rng(self.seed + read_pct)

        # Build clean index
        idx = factory()
        idx.build(train_keys)

        # Plan operations: 0 = read, 1 = write
        reads_count = int(num_ops * (read_pct / 100.0))
        writes_count = num_ops - reads_count

        op_types = np.array([0] * reads_count + [1] * writes_count)
        rng.shuffle(op_types)

        read_targets = rng.choice(train_keys, size=reads_count, replace=True)
        write_targets = insert_keys[:writes_count]

        r_ptr = 0
        w_ptr = 0

        t0 = time.perf_counter()
        for op in op_types:
            if op == 0:
                idx.lookup(int(read_targets[r_ptr]))
                r_ptr += 1
            else:
                idx.insert(int(write_targets[w_ptr]), int(w_ptr + 500_000))
                w_ptr += 1
        elapsed = time.perf_counter() - t0
        throughput = num_ops / max(1e-6, elapsed)

        metric_name = f"mixed_{read_pct}_{write_pct}_ops_sec"
        self.record(dataset, index_label, config, metric_name, throughput)
        print(f"    Mixed {read_pct}/{write_pct} throughput: {throughput:.1f} ops/sec")

    def _run_sensitivity_rmi(self, train_keys: np.ndarray, insert_keys: np.ndarray) -> None:
        """Sweep RMI across M in {100, 1000, 10000}."""
        print("[*] RMI Sensitivity Sweep over M in {100, 1000, 10000}...")
        for m_val in [100, 1000, 10000]:
            cfg = f"M={m_val}"
            factory = lambda m=m_val: RMIIndex(m=m, buffer_ratio=0.01)
            idx = factory()
            t0 = time.perf_counter()
            idx.build(train_keys)
            b_time = time.perf_counter() - t0

            mem = idx.memory_bytes()
            mean_err, max_err, win_sz = idx.get_error_stats()

            # Measure lookup latency
            rng = np.random.default_rng(self.seed)
            query_keys = rng.choice(train_keys, size=min(20_000, len(train_keys)))
            t_start = time.perf_counter()
            for k in query_keys:
                idx.lookup(int(k))
            lookup_lat_ns = ((time.perf_counter() - t_start) / len(query_keys)) * 1e9

            self.record("sensitivity", "RMI_M_sweep", cfg, "M", m_val)
            self.record("sensitivity", "RMI_M_sweep", cfg, "build_time_s", b_time)
            self.record("sensitivity", "RMI_M_sweep", cfg, "memory_bytes", mem)
            self.record("sensitivity", "RMI_M_sweep", cfg, "lookup_mean_ns", lookup_lat_ns)
            self.record("sensitivity", "RMI_M_sweep", cfg, "mean_error", mean_err)
            self.record("sensitivity", "RMI_M_sweep", cfg, "max_error", max_err)
            self.record("sensitivity", "RMI_M_sweep", cfg, "search_window_size", win_sz)
            print(f"    M={m_val}: Latency={lookup_lat_ns:.1f}ns, Mem={mem:,}B, MeanErr={mean_err:.2f}, Win={win_sz:.1f}")

    def _run_sensitivity_pgm(self, train_keys: np.ndarray, insert_keys: np.ndarray) -> None:
        """Sweep PGM across epsilon in {16, 64, 256}."""
        print("[*] PGM Sensitivity Sweep over epsilon in {16, 64, 256}...")
        for eps_val in [16, 64, 256]:
            cfg = f"eps={eps_val}"
            factory = lambda e=eps_val: PGMIndex(epsilon=e, buffer_ratio=0.01)
            idx = factory()
            t0 = time.perf_counter()
            idx.build(train_keys)
            b_time = time.perf_counter() - t0

            mem = idx.memory_bytes()
            segs = idx.num_segments
            mean_err, max_err, win_sz = idx.get_error_stats()

            rng = np.random.default_rng(self.seed)
            query_keys = rng.choice(train_keys, size=min(20_000, len(train_keys)))
            t_start = time.perf_counter()
            for k in query_keys:
                idx.lookup(int(k))
            lookup_lat_ns = ((time.perf_counter() - t_start) / len(query_keys)) * 1e9

            self.record("sensitivity", "PGM_eps_sweep", cfg, "epsilon", eps_val)
            self.record("sensitivity", "PGM_eps_sweep", cfg, "build_time_s", b_time)
            self.record("sensitivity", "PGM_eps_sweep", cfg, "memory_bytes", mem)
            self.record("sensitivity", "PGM_eps_sweep", cfg, "num_segments", segs)
            self.record("sensitivity", "PGM_eps_sweep", cfg, "lookup_mean_ns", lookup_lat_ns)
            self.record("sensitivity", "PGM_eps_sweep", cfg, "mean_error", mean_err)
            self.record("sensitivity", "PGM_eps_sweep", cfg, "max_error", max_err)
            self.record("sensitivity", "PGM_eps_sweep", cfg, "search_window_size", win_sz)
            print(f"    ε={eps_val}: Latency={lookup_lat_ns:.1f}ns, Mem={mem:,}B, Segments={segs}, Win={win_sz:.1f}")

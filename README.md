# Learned Indexes vs. B+ Trees: Empirical Benchmark Suite

An educational, reproducible database systems benchmark comparing traditional **B+ Trees** against modern **Learned Index Structures** (**Recursive Model Index (RMI)**, **Piecewise Geometric Model Index (PGM)**, and **ALEX-lite**).

> [!IMPORTANT]
> **Key Trade-off & Methodological Limitation: Relative Trends vs. Absolute Speedups**  
> Because this benchmark is implemented in Python, runtime interpreter overhead (bytecode interpretation, dynamic dispatch, and object boxing) blurs absolute speed differences compared to compiled systems. In high-performance C++ implementations, mathematical model inference takes ~2–5 ns (just 2 CPU instructions like `FMADD` running directly in registers), whereas in Python function call overhead alone adds 200–800 ns per operation.  
> **Therefore, you should compare relative trends, memory compression factors, and scaling trade-offs (e.g., write penalties and sensitivity curves), and NOT claim real-world production speedups based on these Python numbers.**

---

## 1. Quickstart & Exact Reproduction

To run the full suite from a clean environment:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline (Correctness Tests -> Benchmarks -> Charts -> Report)
python run_all.py

# 3. Launch the Interactive Streamlit Web UI
streamlit run app.py

# Optional: Run a fast smoke test (N=20,000 keys)
python run_all.py --quick

# Optional: Custom dataset size (e.g., 1,000,000 keys)
python run_all.py --n 1000000 --lookups 100000
```

### 1.1 Exploring the Interactive Streamlit UI (`app.py`)

Run `streamlit run app.py` to open the visual presentation suite with four dedicated tabs:
- **🧠 Tab 1 (How It Works):** Side-by-side visual walkthrough. The left panel renders a live B+ Tree diagram highlighting every visited routing node and leaf; the right panel renders the mathematical CDF curve with regression lines, error bands ($\pm\epsilon$), and the bounded search window. Includes a zoomed-in probe viewer and beginner explanations.
- **🏎️ Tab 2 (Live Race):** Live head-to-head execution of lookups, inserts, and mixed workloads. Plots cumulative execution time to make retraining spikes (the write penalty) directly visible, displays delta buffer fill gauges, and runs silent post-workload correctness verification.
- **📊 Tab 3 (Benchmark Dashboard):** Interactive analysis of pre-computed benchmark results (`results/results.csv`). Features multi-metric filtering, a performance scoreboard, sensitivity curves ($M$ and $\epsilon$), CSV exports, and dynamic validation testing whether results match published literature.
- **🔍 Tab 4 (Data Explorer):** Explores data distributions through interactive CDF curves and gap histograms. Shows why smooth distributions yield compact models while clustered step-functions degrade accuracy, with a computed suitability verdict per dataset.

---

## 2. Executive Overview: What Is a Learned Index?

In traditional database systems, an index (such as a B+ Tree) is a search structure composed of balanced tree nodes filled with pointers. When searching for a key, the database traverses from the root node to internal nodes, and finally to leaf pages, following pointers in memory:

```
Traditional B+ Tree:
[Root: Keys & Pointers] ---> [Internal Nodes] ---> [Leaf Pages: Data Pointers]
(Multiple pointer hops, CPU cache misses, ~15-20% memory footprint tax)
```

In 2018, Kraska et al. (MIT & Google) introduced a groundbreaking idea:
> **An index is simply a function that predicts where a key is located in a sorted array.**
> If you think of your data as points along a curve, finding a key's position is equivalent to approximating the **Cumulative Distribution Function (CDF)** of the data:
> $$\text{Position} = \text{CDF}(key) \times N$$

Instead of allocating millions of pointer nodes, we can train small, lightweight mathematical models (like linear regression: $y = w \cdot x + b$) that compute the address directly:

```
Learned Index (RMI / PGM):
Key ----[ Mathematical Model: y = w*x + b ]----> Predicted Pos +/- Error Window
(Pure CPU arithmetic, registers, >95% smaller memory footprint)
```

---

## 3. Explaining Each Index (For a Smart Beginner)

### 3.1 B+ Tree Baseline (`src/btree_index.py`)
- **How it works:** A balanced search tree where every internal node holds keys and child pointers, and leaf nodes contain data pointers linked horizontally for sequential scanning.
- **Implementation Note:** Implemented using `BTrees.OOBTree` (with `sortedcontainers.SortedDict` as fallback), a production-grade C-extension B-tree library standing in for a B+ Tree in Python.
- **Strengths:** Robust, predictable $O(\log n)$ operations regardless of data distribution. Excellent for dynamic inserts via local page splits.
- **Weaknesses:** High pointer overhead (in our tests, ~20 MB for 200,000 keys) and pointer chasing causing CPU cache misses.

### 3.2 Recursive Model Index - RMI (`src/rmi_index.py`)
- **How it works (Kraska et al. 2018):** Instead of one massive deep neural network (which would be too slow to evaluate), RMI uses a 2-stage hierarchy of simple linear models:
  1. **Stage 1 (Root):** A linear model that maps any key into one of $M$ stage-2 models ($M=1000$ by default).
  2. **Stage 2 (Leaf Models):** $M$ small linear models, each trained strictly on the keys routed to it. It predicts the exact sorted array position: $\text{pred} = w \cdot \text{key} + b$.
  3. **Error Bounds:** During training, each stage-2 model records its worst-case training errors: $[min\_err, max\_err]$.
  4. **Lookup:** The index calculates $\text{pred}$, then conducts a narrow binary search strictly inside $[\text{pred} + min\_err, \text{pred} + max\_err]$.
  5. **Dynamic Writes:** Uses a sorted **delta buffer**. New inserts enter the buffer. When the buffer exceeds 1% of $N$, the index merges and **retrains** the entire model hierarchy.

### 3.3 Piecewise Geometric Model - PGM (`src/pgm_index.py`)
- **How it works (Ferragina & Vinciguerra 2020):** Instead of fixing $M$ models upfront, PGM constructs piecewise linear segments with a **guaranteed error bound** $\pm \epsilon$ (default $\epsilon = 64$):
  1. **Greedy Shrinking Cone ($O(n)$):** Starts a linear segment at $(key_0, pos_0)$. As new keys are examined, it calculates the allowable slope range $[s_{low}, s_{high}]$ that keeps every single key within $\pm \epsilon$. The cone narrows with each key until $s_{low} > s_{high}$, at which point it finalizes the segment and starts a new one.
  2. **Lookup:** Binary searches the segments' start keys in $O(\log(\text{segments}))$, evaluates the segment's line equation, and performs binary search inside the guaranteed window $[\text{pred} - \epsilon, \text{pred} + \epsilon]$ (at most $2\epsilon = 128$ keys!).
  3. **Compression:** Smooth data produces very few segments (high compression); erratic data produces more segments.

### 3.4 ALEX-lite Updatable Learned Index (`src/alex_lite_index.py`)
- **How it works (Ding et al. 2020):** An educational simplification of the Adaptive Learned Index (ALEX):
  1. **Gapped Array:** Allocates an array with 30–50% intentional empty slots interspersed among active elements.
  2. **Direct Inserts:** Uses model prediction to place new keys directly into nearby empty gaps with minimal element shifting.
  3. **Adaptive Expansion:** When array density exceeds 80%, doubles capacity, redistributes keys with fresh gaps, and retrains the routing model.

---

## 4. Benchmark Datasets (`src/datasets.py`)

1. **Uniform:** Keys distributed evenly across the integer domain. CDF is virtually a straight diagonal line. The ideal case for linear regression.
2. **Lognormal:** Skewed distribution with a heavy right tail (matching the SOSD benchmark). Dense head, sparse tail.
3. **Clustered (Weblog / Timestamps):** Several dense bursts separated by large gaps. Produces step-function-like CDFs that challenge linear approximations.
4. **Sequential with Noise:** Auto-incrementing IDs with local jitter. Nearly linear.

Each dataset's empirical CDF is visualized in `results/cdf_<name>.png`.

---

## 5. How to Read Each Benchmark Chart

All charts are saved in high resolution under `results/`:

1. **`results/lookup_latency.png` (Grouped Bar Chart):**
   - **X-axis:** Dataset distribution (Clustered, Lognormal, Sequential Noise, Uniform).
   - **Y-axis:** Mean point lookup latency in nanoseconds (lower is better).
   - **Key takeaway:** Learned indexes (RMI, PGM, ALEX-lite) consistently beat the B+ Tree on lookup latency because arithmetic calculation in CPU registers is faster than traversing tree pointer nodes.

2. **`results/memory_footprint.png` (Log-Scale Bar Chart):**
   - **X-axis:** Index structure.
   - **Y-axis:** Index metadata overhead in KB on a logarithmic scale.
   - **Key takeaway:** The B+ Tree consumes ~20.6 MB, while RMI consumes only 32 KB and PGM consumes < 1 KB (over a **99.8% memory reduction**!).

3. **`results/insert_throughput.png` (Grouped Bar Chart):**
   - **X-axis:** Dataset distribution.
   - **Y-axis:** Batch insert throughput in operations per second (higher is better).
   - **Key takeaway:** Demonstrates the "write penalty": learned indexes using delta buffers (RMI, PGM) experience periodic retraining pauses, whereas B+ Trees absorb inserts smoothly.

4. **`results/mixed_workloads.png` (Throughput Comparison):**
   - **Workloads:** 100% Read / 0% Write, 90% Read / 10% Write, 50% Read / 50% Write.
   - **Key takeaway:** In read-dominated workloads (100/0), learned indexes dominate. As write traffic increases (50/50), the write penalty degrades throughput for buffer-retrained indexes.

5. **`results/sensitivity_rmi.png` (RMI Trade-off Curves):**
   - Plots lookup latency and memory overhead against model count $M \in \{100, 1000, 10000\}$.
   - More models narrow the search window, but increase metadata memory footprint.

6. **`results/sensitivity_pgm.png` (PGM Trade-off Curves):**
   - Plots lookup latency and segment count against error bound $\epsilon \in \{16, 64, 256\}$.
   - A larger $\epsilon$ compresses the index into fewer segments, but widens the binary search window.

---

## 6. Memory Estimation Methodology

Memory numbers reported in this benchmark are **approximate index overheads**:
- **Learned Indexes (Analytical Calculation):**
  - RMI: $M \times (\text{slope} + \text{intercept} + \text{min\_err} + \text{max\_err} + \text{flag}) + \text{delta\_buffer}$.
  - PGM: $\text{num\_segments} \times (\text{key} + \text{pos} + \text{slope}) + \text{delta\_buffer}$.
  - ALEX-lite: Empty slot overhead + occupancy mask + model parameters.
- **B+ Tree (Heap Traversal):**
  - Measured via `pympler.asizeof` on the underlying `OOBTree` object, capturing internal node buckets and pointers.

---

## 7. Assumptions and Limitations

- **Python Interpreter Overhead Blurs Absolute Speed Differences:**  
  In production C++ database engines (such as the original SOSD, RMI, and PGM implementations), linear regression inference runs in 2–5 nanoseconds via vectorized CPU instructions (`FMADD`) with hardware prefetching and zero allocation. In pure Python, interpreter bytecode dispatch, dynamic attribute resolution, and object memory wrapping contribute 200–800 nanoseconds of baseline overhead to every lookup call.  
  **Trade-off consequence:** This runtime overhead blurs the absolute speed differences between indexes. Readers and evaluators must **compare relative algorithmic trends** (such as scaling curves, write penalties under mixed workloads, and memory compression ratios) rather than claiming real-world production speedups.
- **Simplified Index Architectures:**  
  RMI, PGM, and ALEX-lite in this project are faithful educational reference models designed for clarity and correctness:
  - RMI uses a 2-stage linear regression hierarchy with min/max bounds and a delta buffer rather than multi-stage spline/neural optimizers.
  - PGM implements single-level piecewise linear shrinking-cone segmentation rather than a multi-level recursive PGM-tree.
  - ALEX-lite implements a single-node gapped array with dynamic expansions rather than the complete C++ ALEX engine with cost-driven tree node splits.
- **Library B-Tree Variant:**  
  The B+ Tree baseline uses `BTrees.OOBTree`, a battle-tested C-extension in-memory B-Tree library. Memory overhead is approximated via recursive heap traversal using `pympler.asizeof`.

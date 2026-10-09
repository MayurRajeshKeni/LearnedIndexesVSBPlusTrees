# Project Runbook: Learned Indexes vs. B+ Trees

A complete operational guide for setting up, running, testing, benchmarking, and troubleshooting the project across **Windows**, **macOS**, and **Linux**.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Environment Setup (All Operating Systems)](#2-environment-setup-all-operating-systems)
3. [Installing Dependencies](#3-installing-dependencies)
4. [Running Correctness Tests (pytest)](#4-running-correctness-tests-pytest)
5. [Running the Automated Benchmark Pipeline (run_all.py)](#5-running-the-automated-benchmark-pipeline-run_allpy)
6. [Launching the Interactive Web UI (Streamlit)](#6-launching-the-interactive-web-ui-streamlit)
7. [Inspecting Benchmark Results and Artifacts](#7-inspecting-benchmark-results-and-artifacts)
8. [Command Cheat Sheet (Quick Reference)](#8-command-cheat-sheet-quick-reference)
9. [Troubleshooting & Platform-Specific Gotchas](#9-troubleshooting--platform-specific-gotchas)

---

## 1. Prerequisites

Before running any commands, ensure you have:
- **Python 3.10, 3.11, 3.12, 3.13, or 3.14** installed.
- **Git** (optional, for version control).
- Terminal access:
  - **Windows:** PowerShell, Windows Terminal, or Command Prompt (cmd.exe).
  - **macOS:** Terminal or iTerm2 (zsh/bash).
  - **Linux:** Any standard shell (bash, zsh).

Check your Python version:
```bash
# Windows
python --version

# macOS / Linux
python3 --version
```

---

## 2. Environment Setup (All Operating Systems)

It is strongly recommended to use a dedicated virtual environment (`.venv`) so project libraries do not conflict with system packages.

### 🪟 Windows (PowerShell)
```powershell
# 1. Navigate to the project folder
cd C:\path\to\DBMS_B+Trees

# 2. Create the virtual environment
python -m venv .venv

# 3. If PowerShell blocks script activation, permit it for this session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### 🪟 Windows (Command Prompt - cmd.exe)
```cmd
cd C:\path\to\DBMS_B+Trees
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 🍎 macOS (zsh / bash)
```bash
# 1. Navigate to the project folder
cd /path/to/DBMS_B+Trees

# 2. Create the virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate
```

### 🐧 Linux (Ubuntu, Debian, Fedora, Arch)
```bash
# 1. Navigate to the project folder
cd /path/to/DBMS_B+Trees

# 2. Ensure python3-venv is installed (Debian/Ubuntu only)
sudo apt-get update && sudo apt-get install -y python3-venv

# 3. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

> **Verification:** Once activated, your terminal prompt will be prefixed with `(.venv)`.

---

## 3. Installing Dependencies

Install all required numerical, database, visualization, and testing libraries.

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### What gets installed:
- `numpy`: Fast vectorized array operations and numerical routines.
- `pandas`: Data structures and tabular aggregation.
- `matplotlib`: Publication-quality static chart generation.
- `pytest`: Automated correctness gate testing.
- `sortedcontainers`: Pure-Python sorted dictionary (high-performance fallback).
- `BTrees`: Production C-extension B-Tree library (`BTrees.OOBTree`).
- `pympler`: Deep heap inspection for honest B-Tree memory measurement.
- `streamlit`: Interactive web UI presentation framework.
- `plotly`: Responsive interactive charts.

---

## 4. Running Correctness Tests (pytest)

The project includes an uncompromising correctness gate (`tests/test_correctness.py` and `tests/test_traces.py`). Every index must pass this before any benchmark timing is considered valid.

### 4.1 Run All 36 Test Cases (Standard)
```bash
# Universal command (works on all platforms)
python -m pytest tests/
```

### 4.2 Run with Verbose Output (Per-Test Details)
```bash
python -m pytest -v tests/
```
*Shows individual test names, parameters (e.g. `uniform`, `lognormal`), and execution times.*

### 4.3 Run Only the Core Index Correctness Suite
```bash
python -m pytest -v tests/test_correctness.py
```
*Verifies: bulk build integrity, boundary lookups, absent key detection (returns `None`), duplicate key replacement, and tiny edge datasets ($N=1$ and $N=2$).*

### 4.4 Run Only the UI Trace Verification Suite
```bash
python -m pytest -v tests/test_traces.py
```
*Verifies: `lookup_trace(key)` consistency with `lookup(key)` for RMI, PGM, and ALEX-lite, as well as `VizBTree` node splitting and Graphviz DOT graph generation.*

### 4.5 Stop Immediately on First Failure (Debugging)
```bash
python -m pytest -x tests/
```

---

## 5. Running the Automated Benchmark Pipeline (run_all.py)

`run_all.py` is the **single entry-point orchestrator**. It executes in four sequential phases:
1. **Gate:** Runs `pytest` suite and halts immediately if any test fails.
2. **Benchmark:** Evaluates B+ Tree, RMI, PGM, and ALEX-lite across all 4 datasets + sweeps.
3. **Charts:** Renders 11 publication-grade PNG charts to `results/`.
4. **Report:** Generates `REPORT.md` compiling measured performance numbers.

### 5.1 Standard Full Run (Default: N=200,000 Keys)
```bash
python run_all.py
```
*Duration: ~1 to 2 minutes on modern laptops.*  
*Default parameters: $N = 200,000$ keys, $50,000$ random lookups, seed = 42.*

### 5.2 Quick Smoke Test Run (Fast: N=20,000 Keys)
```bash
python run_all.py --quick
```
*Duration: ~15 to 20 seconds.*  
*Use when verifying code changes rapidly without waiting for full benchmark iterations.*

### 5.3 Custom Scale Run (e.g. N=1,000,000 Keys)
```bash
python run_all.py --n 1000000 --lookups 100000
```
*Executes enterprise-scale benchmark at 1 million records.*

### 5.4 Custom Random Seed
```bash
python run_all.py --seed 999
```
*Ensures exact deterministic reproducibility under a custom random seed.*

### 5.5 Skip Tests (Direct Benchmark)
```bash
python run_all.py --skip-tests
```
*Bypasses the initial pytest step if you have already verified tests.*

---

## 6. Launching the Interactive Web UI (Streamlit)

The Streamlit UI provides an interactive classroom presentation suite across four tabs.

### 6.1 Standard Launch
```bash
streamlit run app.py
```
*Or using the Python module syntax (recommended if `streamlit` is not on your shell PATH):*
```bash
python -m streamlit run app.py
```

### What happens when you run this:
1. Streamlit starts a local web server (typically at `http://localhost:8501`).
2. Your default web browser opens automatically to the app.

### 6.2 Running on a Specific Port
If port 8501 is already in use by another service:
```bash
streamlit run app.py --server.port 8502
```

### 6.3 Running in Headless Mode (Remote Server / SSH / Docker)
If running on a remote cloud VM without a desktop display:
```bash
streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```
*You can then access the UI in your browser via `http://<server-ip>:8501`.*

---

## 7. Inspecting Benchmark Results and Artifacts

After running `python run_all.py`, the following files are automatically produced:

### 7.1 Generated Files Structure
```
results/
├── results.csv                # Tidy tabular metrics for every run
├── lookup_latency.png         # Grouped bar chart of lookup latency
├── memory_footprint.png       # Log-scale chart of index memory overhead
├── insert_throughput.png      # Batch insert throughput across datasets
├── mixed_workloads.png        # Throughput at 100/0, 90/10, 50/50 read/write
├── sensitivity_rmi.png        # RMI trade-off curves: latency vs M models
├── sensitivity_pgm.png        # PGM trade-off curves: segments vs epsilon
├── cdf_uniform.png            # CDF plot: Uniform dataset
├── cdf_lognormal.png          # CDF plot: Lognormal dataset
├── cdf_clustered.png          # CDF plot: Clustered dataset
└── cdf_sequential_noise.png   # CDF plot: Sequential with noise
REPORT.md                      # Auto-generated markdown report with empirical table
```

### 7.2 Viewing the Report in Terminal

#### Windows (PowerShell)
```powershell
Get-Content REPORT.md -TotalCount 50
```

#### macOS / Linux
```bash
head -n 50 REPORT.md
# Or view with pager
less REPORT.md
```

### 7.3 Inspecting the Results CSV via Python CLI
```bash
python -c "import pandas as pd; df = pd.read_csv('results/results.csv'); print(df.head(15))"
```

---

## 8. Command Cheat Sheet (Quick Reference)

| Task | Universal Command |
| :--- | :--- |
| **Install dependencies** | `pip install -r requirements.txt` |
| **Run all tests** | `python -m pytest tests/` |
| **Run tests verbosely** | `python -m pytest -v tests/` |
| **Run smoke benchmark (N=20k)** | `python run_all.py --quick` |
| **Run full benchmark (N=200k)** | `python run_all.py` |
| **Run 1M keys benchmark** | `python run_all.py --n 1000000 --lookups 100000` |
| **Launch Streamlit Web UI** | `streamlit run app.py` |
| **Launch UI (module syntax)** | `python -m streamlit run app.py` |
| **Clean cache directories** | `python -c "import shutil, os; [shutil.rmtree(d, ignore_errors=True) for d in ['.pytest_cache', '__pycache__']]"` |

---

## 9. Troubleshooting & Platform-Specific Gotchas

### Issue 1: PowerShell Script Execution Policy (`Activate.ps1 cannot be loaded`)
- **Symptom:** `File ...\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Fix:** Run this once in your PowerShell window:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  Then reactivate: `.\.venv\Scripts\Activate.ps1`.

### Issue 2: `pytest` or `streamlit` is not recognized as a command
- **Symptom:** `The term 'pytest' is not recognized as the name of a cmdlet...` or `command not found: streamlit`.
- **Cause:** The virtual environment scripts directory is not added to the system `PATH`.
- **Fix:** Prefix the command with `python -m`:
  ```bash
  python -m pytest tests/
  python -m streamlit run app.py
  ```

### Issue 3: C++ Compiler Missing for `BTrees` Installation
- **Symptom:** `error: Microsoft Visual C++ 14.0 or greater is required` during `pip install BTrees`.
- **Cause:** Missing pre-compiled wheel on certain esoteric platforms.
- **Fix:** The codebase automatically detects this and falls back gracefully to `sortedcontainers.SortedDict`, so the benchmark and UI will continue to function without errors.

### Issue 4: Streamlit Port Already in Use
- **Symptom:** `Port 8501 is already in use.`
- **Fix:** Specify a different port explicitly:
  ```bash
  streamlit run app.py --server.port 8505
  ```

### Issue 5: Windows Console Unicode Encoding (`charmap codec can't encode...`)
- **Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character...`
- **Fix:** The project automatically reconfigures `sys.stdout` to UTF-8 on Windows. If running external scripts, you can enforce UTF-8 in PowerShell by running:
  ```powershell
  $OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
  ```

# TensorForge 🔥

**An Interactive, Professional-Grade "LeetCode-Style" Educational Framework for Scientific Computing, Deep Learning, and GPU Kernel Engineering.**

---

## 🌟 Overview

Welcome to **TensorForge** — the premier laboratory training ground designed to bridge the gap between academic theory and production-grade systems engineering. Whether you are mastering high-performance NumPy vectorization, building deep learning architectures from scratch in PyTorch, or authoring bare-metal CUDA kernels with shared-memory tiling, TensorForge gives you an instant, automated verification feedback loop.

Unlike standard tutorial repositories, TensorForge enforces strict industry best practices:

- 🚫 **No Explicit Loops:** Automated Abstract Syntax Tree (AST) inspection forbids explicit Python loops (`for` / `while`) where vectorized C/Fortran routines exist.
- ⚡ **Real-Time Benchmarking & Memory Profiling:** Every solution is evaluated for both numerical precision and runtime efficiency against golden standard reference implementations, monitoring peak RAM and VRAM consumption down to the kilobyte (`KB`).
- 🎮 **Performance Gamification:** Compare your execution speeds against optimal baselines and earn colorized performance badges from `[Forge God / Chad]` down to `[Soy dev]`.
- 🏗️ **Clean Lab Architecture:** Strict physical separation between student scratchpads, automated validation suites, and immutable golden baselines eliminates tampering and guarantees test integrity.

---

## 📋 Setup and Installation Guide

TensorForge uses [uv](https://docs.astral.sh/uv/) as its ultra-fast, reproducible package and environment manager. Follow these simple steps to set up your environment in seconds:

### Step 1: Install System Prerequisites

Ensure your operating system is macOS, Linux, or WSL2 (Windows Subsystem for Linux), and that you have **Python 3.12+** installed.

Install `uv` (if not already installed) using the official standalone installer:

```bash
# On macOS / Linux / WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Clone & Run Bootstrap

Clone the repository and run the interactive multi-platform bootstrap script. The bootstrap script automates environment creation, dependency resolution, and hardware detection:

```bash
# 1. Clone the repository
git clone https://github.com/nguyenpanda/tensorforge.git
cd tensorforge

# 2. Run the interactive bootstrap script
uv run bootstrap.py

# 3. Validate system health and baseline integrity
uv run tforge validate
```

### 🧠 What `bootstrap.py` Automates For You:

- **Hardware & GPU Discovery:** Automatically detects NVIDIA GPUs via `nvidia-smi` or Apple Silicon MPS architectures, configuring PyTorch wheels with exact CUDA index URLs ($12.6, 12.4, 12.1, 11.8$) or CPU/MPS fallbacks.
- **NFS Storage Optimization:** On shared HPC clusters using Network File Systems (NFS), virtual environments are safely initialized in local scratch space (`/tmp`) and symlinked to `./.venv`, preventing sqlite file-locking bottlenecks and I/O degradation.
- **Curriculum Selection:** Allows you to install minimal core dependencies or opt into advanced deep learning (`tensorsmith`) and CUDA computing (`hpcsmith`) toolchains on demand.

---

## 📚 Curriculum Modules

TensorForge is organized into three progressive, highly specialized curriculums:

| Module | Focus Area | Core Competencies | Documentation |
| --- | --- | --- | --- |
| 🧮 **[ArraySmith](arraysmith/README.md)** | **Scientific Computing & NumPy** | Array creation, strided memory layouts, broadcasting rules, boolean masking, and zero-copy slicing. **SAY NO TO FOR LOOPS!** | [Explore ArraySmith](arraysmith/README.md) |
| 🧠 **[TensorSmith](tensorsmith/README.md)** | **Deep Learning & PyTorch** | Autograd mechanics, custom neural network architectures, optimization algorithms, and research paper re-implementations. | [Explore TensorSmith](tensorsmith/README.md) |
| ⚡ **[HPCSmith](hpcsmith/README.md)** | **High-Performance Computing & CUDA** | C++ ATen bindings, CUDA kernel authoring, shared-memory tiling, warp primitives, and on-the-fly JIT compilation. | [Explore HPCSmith](hpcsmith/README.md) |

---

## 💻 The "LeetCode-Style" Workflow

TensorForge is powered by a dedicated command-line interface (`tforge`) designed to make solving exercises frictionless and rewarding.

### Step 1: Check Your Dashboard

Check your overall curriculum progress across all modules and tiers:

```bash
uv run tforge status
```

### Step 2: Open an Exercise

Navigate to any lesson directory (e.g., `arraysmith/basic/01_array_creation`). Each lesson folder contains:

- `INSTRUCTION.md`: Detailed problem specifications, mathematical formulas, and examples.
- `student_code.py` (or `student_code.cpp` / `student_code.cu`): Your workspace containing method stubs raising `NotImplementedError` or `std::runtime_error`.

### Step 3: Implement Your Solution

Open the target file and replace the stub with your implementation. For example, in `arraysmith`:

```python
@classmethod
def create_integer_range(cls, start: int, stop: int, step: int) -> np.ndarray:
    # Replace raise NotImplementedError(...) with your vectorized solution:
    return np.arange(start, stop, step)
```

### Step 4: Verify & Benchmark

Run the automated verification suite for your lesson:

```bash
# Syntax: uv run tforge check <module> <tier> <lesson_id>
uv run tforge check arraysmith basic 01
```

You will receive an instant, colorized test report:

1. **AST Policy Audit:** Confirms zero forbidden constructs (`for` loops, illegal function calls).
2. **Correctness Verification:** Runs parameterized test cases against golden reference outputs.
3. **Performance & Memory Benchmark:** Measures execution time, computes slowdown ratios against the baseline, reports peak memory footprint (`KB`), and awards a performance badge!

### Additional CLI Commands

```bash
# Run in Fast Mode (correctness only, skips timing loops and tracemalloc overhead)
uv run tforge check arraysmith basic 01 --fast

# Check an entire tier or curriculum
uv run tforge check arraysmith basic
uv run tforge check hpcsmith

# Run framework infrastructure tests
uv run tforge check infra

# Scaffold a new clean lesson directory and matching test suite
uv run tforge generate arraysmith basic 04_new_lesson
```

---

## 💡 Using the Hint System

If you get stuck on an exercise without wanting to spoil the solution, replace `raise NotImplementedError` in your method stub with:

```python
from hint import show_hint
show_hint()
```

When you re-run `uv run tforge check <module> <tier> <lesson>`, the CLI halts execution and displays tiered major and minor hints from the centralized registry tailored specifically to the lesson and method under test.

---

## 🏛️ Core Architecture & Features

### 🔌 HPC Bridge & 4-Stage Lifecycle

For C++ and CUDA kernel development in **HPCSmith**, TensorForge provides an extensible plugin backend (`CudaJitBackend`) that automates `torch.utils.cpp_extension.load`. Every execution strictly obeys a 4-stage lifecycle to isolate latency:

1. `setup()`: Transfers host tensors to device memory (`H2D`).
2. `warmup()`: Absorbs JIT compilation overhead and CUDA stream initialization.
3. `execute()`: Launches kernels with exact stream synchronization (`torch.cuda.synchronize()`).
4. `teardown()`: Cleans up VRAM pools (`torch.cuda.empty_cache()`).

### 📦 Centralized Dataset Manager

TensorForge includes a robust `DatasetManager` for downloading, verifying, and caching benchmark datasets atomically:

```python
from forge_core import DatasetManager

# Downloads atomically to ~/.cache/tensorforge/datasets (or local .tensorforge_datasets/)
dataset_path = DatasetManager.get_dataset(
    name="mnist_sample",
    url="https://example.com/mnist.tar.gz",
    sha256="expected_hash_here..."
)
```

### 🎮 Performance Gamification Tiers

Your code execution time is compared against optimized C/Fortran/CUDA baselines ($R = \frac{T_{\text{student}}}{T_{\text{baseline}}}$):

- 👑 **`[1.0x — Forge God / Chad]`**: Perfectly optimal, peak hardware utilization (Magenta `#ff00ff`).
- 🚀 **`[1.2x — Optimized]`**: Excellent performance, clean memory access (Green `#00ff87`).
- ⚠️ **`[2.0x — Sub-optimal]`**: Functional, but leaving performance on the table (Yellow `#ffd700`).
- 🐢 **`[5.0x — Mid-wit]`**: Noticeable overhead or uncoalesced memory access (Orange `#ff8800`).
- 🤡 **`[15.0x+ — Soy dev]`**: Terrible performance, accidental copying, or severe bloat (Red `#ff3333`).

---

## 🤝 Contributing & Development

We welcome contributions from systems engineers and educators! Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Authoring new curriculum lessons and reference baselines.
- Adding AST inspection rules and custom feedback messages.
- Extending HPC backends and registering automated hints.

---

## 👤 Author Information

- **Author & Principal Architect:** nguyenpanda
- **Affiliation:** High Performance Computing Laboratory, HCMUT-VNU

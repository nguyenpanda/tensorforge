# TensorForge 🔥

An interactive, professional-grade laboratory training ground for mastering scientific computing and deep learning — from **NumPy** fundamentals to advanced **PyTorch** architectures.

Designed to train engineers through rigorous, hands-on coding challenges enforced by automated correctness verification, real-time performance benchmarking, memory profiling, and static AST structural analysis under a **Clean Lab** architecture.

---

## 🌟 Overview

TensorForge bridges the gap between theoretical knowledge and production-grade software engineering. Rather than simple toy problems, students implement core numerical and tensor algorithms against optimized reference baselines.

### Key Architectural Pillars
- **Clean Lab Architecture:** Student workspaces (`arraysmith/`) are cleanly separated from test verification suites and reference baselines (`tests/arraysmith/`), eliminating clutter and preventing accidental cheating or baseline tampering.
- **Tier-Scoped Numbering:** The curriculum is organised into semantic tiers (`basic`, `intermediate`, `advanced`, `applications`). Lesson IDs reset to `01` per tier, eliminating the sequential-numbering ripple-effect that makes flat curricula brittle to reorganisation.
- **Static AST Enforcement:** The `@ast_policy` engine inspects Abstract Syntax Trees before code execution to strictly forbid explicit Python loops (`for` / `while`) and enforce required numerical APIs (e.g., `np.where`, `np.isclose`).
- **Real-Time Performance & Memory Profiling:** Solutions are evaluated for execution speed against optimized C/C++ backed kernels alongside peak memory footprint tracking in kilobytes (`KB`) via `tracemalloc`.
- **Strict Dtype Verification:** Automated assertions guarantee exact data type matching between student outputs and reference baselines.
- **Dynamic Context-Aware Hints:** An integrated auto-discovering hint system provides non-blocking guidance tailored to the exact class and method under test.
- **HPC Backend Plugin Architecture:** A `forge_core/backends/` plugin system with an `ExecutionBackend` ABC enforces a strict four-stage HPC lifecycle (`setup → warmup → execute → teardown`), making the harness CUDA/Triton-ready.
- **Dependency Isolation:** Optional extras (`torch`, `hpc`) are isolated from the core install. The core environment requires only NumPy, pytest, and swan.

---

## 🏗️ Project Structure

The project follows a mirrored Clean Lab design with Tier-Scoped Numbering:

```
tensorforge/
├── forge_core/                ← Shared infrastructure
│   ├── backends/              ← HPC backend plugin system
│   │   ├── base.py            ← ExecutionBackend ABC (setup/warmup/execute/teardown)
│   │   └── numpy_backend.py   ← CPU NumPy backend (default)
│   ├── ast_validator.py       ← Dynamic AST policy enforcement engine
│   └── benchmark.py           ← Time & memory benchmarking runner
├── hint/                      ← Dynamic, auto-discovering hint registry
│   └── hint/arraysmith/
│       ├── basic/             ← Basic-tier hint files
│       ├── intermediate/      ← Intermediate-tier hint files
│       └── advanced/          ← Advanced-tier hint files
├── arraysmith/                ← Student workspaces (Tier-Scoped NumPy curriculum)
│   ├── basic/
│   │   ├── 01_array_creation/
│   │   │   ├── INSTRUCTION.md ← Step-by-step specifications & math rules (LaTeX)
│   │   │   └── student_code.py ← Write your implementations here
│   │   ├── 02_properties_reshaping/
│   │   └── 03_indexing_slicing/
│   ├── intermediate/
│   │   ├── 01_vectorized_math/
│   │   ├── 02_array_manipulation/
│   │   └── 03_boolean_logic/
│   ├── advanced/
│   │   └── 01_memory_layout/
│   └── applications/          ← Future end-to-end project lessons
├── tensorsmith/               ← PyTorch curriculum (requires: uv sync --extra torch)
├── tests/                     ← Mirrored verification suite & reference baselines
│   ├── conftest.py            ← Consolidated dynamic path & module resolution
│   ├── test_infrastructure.py ← Benchmark, backend, AST & hint system unit tests
│   ├── tensorsmith/
│   │   └── conftest.py        ← torch importorskip guard
│   └── arraysmith/            ← Mirrored tiered test directories
│       ├── basic/
│       │   ├── 01_array_creation/
│       │   │   ├── _baseline.py  ← Reference solutions (read-only)
│       │   │   └── test_01.py    ← Pytest verification suite with AST policies
│       │   └── ...
│       ├── intermediate/
│       └── advanced/
├── main.py                    ← CLI entry point
├── validate_baselines.py      ← Multi-tier project validation tool
├── conftest.py                ← Root Pytest configuration & shared fixtures
└── pyproject.toml             ← Root project and dependency configuration
```

---

## 📋 Prerequisites & Installation

Ensure you have Python 3.12+ and `uv` installed ([installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
# 1. Clone the repository and sync the virtual environment (core dependencies only)
git clone <repository-url>
cd tensorforge
uv sync

# 2. Install optional PyTorch dependency (required for tensorsmith)
uv sync --extra torch

# 3. Verify project health and infrastructure integrity
tforge validate
```

Running `uv sync` installs the core environment: NumPy, pytest, matplotlib, seaborn, and `nguyenpanda.swan`. PyTorch is an **optional extra** that does not pollute the core install.

---

## 🖥️ CLI API Reference

TensorForge provides a unified global entry point via `tforge`.

### 1. Check Solutions (`check`)

The `check` command accepts tier-scoped addressing: `tforge check <curriculum> <tier> <lesson_id> [method]`

```bash
# Check a specific lesson (tier-scoped)
tforge check arraysmith basic 01
tforge check arraysmith intermediate 02
tforge check arraysmith advanced 01

# Check a specific method within a lesson
tforge check arraysmith basic 01 create_squared_range

# Check all lessons in a tier
tforge check arraysmith basic
tforge check arraysmith intermediate

# Check the entire arraysmith curriculum
tforge check arraysmith

# Run infrastructure unit tests (benchmark, backend ABC, AST enforcer, hints)
tforge check infra

# Run all tests (all curricula)
tforge check all
```

### 2. View Curriculum Status (`status`)
Display a tier-grouped, 24-bit colorized progress report across all modules:
```bash
tforge status
```

### 3. Validate Baselines (`validate`)
Execute a rigorous multi-tier audit verifying baseline self-consistency, memory footprints, infrastructure integrity, and hint registry completeness:
```bash
tforge validate
```

### 4. Scaffold New Lessons (`generate`)
Dynamically scaffold clean student workspaces and mirrored verification test suites for new curriculum lessons, respecting tier placement:
```bash
tforge generate arraysmith basic 04_new_lesson
tforge generate arraysmith intermediate 04_new_topic
```

---

## 🧪 Evaluation & Feedback Criteria

When you run a test using `tforge check <module> <tier> <lesson>`, your code is evaluated across four strict dimensions:

| Criteria | Verification Engine | Common Failure Cause |
|---|---|---|
| **Smart AST Compliance** | `ast_validator.py` (`@ast_policy`) | Using explicit Python `for` / `while` loops; custom feedback guides you to required vectorized APIs. |
| **Exact Values & Shapes** | `assert_array_equal` / `assert_allclose` | Incorrect numerical formula, logic error, or shape mismatch against baseline. |
| **Strict Dtype Enforcement** | Exact dtype assertions | Returning default `float64` or `int64` when `int8` or `int32` is explicitly required. |
| **Speed & Memory Footprint** | `benchmark.py` (`tracemalloc`) | Unoptimized array copying or slow operations (slowdown > 5×); tracks peak memory in `KB`. |

### 🎮 5-Tier Gamification & Performance Badges

When your solution passes correctness verification, TensorForge benchmarks its execution speed against the reference baseline and displays a 24-bit true color badge based on your speed ratio ($\text{Time}_{\text{student}} / \text{Time}_{\text{baseline}}$):

- **Ratio $\le 1.05\times$**: `[Forge God / Chad]` (Magenta `#ff00ff`) — State-of-the-art C/Fortran kernel performance.
- **$1.05\times < \text{Ratio} \le 1.5\times$**: `[Optimized]` (Green `#00ff87`) — Clean, production-grade vectorized code.
- **$1.5\times < \text{Ratio} \le 3.0\times$**: `[Sub-optimal]` (Yellow `#ffd700`) — Valid vectorization with minor memory or copying overhead.
- **$3.0\times < \text{Ratio} \le 10.0\times$**: `[Mid-wit]` (Orange `#ff8800`) — Significant overhead or redundant intermediate allocations.
- **Ratio $> 10.0\times$**: `[Soy dev]` (Red `#ff3333`) — Severe slowdown or unvectorized bottleneck.

### 🧠 Smart AST Feedback & Memory Profiling

- **Smart AST Feedback:** If your code violates structural rules, the `@ast_policy` engine outputs custom instructional feedback tailored to the exact constraint violated.
- **Peak Memory Profiling:** Every evaluation tracks peak RAM footprint in kilobytes (`KB`) using `tracemalloc`.
- **Strict Dtype Enforcement:** Submissions must match the exact NumPy data type required by the specification.

---

## 💡 Using the Hint System

If you get stuck on an exercise, simply replace the `raise NotImplementedError` line inside your exercise method in `student_code.py` with `show_hint()`:

```python
class ArrayCreation:
    @classmethod
    def create_integer_range(cls) -> np.ndarray:
        show_hint()  # ← Calling show_hint() automatically halts execution and displays help
```

When you re-run the check CLI, the console outputs structured, interactive guidance:

```
======================================================================
💡  HINT
======================================================================
📂  Module  : basic/01_array_creation
📄  File    : .../student_code.py
📦  Class   : ArrayCreation
🔧  Method  : create_integer_range()
----------------------------------------------------------------------
🎯  Major Hint:
    Use np.arange(start, stop) to create an integer sequence.
🔍  Minor Hint:
    np.arange(0, 100) gives exactly what you need here.
======================================================================
```

---

## 🔌 HPC Backend Plugin Architecture

The `forge_core/backends/` package provides an extensible execution backend system for future GPU/HPC targets:

```python
from forge_core.backends import ExecutionBackend, NumpyBackend

# Use the default NumPy backend (transparent, backward-compatible)
compare_and_benchmark(student_fn=MyClass.method, baseline_fn=BaselineClass.method)

# Or inject an explicit backend for future CUDA/Triton experiments
compare_and_benchmark(
    student_fn=MyClass.method,
    baseline_fn=BaselineClass.method,
    student_backend=NumpyBackend(fn=MyClass.method),
)
```

New backends only need to implement four lifecycle methods defined in `ExecutionBackend`:
- `setup()` — Allocate memory / H2D transfer.
- `warmup()` — Eliminate JIT / CUDA cold-start overhead.
- `execute()` — The measured computation.
- `teardown()` — Release resources.

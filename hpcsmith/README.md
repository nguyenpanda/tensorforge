# HPCSmith: High-Performance Computing & CUDA Kernel Mastery ⚡

Welcome to **HPCSmith**, the low-level scientific computing and GPU kernel programming curriculum within the TensorForge framework. This module focuses on writing custom C++ and CUDA extensions, optimizing shared memory tiling, mastering thread-level parallelism, and bridging native code into PyTorch via the HPC Bridge.

---

## Table of Contents
1. [Overview & Objectives](#overview--objectives)
2. [Prerequisites](#prerequisites)
3. [Course Syllabus](#course-syllabus)
4. [Theoretical Foundations](#theoretical-foundations)
5. [The HPC Bridge & Plugin Architecture](#the-hpc-bridge--plugin-architecture)
6. [Getting Started & CLI Commands](#getting-started--cli-commands)

---

## Overview & Objectives

HPCSmith takes you below the Python API layer directly into hardware execution. You will write high-performance C++ and CUDA kernels from scratch, compiling them on-the-fly and binding them into PyTorch tensors.

Key learning outcomes include:
- Writing custom C++ ATen extensions for PyTorch tensor manipulation.
- Programming CUDA kernels with custom grid/block thread configurations.
- Implementing shared-memory tiling strategies to eliminate global memory bandwidth bottlenecks.
- Managing memory coalescing, bank conflicts, and warp divergence.
- Utilizing the four-stage HPC Bridge lifecycle (`setup → warmup → execute → teardown`).

---

## Prerequisites

Before beginning the HPCSmith curriculum, ensure you have the required toolchain and dependencies:
- **Python Environment:** Python 3.12+ configured via `python bootstrap.py`.
- **HPC & PyTorch Extras:** Install both `torch` and `hpc` build dependencies (`scikit-build-core`, `nanobind`, `ninja`):
  ```bash
  uv sync --extra torch --extra hpc
  ```
- **C++ / CUDA Toolchain:** A modern C++17 compiler (GCC, Clang, or MSVC) and the NVIDIA CUDA Toolkit (`nvcc`) for GPU kernels. On CPU-only systems (e.g., Apple Silicon or standard CI nodes), C++ kernels execute natively while CUDA kernel tests skip gracefully.
- **Prior Knowledge:** Proficiency in C/C++, linear algebra, and basic PyTorch tensor operations.

---

## Course Syllabus

HPCSmith is structured into progressive, tier-scoped lessons:

### Tier 1: Basic
- **`01_cpp_integration`**: Introduction to the HPC Bridge. Writing a custom C++ ATen tensor addition extension (`student_code.cpp`), compiling it via JIT, and verifying numerical parity against NumPy/PyTorch.

### Tier 2: Intermediate
- **`01_cuda_gemm`**: General Matrix Multiplication (GEMM) on NVIDIA GPUs (`student_code.cu`).
  - *Naive GEMM:* Direct global memory memory access per thread.
  - *Tiled GEMM:* Shared-memory tiling with 2D thread blocks to drastically reduce global memory bandwidth consumption.

### Tier 3: Advanced
- **`advanced`**: Warp-level primitives, parallel reduction trees, FlashAttention-style kernel design, and custom Triton kernel implementations (coming soon).

### Tier 4: Applications
- **`applications`**: End-to-end custom inference operators and large-scale parallel PDE solvers.

---

## Theoretical Foundations

### 1. CUDA Thread Hierarchy & Execution Model
NVIDIA GPUs execute kernels using a hierarchy of threads:
- **Grid:** The complete domain of threads executing a kernel, divided into 1D, 2D, or 3D blocks.
- **Thread Block:** A group of up to 1024 threads that execute on a single Streaming Multiprocessor (SM), sharing fast on-chip shared memory and synchronization barriers (`__syncthreads()`).
- **Warp:** A group of 32 threads executing in lockstep (SIMT — Single Instruction, Multiple Threads).

### 2. Global Memory Bandwidth vs. Shared Memory Tiling
Global memory on a GPU has high latency (hundreds of clock cycles). In a naive matrix multiplication, each thread reads $2N$ global memory elements to compute one output dot product. By loading sub-matrices (tiles) into fast on-chip **Shared Memory**, $L \times L$ threads cooperate to reuse data, reducing global memory traffic by a factor of $L$.

### 3. Memory Coalescing & Bank Conflicts
To maximize throughput, sequential threads in a warp must access sequential memory addresses (coalesced memory access). In shared memory, addresses are mapped to 32 parallel memory banks; if multiple threads access different addresses in the same bank simultaneously, a **bank conflict** occurs, serializing access.

---

## The HPC Bridge & Plugin Architecture

HPCSmith leverages TensorForge's `forge_core.backends` plugin system. The `CudaJitBackend` automates compilation and lifecycle management:

```python
from forge_core.backends import CudaJitBackend
from forge_core.benchmark import compare_and_benchmark

# CudaJitBackend compiles C++/CUDA source on-the-fly and binds it to PyTorch
backend = CudaJitBackend(
    source_path="student_code.cu",
    module_name="hpcsmith_intermediate_01_cuda_gemm",
    function_name="run_tiled_gemm",
)

compare_and_benchmark(
    student_fn=student_gemm_wrapper,
    baseline_fn=baseline_gemm_wrapper,
    student_backend=backend,
)
```

The execution lifecycle guarantees rigorous profiling:
1. **`setup()`**: Transfers input matrices to GPU device memory (`H2D`).
2. **`warmup()`**: Triggers JIT compilation and absorbs CUDA stream initialization latency.
3. **`execute()`**: Invokes the kernel and calls `torch.cuda.synchronize()` for accurate timing.
4. **`teardown()`**: Reclaims GPU VRAM pool cache (`torch.cuda.empty_cache()`).

---

## Getting Started & CLI Commands

Write your C++ and CUDA code inside your lesson folder (`student_code.cpp` or `student_code.cu`) and your Python binding wrappers inside `student_code.py`.

### Running Verification
Use the `tforge` CLI to compile and test your kernels:

```bash
# Ensure toolchain dependencies are installed
uv sync --all-extras

# Check C++ integration lesson
uv run tforge check hpcsmith basic 01

# Check CUDA GEMM lesson
uv run tforge check hpcsmith intermediate 01

# Run in Fast Mode (correctness only, skips timing and tracemalloc)
uv run tforge check hpcsmith basic 01 --fast
```

### Using Hints
To view interactive architectural guidance, replace your stub in `student_code.py` with:
```python
from hint import show_hint
show_hint()
```

# Module 01 — CUDA General Matrix Multiplication (GEMM)

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:

- Understand the memory hierarchy of CUDA architectures (global memory vs. shared memory).
- Implement a naive CUDA kernel for General Matrix Multiplication (GEMM) using global memory accesses.
- Implement an optimized tiled GEMM kernel using shared memory to reduce global memory bandwidth requirements.
- Execute custom CUDA kernels via the TensorForge `CudaJitBackend` lifecycle using PyTorch C++ bindings (`PYBIND11_MODULE`).

---

## 2. Concept Introduction

### 2.1 Naive GEMM: Global Memory Bound

In a naive matrix multiplication $C = A \times B$ where $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, each element $C[i, j]$ is computed by taking the dot product of row $i$ of matrix $A$ and column $j$ of matrix $B$:

$$C[i, j] = \sum_{k=0}^{K-1} A[i, k] \times B[k, j]$$

When mapped directly to a CUDA kernel without caching or tiling, each thread computes a single element $C[i, j]$. To do so, the thread must load $K$ elements from matrix $A$ and $K$ elements from matrix $B$ directly from global memory (DRAM). Across the entire matrix, this results in $2 \times M \times N \times K$ global memory reads. Because global memory latency is high and bandwidth is limited, the naive kernel is severely **memory-bandwidth bound**.

### 2.2 Tiled GEMM: Shared Memory Optimization

To overcome the global memory bottleneck, we utilize on-chip **shared memory** (SRAM), which has significantly higher bandwidth and lower latency than DRAM.

Tiling divides matrices $A$ and $B$ into square sub-blocks of dimensions `TILE_SIZE x TILE_SIZE` (typically $16 \times 16$ or $32 \times 32$). All threads within a CUDA thread block collaborate to load one tile of $A$ and one tile of $B$ from global memory into shared memory (`__shared__ float As[TILE_SIZE][TILE_SIZE]` and `Bs[TILE_SIZE][TILE_SIZE]`).

Once the tiles are loaded and synchronized via `__syncthreads()`, each thread reads from shared memory to accumulate partial dot products. Because each loaded element in shared memory is reused `TILE_SIZE` times by different threads in the block, global memory bandwidth demand is reduced by a factor of approximately `TILE_SIZE`.

### 2.3 The HPC Bridge Lifecycle and LRU Caching

In Python, we use `@functools.lru_cache` to ensure the `CudaJitBackend` instance is created only once per process:

```python
@functools.lru_cache(maxsize=None)
def _get_backend() -> CudaJitBackend:
    return CudaJitBackend(
        source_path=_CU_SOURCE,
        module_name="cuda_gemm",
        function_name="run_gemm",
    )
```

The execution follows the four-stage lifecycle:
1. `setup(a, b, use_tiled=True)`: Moves tensors to CUDA memory and prepares arguments.
2. `warmup()`: Initiates CUDA context and stream initialization.
3. `execute()`: Runs the kernel and synchronizes the stream.
4. `teardown()`: Clears device references and cleans up allocator state.

---

## 3. Exercises

Open `student_code.py` and implement the `CudaGemm` class methods:

### Exercise 1 — `run_naive_gemm(a, b)`
Execute the naive global-memory GEMM kernel by invoking `CudaJitBackend` with `use_tiled=False`.

### Exercise 2 — `run_tiled_gemm(a, b)`
Execute the shared-memory tiled GEMM kernel by invoking `CudaJitBackend` with `use_tiled=True`.

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI or pytest:

```bash
# Run tests for Lesson 01
uv run pytest tests/hpcsmith/intermediate/01_cuda_gemm/test_01.py -v -s

# Check using the TensorForge CLI
tforge check hpcsmith intermediate 01
```

# Module: CUDA General Matrix Multiplication (GEMM)

## Learning Objectives
- Understand the memory hierarchy of CUDA hardware architectures (global DRAM vs. on-chip SRAM shared memory).
- Analyze memory bandwidth utilization and latency bottlenecks in naive parallel matrix multiplication kernels.
- Implement optimized tiled execution strategies using collaborative shared memory loading and thread synchronization barriers.
- Integrate native CUDA execution routines into Python pipelines via transparent JIT compilation.

---

## Exercise 1: Naive Global Memory GEMM (`run_naive_gemm`)

In a direct parallel matrix multiplication mapping, individual execution threads calculate discrete elements of the output matrix by repeatedly accessing operand elements from off-chip global memory.

### Input Specifications
- **`a`**: A PyTorch tensor representing matrix $A$.
- **Shape**: `(M, K)` where $M, K \ge 1$.
- **`b`**: A PyTorch tensor representing matrix $B$.
- **Shape**: `(K, N)` where $K, N \ge 1$.
- **Data Type**: Standard floating-point precision (`float32` or `float64`).
- **Device**: CUDA-compatible GPU memory allocation.

### Output Specifications
- **Return Type**: `torch.Tensor`.
- **Shape**: `(M, N)` — the matrix product $C = A \times B$.
- **Data Type**: Identical to input operands `a` and `b`.
- **Constraints**:
  - The matrix multiplication must be executed via a native CUDA kernel without on-chip memory caching.
  - The implementation must be exposed via Python bindings and invoked through the execution wrapper.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Tiled Shared Memory GEMM (`run_tiled_gemm`)

To overcome DRAM bandwidth constraints, thread blocks collaborate to stage square data sub-blocks into high-speed on-chip shared memory, amortizing global memory access costs across multiple arithmetic operations.

### Input Specifications
- **`a`**: A PyTorch tensor representing matrix $A$ of shape `(M, K)`.
- **`b`**: A PyTorch tensor representing matrix $B$ of shape `(K, N)`.
- **Data Type**: Floating-point (`float32` or `float64`).
- **Device**: CUDA-compatible GPU memory allocation.

### Output Specifications
- **Return Type**: `torch.Tensor`.
- **Shape**: `(M, N)` — the matrix product $C = A \times B$.
- **Data Type**: Identical to input operands `a` and `b`.
- **Constraints**:
  - The matrix multiplication must utilize collaborative shared memory tiling to minimize global memory traffic.
  - Proper thread synchronization barriers must be employed to prevent data race conditions during shared memory staging.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check hpcsmith intermediate 01

# Verify a specific exercise
uv run tforge check hpcsmith intermediate 01 -t run_tiled_gemm
```

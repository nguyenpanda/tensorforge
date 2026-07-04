# Module 01 — C++ Integration (HPC Bridge)

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:

- Connect student-authored C++ / CUDA kernels to Python using PyTorch's `torch.utils.cpp_extension.load`.
- Execute accelerated native code via TensorForge's `CudaJitBackend` four-stage lifecycle (`setup → warmup → execute → teardown`).
- Understand device-aware execution and graceful fallback to CPU-only mode on non-CUDA environments (e.g., Apple Silicon macOS).
- Seamlessly transition data between NumPy arrays (`np.ndarray`) and PyTorch tensors (`torch.Tensor`) with zero copying where possible using `torch.from_numpy()`.

---

## 2. Concept Introduction

### 2.1 Why Native C++ Extensions?

While PyTorch provides optimized built-in operations, advanced research and high-performance computing (HPC) often require writing custom kernels in C++ or CUDA (e.g., custom fused attention kernels, specialized differential equations, or custom activation layers).

TensorForge bridges this gap through the **HPC Bridge**, which dynamically compiles C++/CUDA source files at runtime and presents them as standard Python callables.

### 2.2 The HPC Bridge Lifecycle

To isolate initialization latency, JIT compilation time, and device memory transfer overhead from steady-state compute timing, all HPC backends implement the `ExecutionBackend` four-stage contract:

```python
from forge_core.backends.cuda_backend import CudaJitBackend

# 1. Instantiate (compiles C++/CUDA source on first load and caches the module)
backend = CudaJitBackend(
    source_path="hpcsmith/native/01_cpp_addition.cpp",
    module_name="cpp_addition",
    function_name="add_tensors",
)

# 2. Setup: transfer host tensors to device memory (H2D)
backend.setup(tensor_a, tensor_b)

# 3. Warmup: execute once to initialize CUDA context, PTX JIT, and streams
backend.warmup()

# 4. Execute: invoke kernel and synchronize device stream for accurate timing
result = backend.execute()

# 5. Teardown: release tensor references and clear device allocator cache
backend.teardown()
```

### 2.3 Device-Aware Execution

The `CudaJitBackend` automatically detects whether a CUDA-capable GPU is available via `torch.cuda.is_available()`. On systems without CUDA (such as macOS or CPU-only Linux), device transfers (`.cuda()`) and stream synchronization (`torch.cuda.synchronize()`) are bypassed transparently. Standard C++ extensions execute natively on the CPU without modification.

---

## 3. Exercises

Open `student_code.py` and review the implementation of `CppIntegration.run_cpp_addition(a, b)`.

### Exercise 1 — `run_cpp_addition()`

**Task:** Add two NumPy arrays $a, b \in \mathbb{R}^{N \times M}$ by bridging them to a C++ PyTorch extension kernel:

$$y = \text{add\_tensors}(\text{tensor}(a), \text{tensor}(b))$$

**Implementation Steps:**
1. Convert input NumPy arrays to PyTorch tensors using `torch.from_numpy(np.ascontiguousarray(x))`.
2. Retrieve the process-scoped compiled backend instance via `_get_backend()`.
3. Execute the full four-stage lifecycle (`setup`, `warmup`, `execute`, `teardown`).
4. Convert the resulting PyTorch tensor back to a NumPy array via `.numpy()`.

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI (Full Benchmark Mode)
tforge check hpcsmith basic 01

# Check in Fast Mode (correctness only, skips timing loops and tracemalloc overhead)
tforge check hpcsmith basic 01 --fast
tforge check hpcsmith basic 01 -f

# Check a specific test method within this lesson
tforge check hpcsmith basic 01 test_addition_float32_small

# Run all tests in the hpcsmith curriculum
tforge check hpcsmith

# View progress across all curricula (arraysmith, tensorsmith, and hpcsmith)
tforge status
```

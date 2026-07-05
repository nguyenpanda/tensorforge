# Module 01 — C++ Integration (HPC Bridge)

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:
- Connect student-authored C++ kernels to Python using PyTorch's `torch.utils.cpp_extension.load`.
- Seamlessly transition data between NumPy arrays (`np.ndarray`) and PyTorch tensors (`torch.Tensor`) with zero copying where possible using `torch.from_numpy()`.
- Understand how pybind11 exposes native ATen operators to the Python runtime.

---

## 2. Concept Introduction

### 2.1 Why Native C++ Extensions?

While PyTorch provides optimized built-in operations, high-performance computing (HPC) and custom systems research often require authoring native C++ or CUDA kernels. The TensorForge framework transparently compiles `student_code.cpp` on-the-fly and binds native symbols directly to Python.

### 2.2 Transparent JIT Compilation

The Python wrapper `student_code.py` invokes `torch.utils.cpp_extension.load` to compile the C++ source file and map symbols into the module namespace:

```python
from torch.utils.cpp_extension import load

_cpp_module = load(
    name="hpc_basic_01_cpp_integration",
    sources=["student_code.cpp"],
    extra_cflags=["-O3"],
)
add_tensors = _cpp_module.add_tensors
```

---

## 3. Exercises

Open `student_code.cpp` and implement the C++ function `add_tensors`.

### Exercise 1 — `add_tensors()`

**Task:** Add two PyTorch tensors $a, b$ using C++ ATen arithmetic operator overloading:

$$y = a + b$$

**Implementation Steps:**
1. In `student_code.cpp`, remove the `throw std::runtime_error(...)` statement.
2. Return `a + b;` using ATen's overloaded addition operator.

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI:

```bash
# Check this lesson
uv run tforge check hpcsmith basic 01

# Run tests directly with pytest
uv run pytest tests/curriculum/hpcsmith/basic/01_cpp_integration/test_01.py -v -s
```

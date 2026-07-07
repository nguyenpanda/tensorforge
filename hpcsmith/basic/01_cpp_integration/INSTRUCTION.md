# Module: C++ Integration & HPC Bridge

## Learning Objectives
- Connect native C++ compute kernels to the Python runtime environment via JIT compilation.
- Transition tensor data buffers across host memory and native execution backends without redundant copying.
- Understand how native ATen tensor operations bind to Python call interfaces.

---

## Exercise 1: Native Tensor Addition (`add_tensors`)

In high-performance systems and custom operator design, core compute kernels are authored in native C++ to bypass interpreter overhead and leverage specialized compiler optimizations.

### Input Specifications
- **`a`**: A PyTorch tensor (`at::Tensor` in C++, `torch.Tensor` in Python).
- **`b`**: A PyTorch tensor (`at::Tensor` in C++, `torch.Tensor` in Python).
- **Shape**: Both input tensors share identical dimensions $(d_0, d_1, \dots, d_{k-1})$ where $k \ge 1$.
- **Data Type**: Standard numeric precision (`float32`, `float64`, `int32`, `int64`).

### Output Specifications
- **Return Type**: `at::Tensor` (`torch.Tensor`).
- **Shape**: Identical to the dimensions of input tensors `a` and `b`.
- **Data Type**: Identical to the input tensors' data type.
- **Constraints**:
  - The returned tensor must contain the element-wise sum of operands `a` and `b`.
  - The implementation must be authored in `student_code.cpp` and exposed via pybind11 module bindings.
  - You must not perform manual scalar loops across tensor elements in C++.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check hpcsmith basic 01

# Verify a specific exercise
uv run tforge check hpcsmith basic 01 -t test_addition_correctness
```

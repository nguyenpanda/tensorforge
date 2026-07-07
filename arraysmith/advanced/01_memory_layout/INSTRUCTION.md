# Module: Memory Layout, Contiguity & Strides

## Learning Objectives
- Understand C-contiguous (row-major) versus Fortran-contiguous (column-major) memory storage architectures.
- Analyze array byte strides to understand multidimensional memory offset calculations.
- Differentiate between zero-copy memory views and independent data buffer allocations.

---

## Exercise 1: Ensuring C-Contiguity (`ensure_c_contiguous`)

In high-performance computing and hardware accelerators, matrix kernels require C-contiguous row-major alignment in memory to achieve optimal memory coalescing.

### Input Specifications
- **`arr`**: A `numpy.ndarray` of arbitrary dimension $d \ge 1$ and arbitrary data type, stored in either C-contiguous, Fortran-contiguous, or non-contiguous sliced layout.
- **Shape**: $(N_0, N_1, \dots, N_{k-1})$ where $N_i \ge 1$.

### Output Specifications
- **Return Type**: `numpy.ndarray` with identical shape, data type, and element values as the input array.
- **Constraints**: 
  - The returned array must have C-contiguous memory ordering.
  - If the input array is already C-contiguous, it must be returned without allocating a new memory buffer to conserve memory bandwidth.
  - If the input array is not C-contiguous, a new C-contiguous array must be returned.
  - You must not use Python loops.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Detecting Shared Memory Buffers (`check_memory_share`)

When slicing tensors across pipeline stages, unintended data mutations can occur if arrays share underlying memory buffers without explicit copying.

### Input Specifications
- **`arr1`**: A `numpy.ndarray` of arbitrary shape and data type.
- **`arr2`**: A `numpy.ndarray` of arbitrary shape and data type.

### Output Specifications
- **Return Type**: `bool` (`True` or `False`).
- **Constraints**: 
  - Return `True` if `arr1` and `arr2` share any overlapping physical memory addresses in their underlying data buffer.
  - Return `False` if they reference entirely independent memory allocations.
  - You must not use Python loops.

Call `show_hint()` if you are stuck.

---

## Exercise 3: Inspecting Row Strides (`get_row_stride_bytes`)

The physical memory address of element $(i, j)$ in a 2D matrix is determined by axis stride parameters representing the number of bytes required to traverse between consecutive indices.

### Input Specifications
- **`arr`**: A 2D `numpy.ndarray` of shape $(N, M)$ where $N \ge 1$ and $M \ge 1$, with any standard numerical data type.

### Output Specifications
- **Return Type**: `int`.
- **Constraints**: 
  - Return the exact number of bytes required in physical memory to step from row $i$ to row $i+1$ along axis 0.
  - You must not use Python loops.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check arraysmith advanced 01

# Verify a specific exercise
uv run tforge check arraysmith advanced 01 -t ensure_c_contiguous
```

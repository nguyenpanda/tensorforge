# Module 06: Memory Layout, Contiguity & Strides

## Learning Objectives
- Understand C-contiguous (row-major) versus Fortran-contiguous (column-major) memory storage in RAM.
- Inspect array byte strides $S = (s_0, s_1, \dots, s_{k-1})$ to calculate memory offsets for multidimensional indexing.
- Differentiate between zero-copy memory views (inspecting `base` or `np.may_share_memory`) and independent data copies.

---

## Exercise 1: Ensuring C-Contiguity (`ensure_c_contiguous`)

In high-performance computing and neural network accelerators, matrix kernels often demand C-contiguous row-major alignment in memory.

### Task
Given an arbitrary multidimensional array $A \in \mathbb{R}^{d_0 \times d_1 \times \dots \times d_{k-1}}$, verify its storage layout using `flags.c_contiguous`:
- If $A$ is already C-contiguous, return $A$ directly without copying to preserve memory bandwidth.
- If $A$ is non-contiguous (such as a strided slice or Fortran-order transpose), return a newly allocated C-contiguous copy using `np.ascontiguousarray`.

---

## Exercise 2: Detecting Shared Memory Buffers (`check_memory_share`)

When slicing tensors or passing arrays across pipeline stages, unintended data mutations can occur if arrays silently share underlying memory buffers.

### Task
Given two arrays $X$ and $Y$, determine whether they point to overlapping memory addresses by evaluating buffer sharing (via `np.may_share_memory` or inspecting the `base` attribute). Return `True` if memory is shared, and `False` otherwise.

---

## Exercise 3: Inspecting Row Strides (`get_row_stride_bytes`)

The memory address of element $(i, j)$ in a 2D matrix $M \in \mathbb{R}^{N \times M}$ is calculated as:

$$\text{Address}(i, j) = \text{Base} + i \times s_0 + j \times s_1$$

where $s_0$ and $s_1$ represent the byte strides along axis $0$ and axis $1$ respectively.

### Task
Given a 2D matrix $M \in \mathbb{R}^{N \times M}$, return the exact integer number of bytes $s_0$ required to step from row $i$ to row $i+1$ by accessing the `strides` attribute.

---

## Verification & Testing
Run your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI
tforge check arraysmith advanced 01

# Run a specific test method within this lesson
tforge check arraysmith advanced 01 ensure_c_contiguous

# Check the entire arraysmith curriculum
tforge check arraysmith

# View your progress across all modules
tforge status
```

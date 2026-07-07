# Module: Vectorized Math & Reductions

## Learning Objectives
- Execute reduction kernels across arbitrary tensor dimensions without iterative aggregation.
- Master array broadcasting semantics to perform multidimensional arithmetic operations without looping.
- Zero-center multidimensional feature matrices along specific axes as a standard machine learning preprocessing stage.

---

## Concept Overview

Vectorized mathematical operations eliminate interpreter loop overhead by delegating element-wise transformations and reductions to compiled C kernels. Understanding how axes collapse during reductions and how trailing dimensions expand during broadcasting is essential for writing high-performance numerical software.

---

## Exercise 1: Row Reductions (`row_means`)

### Input Specifications
- **`matrix`**: A 2-D `numpy.ndarray` containing numeric values.
- **Shape**: `(N, M)` where $N \ge 1$ and $M \ge 1$.
- **Data Type**: Any standard numeric data type (`float32`, `float64`, `int32`, `int64`).

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(N,)` — a 1-D vector containing exactly one element per row of the input matrix.
- **Data Type**: Floating-point (`float64` or matching float precision of input).
- **Constraints**:
  - The $i$-th element of the returned vector must equal the arithmetic mean of the $i$-th row: $\mu_i = \frac{1}{M} \sum_{j=0}^{M-1} A_{i,j}$.
  - You must not use Python loops or iterate across rows or columns.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Column Normalization (`normalize_columns`)

### Input Specifications
- **`matrix`**: A 2-D `numpy.ndarray` containing floating-point values.
- **Shape**: `(N, M)` where $N \ge 1$ and $M \ge 1$.
- **Data Type**: Floating-point (`float32` or `float64`).

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(N, M)` matching the dimensions of the input matrix.
- **Data Type**: Identical to the input matrix's floating-point data type.
- **Constraints**:
  - Every element $X'_{i,j}$ in the returned matrix must equal the original element minus the arithmetic mean of column $j$: $X'_{i,j} = X_{i,j} - \frac{1}{N} \sum_{k=0}^{N-1} X_{k,j}$.
  - Each column in the returned matrix must have a mean of zero.
  - You must not modify the input matrix in place.
  - You must not use Python loops or iterate across columns or rows.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check arraysmith intermediate 01

# Verify a specific exercise
uv run tforge check arraysmith intermediate 01 -t row_means
```

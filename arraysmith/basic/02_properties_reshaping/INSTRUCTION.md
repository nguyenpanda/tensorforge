# Module: Properties & Reshaping

## Learning Objectives
- Inspect and query array dimensional metadata attributes without altering memory buffers.
- Transform linear vector representations into multidimensional matrix structures while preserving element ordering.
- Perform linear memory transformations and numerical precision casting across array pipelines.

---

## Concept Overview

Multidimensional arrays separate physical memory buffers from structural layout metadata. By modifying dimensional properties and strides, computational routines can interpret identical underlying memory blocks as matrices, tensors, or linear vectors without incurring data duplication costs.

---

## Exercise 1: Matrix Reshaping (`reshape_to_matrix`)

### Input Specifications
- **`arr`**: A 1-D `numpy.ndarray` containing numeric elements.
- **Shape**: `(12,)` — exactly 12 elements.
- **Data Type**: Any standard numeric data type.

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(3, 4)` — a 2-D matrix with 3 rows and 4 columns.
- **Data Type**: Identical to the input data type.
- **Constraints**:
  - The transformation must preserve standard row-major (C-order) element sequencing.
  - You must not use Python loops or manual index calculations.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Flattening and Precision Casting (`flatten_and_cast`)

### Input Specifications
- **`arr`**: A 2-D `numpy.ndarray` with integer elements.
- **Shape**: `(N, M)` where $N \ge 1$ and $M \ge 1$.
- **Data Type**: Any integer data type (`int8`, `int16`, `int32`, `int64`).

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(N * M,)` — a 1-D linear vector containing all elements from the matrix.
- **Data Type**: `numpy.float64` (64-bit floating-point precision).
- **Constraints**:
  - The elements must be ordered sequentially in standard row-major (C-order) layout.
  - You must not use Python loops, comprehensions, or manual element extraction.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check arraysmith basic 02

# Verify a specific exercise
uv run tforge check arraysmith basic 02 -t reshape_to_matrix
```

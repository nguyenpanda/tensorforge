# Module: Array Manipulation & Assembly

## Learning Objectives
- Stack multiple 1-D vectors along a new orthogonal axis to assemble multidimensional matrices.
- Concatenate existing tensors along shared dimensions without altering array ordering or dimensionality.
- Understand dimensional compatibility rules and axis alignment governing tensor assembly routines.

---

## Concept Overview

Assembling complex data structures from constituent vectors or matrices requires precise dimensional alignment. Stacking routines introduce new structural axes to combine lower-dimensional tensors, whereas concatenation routines join tensors along pre-existing shared dimensions.

---

## Exercise 1: Row Stacking (`stack_rows`)

### Input Specifications
- **`arrays`**: A `list` of 1-D `numpy.ndarray` objects: $[v_0, v_1, \dots, v_{k-1}]$.
- **Shape**: Exactly $k$ arrays ($k \ge 1$), where each array $v_i$ has shape `(M,)` ($M \ge 1$).
- **Data Type**: All constituent arrays share an identical standard numeric data type.

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(k, M)` — a 2-D matrix with $k$ rows and $M$ columns.
- **Data Type**: Identical to the input arrays' data type.
- **Constraints**:
  - Each input vector $v_i$ must form the $i$-th row of the returned matrix: $X_{i,j} = (v_i)_j$.
  - You must not use Python loops or comprehensions to iterate across array elements or rows.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Horizontal Concatenation (`concatenate_side_by_side`)

### Input Specifications
- **`left`**: A 2-D `numpy.ndarray` of shape `(N, M_1)` where $N \ge 1$ and $M_1 \ge 1$.
- **`right`**: A 2-D `numpy.ndarray` of shape `(N, M_2)` where $N \ge 1$ and $M_2 \ge 1$.
- **Data Type**: Both matrices share an identical standard numeric data type.

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(N, M_1 + M_2)` — a 2-D matrix with $N$ rows and $M_1 + M_2$ columns.
- **Data Type**: Identical to the input matrices' data type.
- **Constraints**:
  - The returned matrix must contain all columns from `left` followed horizontally by all columns from `right`.
  - You must not use Python loops or iterate across rows or columns.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check arraysmith intermediate 02

# Verify a specific exercise
uv run tforge check arraysmith intermediate 02 -t stack_rows
```

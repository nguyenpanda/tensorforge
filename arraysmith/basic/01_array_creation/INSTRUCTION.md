# Module: Array Creation

## Learning Objectives
- Initialize 1-D numerical vectors in contiguous memory without iterative loops.
- Understand explicit precision control and data type specification at array initialization time.
- Execute element-wise mathematical transformations over vectors leveraging compiled C routines.

---

## Concept Overview

In high-performance scientific computing, data arrays are stored in contiguous blocks of physical memory with uniform numerical data types. This structure avoids the object-boxing overhead and indirection typical of standard language collections, enabling vectorized array processing.

---

## Exercise 1: Integer Range Initialization (`create_integer_range`)

### Input Specifications
- **Arguments**: None (zero-argument method).

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(100,)` — a 1-D vector containing exactly 100 elements.
- **Data Type**: `numpy.int8` (8-bit signed integer).
- **Constraints**:
  - The array must contain the sequential integer progression from $0$ to $99$ inclusive: $[0, 1, 2, \dots, 98, 99]$.
  - You must not use Python loops, comprehensions, or generators.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Squared Sequence Initialization (`create_squared_range`)

### Input Specifications
- **Arguments**: None (zero-argument method).

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(100,)` — a 1-D vector containing exactly 100 elements.
- **Data Type**: `numpy.int64` (64-bit signed integer).
- **Constraints**:
  - The array must contain the squares of the first 100 non-negative integers: $s_i = i^2$ for $i \in \{0, 1, \dots, 99\}$.
  - You must not use Python loops, comprehensions, or generators.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check arraysmith basic 01

# Verify a specific exercise
uv run tforge check arraysmith basic 01 -t create_squared_range
```

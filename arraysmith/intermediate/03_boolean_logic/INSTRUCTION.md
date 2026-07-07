# Module: Boolean Logic & Conditional Array Operations

## Learning Objectives
- Evaluate vector-level boolean predicates and conditional expressions without iterative loops.
- Express complex conditional branch logic across multidimensional arrays via vectorized operations.
- Evaluate numerical floating-point equality within defined absolute and relative tolerance boundaries.

---

## Exercise 1: Conditional Discounting (`apply_discount`)

### Input Specifications
- **`prices`**: A 1-D or 2-D `numpy.ndarray` containing floating-point item prices.
- **Shape**: Arbitrary shape $(N,)$ or $(N, M)$ where $N, M \ge 1$.
- **Data Type**: `numpy.float64` (64-bit floating-point).
- **`threshold`**: A scalar `float` defining the price boundary above which discounts apply.
- **`discount_rate`**: A scalar `float` defining the fractional reduction rate (e.g., $0.20$ for a $20\%$ discount).

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: Exactly identical to the shape of `prices`.
- **Data Type**: `numpy.float64`.
- **Constraints**:
  - Every element $P'_{i,j}$ in the returned array must equal $P_{i,j} \times (1 - \text{discount\_rate})$ if $P_{i,j} > \text{threshold}$, and must equal $P_{i,j}$ otherwise.
  - You must not modify the input array in place.
  - You must not use Python loops or conditional branching statements (`if/else`) across individual elements.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Row-wise Floating-Point Equality (`identify_close_rows`)

### Input Specifications
- **`matrix_a`**: A 2-D `numpy.ndarray` containing floating-point values.
- **`matrix_b`**: A 2-D `numpy.ndarray` containing floating-point values.
- **Shape**: Both matrices share identical dimensions $(N, M)$ where $N \ge 1$ and $M \ge 1$.
- **Data Type**: Floating-point (`float32` or `float64`).
- **`rtol`**: A scalar `float` defining the relative tolerance parameter.

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(N,)` — a 1-D boolean vector containing exactly one element per row of the input matrices.
- **Data Type**: `numpy.bool_` (boolean).
- **Constraints**:
  - The $i$-th element of the returned vector must be `True` if and only if all corresponding elements in row $i$ of both matrices are approximately equal within relative tolerance `rtol`.
  - You must not use Python loops or iterate across rows or columns.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check arraysmith intermediate 03

# Verify a specific exercise
uv run tforge check arraysmith intermediate 03 -t apply_discount
```

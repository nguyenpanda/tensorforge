# Module: Indexing & Slicing

## Learning Objectives
- Filter data elements using conditional logical masking across vectors without explicit iteration.
- Extract discrete elements from arbitrary memory indices using discrete integer array subscripts.
- Differentiate between zero-copy memory slices and independent data copies resulting from advanced subscripting.

---

## Concept Overview

Advanced subscripting in scientific arrays allows selective data extraction and reordering in memory. By supplying logical mask vectors or discrete integer indices, computational routines can extract subsets of data directly at the C-kernel level without incurring iterative interpreter overhead.

---

## Exercise 1: Threshold Filtering (`filter_above_threshold`)

### Input Specifications
- **`arr`**: A 1-D `numpy.ndarray` containing numeric elements.
- **Shape**: `(N,)` where $N \ge 1$.
- **Data Type**: Any standard numeric data type.
- **`threshold`**: A scalar `float` or `int` comparison value.

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(K,)` where $0 \le K \le N$ is the number of elements strictly exceeding the threshold.
- **Data Type**: Identical to the input array's data type.
- **Constraints**:
  - The returned array must contain only those elements $x_i$ from `arr` such that $x_i > \text{threshold}$.
  - The relative ordering of elements must be preserved.
  - You must not modify the input array.
  - You must not use Python loops or comprehensions.

Call `show_hint()` if you are stuck.

---

## Exercise 2: Index Gathering (`gather_by_indices`)

### Input Specifications
- **`arr`**: A 1-D source `numpy.ndarray` containing numeric elements.
- **Shape**: `(N,)` where $N \ge 1$.
- **Data Type**: Any standard numeric data type.
- **`indices`**: A 1-D `numpy.ndarray` of integer subscripts.
- **Shape**: `(K,)` where $K \ge 1$, containing valid index positions $0 \le I_j < N$.

### Output Specifications
- **Return Type**: `numpy.ndarray`.
- **Shape**: `(K,)` matching the dimension of `indices`.
- **Data Type**: Identical to `arr`'s data type.
- **Constraints**:
  - The returned array must contain the gathered elements $y_j = a_{I_j}$ for $j \in \{0, 1, \dots, K-1\}$.
  - Duplicate index accesses are permitted and must return the corresponding element at each occurrence.
  - You must not use Python loops or element-by-element iteration.

Call `show_hint()` if you are stuck.

---

## Verification & Testing
Execute the verification suite via the TensorForge CLI:

```bash
# Verify the full module
uv run tforge check arraysmith basic 03

# Verify a specific exercise
uv run tforge check arraysmith basic 03 -t filter_above_threshold
```

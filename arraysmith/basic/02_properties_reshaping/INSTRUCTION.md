# Module 01 — Properties & Reshaping

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:

- Inspect an array's **shape** ($N \times M$), **dtype**, **ndim**, and **size** attributes.
- Reshape a 1-D vector $v \in \mathbb{R}^K$ into any compatible multidimensional matrix $A \in \mathbb{R}^{N \times M}$ where $N \times M = K$ using `reshape()`.
- Flatten any N-D tensor back to a 1-D vector using `flatten()`.
- Cast array precision to different numerical domains using `astype()`.
- Differentiate between zero-copy **views** and independent data **copies**.

---

## 2. Concept Introduction

### 2.1 Array Properties

Every `ndarray` exposes read-only metadata properties instantaneously without recomputing data:

```python
arr = np.arange(12).reshape(3, 4)

arr.shape   # (3, 4)     <- tuple representing dimensions N x M
arr.ndim    # 2          <- number of dimensions (axes)
arr.size    # 12         <- total element count N * M
arr.dtype   # int64      <- element data type
arr.nbytes  # 96         <- total memory consumption in bytes (12 * 8 bytes)
```

### 2.2 `reshape()` — Transforming Dimensions Without Copying

`reshape()` returns a **view** (whenever possible) over existing memory buffers. The total number of elements must remain invariant ($N \times M = K$):

```
Original vector v (12 elements, shape=(12,)):
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

After reshape(3, 4) — identical buffer, new matrix interpretation A in R^(3 x 4):
[[ 0,  1,  2,  3],  <- row 0
 [ 4,  5,  6,  7],  <- row 1
 [ 8,  9, 10, 11]]  <- row 2
shape = (3, 4)
```

Using `-1` as a wildcard dimension instructs NumPy to deduce the axis size automatically:

```python
arr.reshape(3, -1)   # automatically infers columns M = 4 when size is 12
arr.reshape(-1)      # flattens matrix to a 1-D vector of shape (12,)
```

### 2.3 `flatten()` vs `ravel()`

| Method | Return Behavior | Contiguous Guarantee | Memory Isolation |
|---|---|---|---|
| `arr.flatten()` | Allocates a **new copy** | Always C-contiguous | Completely safe to mutate; no aliasing |
| `arr.ravel()` | Returns a **view** if possible | Usually contiguous | Zero-copy speed, but mutates parent buffer |

For our exercises, prioritize `flatten()` when generating clean independent vector representations.

### 2.4 `astype()` — Precision Casting

```python
arr = np.array([1, 2, 3], dtype=np.int32)

# Casts values to floating-point representation
float_arr = arr.astype(np.float64)  # -> array([1., 2., 3.], dtype=float64)
```

You can cleanly chain method calls across expressions:

```python
arr.flatten().astype(np.float64)   # flatten and cast in a single vectorized pipeline
```

---

## 3. Exercises

Open `student_code.py` and implement the two methods below.

> **Rule:** No Python `for`-loops or iteration constructs. Enforced by static AST analysis and performance benchmarking.

---

### Exercise 1 — `reshape_to_matrix(arr)`

**Task:** Given a 1-D vector $v \in \mathbb{R}^{12}$, transform it into a 2-D matrix $M \in \mathbb{R}^{3 \times 4}$:

$$M_{i,j} = v_{4i + j} \quad \text{for } i \in \{0, 1, 2\}, \; j \in \{0, 1, 2, 3\}$$

```
Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]   shape=(12,)

Output: [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]                          shape=(3, 4)
```

**API to use:** `arr.reshape(rows, cols)`

---

### Exercise 2 — `flatten_and_cast(arr)`

**Task:** Given a 2-D integer matrix $A \in \mathbb{Z}^{N \times M}$ of arbitrary shape, flatten it to a 1-D vector $v \in \mathbb{R}^{N \times M}$ and cast every element to floating-point precision `float64`:

$$v_{M i + j} = \text{float64}(A_{i,j})$$

```
Input:  [[1, 2],    dtype=int32
         [3, 4]]

Output: [1., 2., 3., 4.]   dtype=float64, shape=(4,)
```

**Strategy:** Chain `flatten()` and `astype(np.float64)` directly.

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI
tforge check arraysmith basic 02

# Run a specific test method within this lesson
tforge check arraysmith basic 02 reshape_to_matrix

# Check the entire arraysmith curriculum
tforge check arraysmith

# View your progress across all modules
tforge status
```

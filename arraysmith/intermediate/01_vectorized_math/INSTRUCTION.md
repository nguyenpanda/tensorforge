# Module 03 — Vectorized Math

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:

- Execute reduction kernels (`np.sum`, `np.mean`) across arbitrary tensor dimensions $N \times M$ using the `axis` argument.
- Master array **broadcasting** semantics to perform vectorized operations without iterative loops.
- Zero-center multidimensional datasets column-wise — a standard prerequisite for machine learning pipelines.

---

## 2. Concept Introduction

### 2.1 Reduction Functions and Axis Mechanics

NumPy reduction routines (`np.sum`, `np.mean`, `np.max`) collapse dimensions of a matrix $A \in \mathbb{R}^{N \times M}$. The `axis` parameter specifies which dimension is collapsed:

$$\mu_i = \frac{1}{M} \sum_{j=0}^{M-1} A_{i,j} \quad (\text{axis} = 1, \text{ row means})$$

$$\mu_j = \frac{1}{N} \sum_{i=0}^{N-1} A_{i,j} \quad (\text{axis} = 0, \text{ column means})$$

```
matrix = [[1, 2, 3],       shape = (2, 3)
          [4, 5, 6]]
           ^   ^   ^
          col0 col1 col2

axis=0  ->  collapse ROWS   ->  result shape (3,)   [mean across rows for each column]
axis=1  ->  collapse COLUMNS ->  result shape (2,)   [mean across columns for each row]
```

### 2.2 Broadcasting Mechanics

When subtracting a column mean vector $\mu \in \mathbb{R}^M$ from a matrix $A \in \mathbb{R}^{N \times M}$, NumPy automatically **broadcasts** the lower-dimensional array across compatible dimensions:

$$A'_{i,j} = A_{i,j} - \mu_j \quad \forall i \in \{0, \dots, N-1\}, \; j \in \{0, \dots, M-1\}$$

```
matrix    shape (3, 4)   [[a, b, c, d],
col_means shape    (4,)   [m0, m1, m2, m3] -> automatically stretched across all 3 rows

matrix - col_means -> identical output shape (3, 4)
```

Broadcasting alignment rules:
1. Dimensions are aligned and compared from right to left.
2. Dimensions of size 1 (or missing trailing axes) are stretched to match the partner dimension.
3. Incompatible dimensions raise a `ValueError`.

### 2.3 Why Avoid Loops?

```python
# ❌ Python loop — strictly forbidden by AST policy
result = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    result[i] = matrix[i] - col_means

# ✅ Broadcasting — executed inside optimized C kernels
result = matrix - col_means
```

---

## 3. Exercises

Open `student_code.py` and implement the two methods below.

> **Rule:** No Python `for`-loops or iteration constructs. Enforced by static AST analysis and performance benchmarking.

---

### Exercise 1 — `row_means(matrix)`

**Task:** Given matrix $A \in \mathbb{R}^{N \times M}$, return a 1-D vector $\mu \in \mathbb{R}^N$ containing the mean of each row:

$$\mu_i = \frac{1}{M} \sum_{j=0}^{M-1} A_{i,j}$$

```
matrix = [[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]]   shape=(2, 3)

Output = [2.0, 5.0]          shape=(2,)
```

**API to use:** `np.mean(matrix, axis=1)`

---

### Exercise 2 — `normalize_columns(matrix)`

**Task:** Given matrix $X \in \mathbb{R}^{N \times M}$, subtract the column mean vector $\mu \in \mathbb{R}^M$ from every element such that each column in the resulting matrix $X'$ has zero mean:

$$X'_{i,j} = X_{i,j} - \frac{1}{N} \sum_{k=0}^{N-1} X_{k,j}$$

```
matrix = [[1.0, 10.0],
          [3.0, 20.0],
          [5.0, 30.0]]   shape=(3, 2)

col_means = [3.0, 20.0]  shape=(2,)

Output = [[-2.0, -10.0],
          [ 0.0,   0.0],
          [ 2.0,  10.0]]
```

**Strategy:**
1. Compute column means `col_means = np.mean(matrix, axis=0)`.
2. Evaluate `matrix - col_means` via broadcasting.

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI
tforge check arraysmith intermediate 01

# Run a specific test method within this lesson
tforge check arraysmith intermediate 01 row_means

# Check the entire arraysmith curriculum
tforge check arraysmith

# View your progress across all modules
tforge status
```

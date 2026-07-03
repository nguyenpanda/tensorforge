# Module 02 — Indexing & Slicing

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:

- Apply **boolean masking** to filter elements across vectors $v \in \mathbb{R}^n$ and matrices without explicit iteration.
- Utilize **fancy indexing** with integer arrays to gather discrete elements at arbitrary positions.
- Understand when indexing operations generate zero-copy **views** versus independent data **copies**.
- Replace multi-line Python conditional loops with concise, vectorized expressions.

---

## 2. Concept Introduction

### 2.1 Basic Indexing & Slicing

Standard scalar indexing and slicing return zero-copy views over existing RAM buffers:

```python
arr = np.array([10, 20, 30, 40, 50])

arr[2]       # 30            <- scalar extraction
arr[1:4]     # [20, 30, 40] <- slice returning a view
arr[-1]      # 50            <- negative indexing
```

### 2.2 Boolean Masking

A **boolean mask** is a logical array $m \in \{0, 1\}^n$ of identical shape to the target array. Indexing with a mask extracts only values corresponding to True positions:

$$y = \{x_i \mid m_i = \text{True}\}$$

```
arr   = [ 10, -3,  5, -8, 22,  0 ]
mask  = [  T   F   T   F   T   F ]   <- mask defined by condition: arr > 0

arr[mask] -> [ 10,  5, 22 ]
```

This executes orders of magnitude faster than iterative Python filtering:

```python
# ✅ Vectorized C-speed evaluation
filtered = arr[arr > 0]

# ❌ Loop (slow — strictly forbidden by AST policy)
filtered = []
for x in arr:
    if x > 0:
        filtered.append(x)
```

Chaining compound logical conditionals requires bitwise operators `&` (AND) and `|` (OR):

```python
arr[(arr > 0) & (arr < 20)]   # elements inside open interval (0, 20)
arr[(arr < -5) | (arr > 50)]  # elements outside interval [-5, 50]
```

> ⚠️ Always use `&` and `|` instead of standard Python `and` and `or`, which cannot broadcast across array elements.

### 2.3 Fancy Indexing (Integer Array Indexing)

Passing an array of integers as subscripts allows arbitrary gathering and reordering of elements in memory:

```
arr     = [ 10, 20, 30, 40, 50 ]
            ^0  ^1  ^2  ^3  ^4

indices = [ 4,  0,  2 ]

arr[indices] -> [ 50, 10, 30 ]
```

Key characteristics:
- Always allocates and returns a **new data copy**.
- Supports repeated element gathering: `arr[[0, 0, 1]]` $\rightarrow [10, 10, 20]$.
- Output length matches the index array dimension `len(indices)`.

---

## 3. Exercises

Open `student_code.py` and implement the two methods below.

> **Rule:** No Python `for`-loops or iteration constructs. Enforced by static AST analysis and performance benchmarking.

---

### Exercise 1 — `filter_above_threshold(arr, threshold)`

**Task:** Given a vector $v \in \mathbb{R}^n$ and a scalar threshold $T$, return a new vector containing all elements strictly greater than $T$:

$$y = \{v_i \mid v_i > T\}$$

```
arr       = [-3, -1,  0,  2,  5,  8]
threshold = 1.0

Output    = [2, 5, 8]
```

**Strategy:** Construct boolean mask `mask = arr > threshold` and evaluate `arr[mask]` directly.

---

### Exercise 2 — `gather_by_indices(arr, indices)`

**Task:** Given source array $a \in \mathbb{R}^n$ and integer indices $I \in \mathbb{Z}^k$, return gathered elements $y \in \mathbb{R}^k$:

$$y_j = a_{I_j} \quad \forall j \in \{0, 1, \dots, k-1\}$$

```
arr     = [10, 20, 30, 40, 50]
indices = [ 4,  0,  2]

Output  = [50, 10, 30]
```

**Strategy:** Pass `indices` directly into subscript syntax `arr[indices]`.

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI
tforge check arraysmith basic 03

# Run a specific test method within this lesson
tforge check arraysmith basic 03 filter_above_threshold

# Check the entire arraysmith curriculum
tforge check arraysmith

# View your progress across all modules
tforge status
```

# Module 00 — Array Creation

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:

- Create 1-D NumPy vectors $v \in \mathbb{R}^n$ using `np.arange`, `np.linspace`, `np.zeros`, and `np.ones`.
- Specify the **dtype** of an array at creation time.
- Apply element-wise mathematical operations across dimensions $N \times M$ with **no explicit loops**.
- Understand why vectorized kernels execute orders of magnitude faster than Python iteration.

---

## 2. Concept Introduction

### 2.1 What is a NumPy Array?

A NumPy `ndarray` is a fixed-size, homogeneously-typed grid of numeric values stored in a contiguous block of RAM. Unlike a Python list where each item is a separately allocated boxed object, all elements in an array share a single data type $\text{dtype}$, enabling vectorized C/Fortran execution.

```
Python list  →  [ptr, ptr, ptr, ...]   each ptr → boxed Python number object
NumPy array  →  [val, val, val, ...]   raw numeric bytes, zero boxing overhead
```

### 2.2 Core Creation Functions

| Function | Mathematical Output | Typical Use Case |
|---|---|---|
| `np.arange(start, stop, step)` | Sequence $x_i = \text{start} + i \times \text{step}$ | `np.arange(5)` $\rightarrow [0, 1, 2, 3, 4]$ |
| `np.linspace(start, stop, n)` | $n$ evenly-spaced floats in $[\text{start}, \text{stop}]$ | `np.linspace(0, 1, 5)` $\rightarrow [0., 0.25, 0.5, 0.75, 1.]$ |
| `np.zeros(shape)` | Tensor of zeros $0 \in \mathbb{R}^{d_0 \times d_1 \times \dots}$ | `np.zeros(3)` $\rightarrow [0., 0., 0.]$ |
| `np.ones(shape)` | Tensor of ones $1 \in \mathbb{R}^{d_0 \times d_1 \times \dots}$ | `np.ones(3)` $\rightarrow [1., 1., 1.]$ |

### 2.3 Controlling Dtype

Every array enforces a specific data type ($\text{dtype}$). You can set this explicitly at initialization:

```python
np.arange(5)                   # dtype=int64   (default on 64-bit systems)
np.arange(5, dtype=np.int8)    # dtype=int8    (values -128 to 127)
np.arange(5, dtype=np.float32) # dtype=float32
```

You can also cast an existing array to a new precision:

```python
arr = np.arange(5)
arr.astype(np.float64)         # allocates and returns a new float64 array
```

### 2.4 Element-Wise Operations (No Loops!)

NumPy overloads standard arithmetic operators to evaluate element-wise across the entire array:

$$y_i = x_i^2 \quad \forall i \in \{0, 1, \dots, n-1\}$$

```python
arr = np.array([0, 1, 2, 3, 4])

# ✅ Vectorized C-speed evaluation
result = arr ** 2  # -> array([0, 1, 4, 9, 16])

# ❌ Python loop (slow — strictly forbidden by AST policy)
result = []
for x in arr:
    result.append(x ** 2)
```

The vectorized operation delegates computation directly to compiled C kernels and executes **10–100× faster** than iterative Python loops.

---

## 3. Exercises

Open `student_code.py` and implement the two methods below.

> **Rule:** You MUST NOT use Python `for`-loops, `range()`, or list comprehensions. Our static AST policy scanner will fail your submission if iteration constructs are detected.

---

### Exercise 1 — `create_integer_range()`

**Task:** Return a 1-D vector $v \in \mathbb{Z}^{100}$ containing integers $0$ through $99$ inclusive with data type `int8`:

$$v = [0, 1, 2, \dots, 98, 99]^T$$

```
Expected output:
  array([ 0,  1,  2, ..., 97, 98, 99], dtype=int8)
  shape : (100,)
  dtype : int8
```

**API to use:** `np.arange()`

---

### Exercise 2 — `create_squared_range()`

**Task:** Return a 1-D vector $s \in \mathbb{Z}^{100}$ representing the first 100 perfect squares:

$$s_i = i^2 \quad \forall i \in \{0, 1, \dots, 99\}$$

```
Expected output:
  array([   0,    1,    4,    9,   16, ..., 9801])
  shape : (100,)
  values: index i -> i^2
```

**Strategy:**
1. Generate sequence $i \in \{0, \dots, 99\}$ using a single NumPy call.
2. Apply the exponentiation operator `**` across the entire array.

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI
tforge check arraysmith basic 01

# Run a specific test method within this lesson
tforge check arraysmith basic 01 create_squared_range

# Check the entire arraysmith curriculum
tforge check arraysmith

# View your progress across all modules
tforge status
```

# Module 04 — Array Manipulation

## Table of Contents
1. [Objectives](#1-objectives)
2. [Concept Introduction](#2-concept-introduction)
3. [Exercises](#3-exercises)
4. [Running Your Tests](#4-running-your-tests)

---

## 1. Objectives

By the end of this module you will be able to:

- Stack multiple 1-D vectors along a new orthogonal axis to construct matrices using `np.vstack`.
- Concatenate existing tensors along shared dimensions using `np.concatenate`.
- Select the appropriate `axis` parameter for row-wise versus column-wise joining.
- Understand dimensional compatibility rules governing tensor assembly.

---

## 2. Concept Introduction

### 2.1 Stacking vs Concatenating

| Operation | Function | Dimensional Effect |
|---|---|---|
| Stacking | `np.vstack` | Introduces a new axis ($N$ vectors of length $M \rightarrow$ matrix $N \times M$) |
| Concatenating | `np.concatenate` | Joins along an existing axis without changing array order ($\text{ndim}$ constant) |

### 2.2 `np.vstack` — Vertical Stacking

`vstack` treats each input vector $v_i \in \mathbb{R}^M$ as a row and stacks them vertically to generate matrix $A \in \mathbb{R}^{N \times M}$:

$$A = \begin{bmatrix} v_0^T \\ v_1^T \\ \vdots \\ v_{N-1}^T \end{bmatrix}$$

```
a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]

np.vstack([a, b, c])
-> [[1, 2, 3],    shape (3, 3)
    [4, 5, 6],
    [7, 8, 9]]
```

### 2.3 `np.concatenate` — Axis-Controlled Joining

`concatenate` joins matrices $A \in \mathbb{R}^{N \times M_1}$ and $B \in \mathbb{R}^{N \times M_2}$ along existing axis $d$:

$$\text{axis} = 0 \implies \text{stack rows vertically (requires matching column count } M)$$

$$\text{axis} = 1 \implies \text{join columns horizontally (requires matching row count } N)$$

```
left  = [[1, 2],     shape (2, 2)
         [3, 4]]

right = [[5, 6, 7],  shape (2, 3)
         [8, 9, 0]]

np.concatenate([left, right], axis=1)
-> [[1, 2, 5, 6, 7],   shape (2, 5)
    [3, 4, 8, 9, 0]]
```

### 2.4 Shape Compatibility Rules

```
np.vstack(arrays)
  ✅ All input vectors share identical dimension size M
  ❌ Mismatched lengths raise ValueError

np.concatenate([A, B], axis=1)
  ✅ Matrices share identical row dimension N (A.shape[0] == B.shape[0])
  ❌ Mismatched row counts raise ValueError
```

---

## 3. Exercises

Open `student_code.py` and implement the two methods below.

> **Rule:** No Python `for`-loops or iteration constructs. Enforced by static AST analysis and performance benchmarking.

---

### Exercise 1 — `stack_rows(arrays)`

**Task:** Given a sequence of 1-D vectors $v_0, v_1, \dots, v_{k-1} \in \mathbb{R}^M$, assemble them into a matrix $X \in \mathbb{R}^{k \times M}$:

$$X_{i,j} = (v_i)_j$$

```
arrays = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 0]]

Output = [[1, 2, 3, 4, 5],   shape=(2, 5)
          [6, 7, 8, 9, 0]]
```

**API to use:** `np.vstack(arrays)`

---

### Exercise 2 — `concatenate_side_by_side(left, right)`

**Task:** Given matrices $L \in \mathbb{R}^{N \times M_1}$ and $R \in \mathbb{R}^{N \times M_2}$, join them horizontally to produce matrix $Y \in \mathbb{R}^{N \times (M_1 + M_2)}$:

$$Y = \begin{bmatrix} L & R \end{bmatrix}$$

```
left  = [[1, 2],         right = [[5, 6, 7],
         [3, 4]]                  [8, 9, 0]]

Output = [[1, 2, 5, 6, 7],   shape=(2, 5)
          [3, 4, 8, 9, 0]]
```

**API to use:** `np.concatenate([left, right], axis=1)`

---

## 4. Running Your Tests

Execute your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI
tforge check arraysmith intermediate 02

# Run a specific test method within this lesson
tforge check arraysmith intermediate 02 stack_rows

# Check the entire arraysmith curriculum
tforge check arraysmith

# View your progress across all modules
tforge status
```

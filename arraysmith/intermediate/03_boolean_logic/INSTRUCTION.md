# Module 05: Boolean Logic & Conditional Array Operations

## Learning Objectives
- Master vector-level boolean evaluations using `np.where`, `np.isclose`, `np.all`, and `np.any`.
- Avoid Python iterative loops by expressing conditional logic as boolean array masks over dimensions $N \times M$.
- Understand floating-point precision comparisons and tolerance thresholds defined by $f(a, b) \le \text{atol} + \text{rtol} \times |b|$.

---

## Exercise 1: Conditional Discounting (`apply_discount`)

In quantitative finance and retail analytics, pricing algorithms frequently apply conditional adjustments across large product catalogs without iterative branching.

### Task
Given a matrix of item prices $P \in \mathbb{R}^{N \times M}$, apply a fractional discount $d$ (e.g., $d = 0.20$ for a $20\%$ discount) to all prices strictly exceeding a threshold $T$:

$$P_{i,j}' = \begin{cases} P_{i,j} \times (1 - d) & \text{if } P_{i,j} > T \\ P_{i,j} & \text{otherwise} \end{cases}$$

You **must** use `np.where` to construct the resulting price array $P'$.

---

## Exercise 2: Row-wise Floating-Point Equality (`identify_close_rows`)

When comparing numerical outputs from floating-point matrix operations or neural network tensors, exact equality ($A == B$) often fails due to machine epsilon $\epsilon$. 

### Task
Given two matrices $A, B \in \mathbb{R}^{N \times M}$, identify which rows are approximately identical within relative tolerance $\text{rtol}$ (where $|A_{i,j} - B_{i,j}| \le \text{rtol} \times |B_{i,j}|$). Return a boolean vector $v \in \{0, 1\}^N$ such that:

$$v_i = \bigwedge_{j=1}^{M} \text{isclose}(A_{i,j}, B_{i,j}, \text{rtol})$$

You **must** use `np.isclose` combined with `np.all` across axis $d = 1$.

---

## Verification & Testing
Run your verification suite using the TensorForge CLI:

```bash
# Check this lesson using the CLI
tforge check arraysmith intermediate 03

# Run a specific test method within this lesson
tforge check arraysmith intermediate 03 apply_discount

# Check the entire arraysmith curriculum
tforge check arraysmith

# View your progress across all modules
tforge status
```

# Contributing to TensorForge 🛠️

Thank you for your interest in contributing to **TensorForge**! As an interactive, professional-grade laboratory training ground for scientific computing, deep learning, and HPC, TensorForge relies on well-structured, rigorous curriculum lessons.

This guide explains the architectural conventions and step-by-step process for adding new lessons to any of the three core curriculum modules (`arraysmith`, `tensorsmith`, or `hpcsmith`).

---

## Table of Contents

1. [Curriculum Architecture & Tier-Scoped Numbering](#curriculum-architecture--tier-scoped-numbering)
2. [Required File Structure for a New Lesson](#required-file-structure-for-a-new-lesson)
3. [Step-by-Step Guide to Adding a Lesson](#step-by-step-guide-to-adding-a-lesson)
4. [Writing Specifications (`INSTRUCTION.md`)](#writing-specifications-instructionmd)
5. [Enforcing Clean Lab & AST Policies](#enforcing-clean-lab--ast-policies)
6. [Registering Context-Aware Hints](#registering-context-aware-hints)
7. [Code Quality & Testing Checklist](#code-quality--testing-checklist)

---

## Curriculum Architecture & Tier-Scoped Numbering

TensorForge enforces a **Clean Lab Architecture**, separating student workspaces from verification test suites and reference baselines. Furthermore, lessons are organized using **Tier-Scoped Numbering** across four tiers:

- `basic/`
- `intermediate/`
- `advanced/`
- `applications/`

Within each tier, lesson directories are prefixed with a two-digit integer that resets to `01` per tier (e.g., `01_array_creation`, `02_properties_reshaping`). This prevents sequential numbering ripples when lessons are re-ordered or inserted.

---

## Required File Structure for a New Lesson

When contributing a new lesson to any curriculum module (e.g., `arraysmith/basic/04_new_topic`), you must establish the following required file structure across the workspace and test verification trees:

```bash
tensorforge/
├── <module>/<tier>/<lesson_id>_<lesson_name>/
│   ├── INSTRUCTION.md       ← Student-facing specification & mathematical rules
│   └── student_code.py      ← Skeleton implementation with NotImplementedError
└── tests/<module>/<tier>/<lesson_id>_<lesson_name>/
    ├── solution.py          ← Optimized golden reference baseline (or _baseline.py)
    └── test_<lesson_id>.py  ← Automated pytest verification & benchmark suite
```

### File Responsibilities

1. **`student_code.py`**: Contains the class and method skeletons where students write their code. Function signatures, type hints, and docstrings must be 100% complete. The implementation body must be stripped and replaced with:

   ```python
   raise NotImplementedError("TODO: Implement this function")
   ```

2. **`solution.py`** (or `_baseline.py` in the verification suite): Contains the state-of-the-art reference implementation. This code is read-only for students and serves as the golden standard for numerical accuracy, execution speed benchmarking, and memory footprint profiling.
3. **`INSTRUCTION.md`**: Detailed Markdown specification for the exercise. Must clearly define objectives, mathematical formulas (using LaTeX), AST restrictions, and expected terminal outputs.
4. **`test_*.py`**: Automated verification test suite using pytest. Must check correctness across edge cases, enforce AST policies, and invoke `compare_and_benchmark`.

---

## Step-by-Step Guide to Adding a Lesson

### 1. Scaffold the Lesson Directory

You can use the automated scaffolding CLI to generate clean workspace and test directory trees:

```bash
tforge generate arraysmith basic 04_matrix_algebra
```

### 2. Implement the Golden Reference (`solution.py` / `_baseline.py`)

Write the optimized reference implementation inside the test verification folder (`tests/<module>/<tier>/<lesson_name>/`). Ensure your baseline avoids unnecessary memory allocations and achieves optimal C/Fortran vectorization speed.

### 3. Create the Student Skeleton (`student_code.py`)

In the student workspace folder (`<module>/<tier>/<lesson_name>/`), create `student_code.py`. Mirror the class and function signatures from the reference baseline, but replace all logic with:

```python
raise NotImplementedError("TODO: Implement this function")
```

### 4. Build the Verification Suite (`test_*.py`)

In the test directory, create `test_<lesson_id>.py` (e.g., `test_04.py`). Structure your tests as follows:

- For Python/NumPy tests, import `@ast_policy` from `forge_core.ast_validator` to forbid loops.
- For PyTorch/CUDA tests (`tensorsmith` / `hpcsmith`), always add:

  ```python
  import pytest
  import torch
  
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
  class TestMyLesson:
      ...
  ```

- Invoke `compare_and_benchmark(student_fn=..., baseline_fn=...)` to test both correctness and speed/memory performance.

---

## Writing Specifications (`INSTRUCTION.md`)

The `INSTRUCTION.md` file must be thorough, professional, and mathematically precise.

### Terminal Formatting Rules
When showing terminal CLI commands, code snippets, or simulated output hints in instructions, you **MUST NOT** use raw ANSI escape sequences. Always reference and mandate the official color library:
```python
from nguyenpanda.swan import c24, reset
```

---

## Enforcing Clean Lab & AST Policies

To prevent students from bypassing vectorization requirements with brute-force Python loops, decorate test verification methods with `@ast_policy`:

```python
from forge_core.ast_validator import ast_policy

@ast_policy(forbid_loops=True, required_calls=["numpy.where"])
def test_exercise_correctness():
    ...
```

---

## Registering Context-Aware Hints

To provide non-blocking assistance when students invoke `show_hint()`, register your lesson in the `hint/` package:

1. Navigate to `hint/hint/<module>/<tier>/`.
2. Create `hint_<lesson_id>_<lesson_name>.py`.
3. Define the composite key and hint mapping:

   ```python
   MODULE = "<tier>/<lesson_id>_<lesson_name>"
   
   HINT = {
       "ClassName": {
           "method_name": {
               "major": "Use np.dot or @ operator for matrix multiplication.",
               "minor": "Ensure the inner dimensions match: (M, K) @ (K, N) -> (M, N)."
           }
       }
   }
   ```

---

## Using DatasetManager for Benchmark Datasets

When authoring lessons that require external data (such as MNIST images, corpus text, or pre-computed benchmark matrices), use the centralized `DatasetManager` utility from `forge_core`. It guarantees atomic downloads (via `.tmp` file rename) and SHA-256 checksum verification, caching files locally to `./.tensorforge_datasets/`:

```python
from forge_core import DatasetManager

# Initialize manager (defaults to ./.tensorforge_datasets/)
dm = DatasetManager()

# Atomically download and verify dataset
dataset_path = dm.get_dataset(
    url="https://github.com/nguyenpanda/tensorforge/releases/download/v1.0/sample_data.bin",
    expected_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
)
```

---

## Code Quality & Testing Checklist

Before submitting a pull request, run the following automated checks to guarantee structural integrity:

```bash
# 1. Format code using Ruff
uv run ruff format .

# 2. Run static linting and structural analysis
uv run tforge lint
uv run ruff check .

# 3. Verify project infrastructure health
uv run tforge check infra
uv run tforge validate

# 4. Execute full test suite
uv run pytest tests/ -v
```

All tests must pass cleanly without warnings or errors. Thank you for building the future of scientific computing education!

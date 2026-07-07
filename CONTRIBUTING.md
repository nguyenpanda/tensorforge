# Contributing to TensorForge 🛠️

Thank you for contributing to **TensorForge**. This framework serves as an interactive, professional-grade training laboratory for scientific computing, deep learning, and high-performance GPU kernel engineering.

This document outlines the architectural conventions and requirements for contributing new lessons to the core curriculum modules (`arraysmith`, `tensorsmith`, and `hpcsmith`).

---

## Table of Contents

1. [Curriculum Architecture & Tier-Scoped Numbering](#curriculum-architecture--tier-scoped-numbering)
2. [Required File Structure](#required-file-structure)
3. [Lesson Authoring Workflow](#lesson-authoring-workflow)
4. [Writing Specifications (`INSTRUCTION.md`)](#writing-specifications-instructionmd)
5. [Enforcing AST Policies](#enforcing-ast-policies)
6. [Registering Context-Aware Hints](#registering-context-aware-hints)
7. [Dataset Management & Atomic Downloads](#dataset-management--atomic-downloads)
8. [Verification & Testing Checklist](#verification--testing-checklist)

---

## Curriculum Architecture & Tier-Scoped Numbering

TensorForge enforces a **Clean Lab Architecture**, isolating student implementation workspaces from automated verification suites and reference baselines.

Lessons are structured across four progressive tiers:
- `basic/`
- `intermediate/`
- `advanced/`
- `applications/`

Within each tier, lesson directories use two-digit integer prefixes starting at `01` (e.g., `01_array_creation`, `02_properties_reshaping`). Numbering resets to `01` within each tier to prevent sequential numbering conflicts across tiers.

---

## Required File Structure

When contributing a new lesson (e.g., `arraysmith/basic/04_matrix_algebra`), you must establish the following structure across the workspace and test trees:

```bash
tensorforge/
├── <module>/<tier>/<lesson_id>_<lesson_name>/
│   ├── INSTRUCTION.md       ← Student specification & mathematical rules
│   └── student_code.py      ← Skeleton implementation with NotImplementedError
└── tests/<module>/<tier>/<lesson_id>_<lesson_name>/
    ├── solution.py          ← Optimized golden reference baseline
    └── test_<lesson_id>.py  ← Automated pytest verification & benchmark suite
```

### File Responsibilities

1. **`student_code.py`**: Contains class and method skeletons for student implementation. All function signatures, type hints, and PEP 257 docstrings must be complete. The implementation body must be replaced with:
   ```python
   raise NotImplementedError("TODO: Implement this function")
   ```
2. **`solution.py`**: Contains the optimized reference implementation. This code serves as the golden standard for numerical correctness, execution speed benchmarking, and memory footprint profiling.
3. **`INSTRUCTION.md`**: Student-facing Markdown specification. Must define objectives, input/output types, shapes, constraints, mathematical formulas (using LaTeX), and AST restrictions. Must include: `"Call show_hint() if you are stuck."`
4. **`test_<lesson_id>.py`**: Automated verification test suite using `pytest`. Must verify correctness across edge cases, enforce AST policies, and invoke `compare_and_benchmark`.

---

## Lesson Authoring Workflow

### 1. Scaffold the Lesson Directory
Use the CLI scaffolding tool to generate the workspace and test directory trees:
```bash
uv run tforge generate arraysmith basic 04_matrix_algebra
```

### 2. Implement the Golden Reference (`solution.py`)
Write the reference implementation inside the test directory (`tests/<module>/<tier>/<lesson_name>/`). Ensure the baseline avoids unnecessary memory allocations and leverages vectorized C/Fortran/CUDA routines.

### 3. Create the Student Skeleton (`student_code.py`)
In the workspace directory (`<module>/<tier>/<lesson_name>/`), create `student_code.py`. Mirror the reference class and function signatures, replacing logic with:
```python
raise NotImplementedError("TODO: Implement this function")
```

### 4. Build the Verification Suite (`test_<lesson_id>.py`)
In the test directory, create `test_<lesson_id>.py` adhering to these conventions:
- For Python/NumPy tests, import and apply `@ast_policy` from `forge_core.ast_validator` to enforce vectorization.
- For PyTorch/CUDA tests (`tensorsmith` / `hpcsmith`), include GPU availability checks:
  ```python
  import pytest
  import torch

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
  class TestMyLesson:
      ...
  ```
- Invoke `compare_and_benchmark(student_fn=..., baseline_fn=...)` to evaluate correctness, speed, and memory usage.

---

## Writing Specifications (`INSTRUCTION.md`)

The `INSTRUCTION.md` file must be precise, professional, and unambiguous. Every instruction file must explicitly define input/output types, shapes, and constraints without providing code snippets or logic spoilers.

### Terminal Formatting Rules
When illustrating terminal CLI commands or expected outputs, never use raw ANSI escape sequences. Reference the standard color formatting library:
```python
from nguyenpanda.swan import c24, reset
```

---

## Enforcing AST Policies

To prevent students from bypassing vectorization requirements using brute-force Python loops, decorate test verification methods with `@ast_policy`:

```python
from forge_core.ast_validator import ast_policy

@ast_policy(forbid_loops=True, required_calls=["numpy.where"])
def test_exercise_correctness():
    ...
```

---

## Registering Context-Aware Hints

To provide incremental assistance when students invoke `show_hint()`, register the lesson in the hint configuration package:

1. Navigate to `hint/hint/<module>/<tier>/`.
2. Create `hint_<lesson_id>_<lesson_name>.py`.
3. Define the module path and hierarchical hint mapping:

```python
MODULE = "<tier>/<lesson_id>_<lesson_name>"

HINT = {
    "ClassName": {
        "method_name": {
            "major": "Use np.dot or the @ operator for matrix multiplication.",
            "minor": "Ensure inner matrix dimensions match: (M, K) @ (K, N) -> (M, N)."
        }
    }
}
```

---

## Dataset Management & Atomic Downloads

When authoring lessons that require external datasets or pre-computed benchmark matrices, use `DatasetManager` from `forge_core`. It guarantees atomic file downloads (via `.tmp` file renaming) and SHA-256 checksum verification, caching files to `./.tensorforge_datasets/`:

```python
from forge_core import DatasetManager

dm = DatasetManager()
dataset_path = dm.get_dataset(
    url="https://github.com/nguyenpanda/tensorforge/releases/download/v1.0/sample_data.bin",
    expected_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
)
```

---

## Verification & Testing Checklist

Before submitting changes, execute the automated verification suite to ensure structural and operational integrity:

```bash
# 1. Format code using Ruff
uv run ruff format .

# 2. Run static linting and structural analysis
uv run tforge lint

# 3. Verify infrastructure health and AST validators
uv run pytest tests/infra/ -v
uv run tforge validate

# 4. Execute full verification suite across all curricula
uv run tforge check all
```

All tests must pass cleanly without warnings or errors.

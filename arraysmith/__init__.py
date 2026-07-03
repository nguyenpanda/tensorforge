"""
arraysmith/__init__.py
======================
Root package for the TensorForge NumPy curriculum (Clean Lab Workspace).

Curriculum Structure — Tier-Scoped Numbering
---------------------------------------------
Lessons are organised into semantic tiers.  Each tier resets its lesson
counter to ``01``, eliminating the sequential-numbering ripple-effect that
makes flat curricula brittle to reorganisation.

    arraysmith/
    ├── basic/
    │   ├── 01_array_creation/       INSTRUCTION.md, student_code.py
    │   ├── 02_properties_reshaping/
    │   └── 03_indexing_slicing/
    ├── intermediate/
    │   ├── 01_vectorized_math/
    │   ├── 02_array_manipulation/
    │   └── 03_boolean_logic/
    ├── advanced/
    │   └── 01_memory_layout/
    └── applications/                (empty — future lessons)

Reference baselines and verification suites are securely isolated in
``tests/arraysmith/<tier>/<lesson_dir>/``.

Check your solutions using the CLI::

    tforge check arraysmith basic 01
    tforge check arraysmith intermediate 02
    tforge check arraysmith advanced 01
"""

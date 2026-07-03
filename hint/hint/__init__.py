"""
hint/__init__.py
================
Public API for the TensorForge hint system.

Registry Auto-Discovery (Two-Level Scan)
-----------------------------------------
At import time, :func:`_build_registry` performs a **two-level scan** of the
``hint/`` package tree:

Level 1 — curriculum sub-packages (e.g. ``hint.arraysmith``).
Level 2 — tier sub-packages within each curriculum (e.g. ``hint.arraysmith.basic``).

Within each tier package, every ``hint_*.py`` file is loaded.  Each such file
must expose:

- ``MODULE: str`` — composite key in the form ``"<tier>/<lesson_dir>"``
  (e.g. ``"basic/01_array_creation"``).
- ``HINT: dict``  — nested mapping ``{ClassName: {method_name: {major, minor}}}``.

The composite key format is required so that the look-up in
:func:`get_hint_str` can derive a unique, collision-free identifier from the
two-directory-deep path of the calling file inside the ``arraysmith/`` tree.

Adding a New Tier or Lesson
---------------------------
1. Create the sub-package: ``hint/hint/<curriculum>/<tier>/__init__.py``
2. Drop a ``hint_*.py`` file into that sub-package with a correct ``MODULE``
   composite key.

No changes to this file are required.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from nguyenpanda.swan import c24, reset

_GOLD = c24["ffd700"]
def yellow(s: str) -> str: return f"{_GOLD}{s}{reset}"


def _build_registry() -> dict:
    """Scan all curriculum and tier sub-packages of ``hint/`` and collect hint definitions.

    Performs a two-level traversal:

    1. Iterates over direct child packages of the ``hint`` namespace
       (e.g. ``hint.arraysmith``).
    2. Within each curriculum package, iterates over tier sub-packages
       (e.g. ``hint.arraysmith.basic``, ``hint.arraysmith.intermediate``).
    3. Within each tier package, loads every ``hint_*.py`` module.

    Each ``hint_*.py`` module must declare ``MODULE`` as a composite key
    ``"<tier>/<lesson_dir>"`` (e.g. ``"basic/01_array_creation"``) and
    ``HINT`` as the nested hint dictionary.

    Returns:
        dict: ``{composite_module_key: {ClassName: {method_name: {major, minor}}}}``.

    Raises:
        AttributeError: When a ``hint_*.py`` file is missing ``MODULE`` or ``HINT``.
    """
    registry: dict = {}

    import hint as _hint_root_pkg

    hint_root_path = Path(_hint_root_pkg.__file__).parent

    for _finder, curriculum_name, is_curriculum_pkg in pkgutil.iter_modules([str(hint_root_path)]):
        if not is_curriculum_pkg:
            continue

        full_curriculum = f"hint.{curriculum_name}"
        curriculum_pkg = importlib.import_module(full_curriculum)
        curriculum_path = Path(curriculum_pkg.__file__).parent

        for _tier_finder, tier_or_module_name, is_tier_pkg in pkgutil.iter_modules(
            [str(curriculum_path)]
        ):
            # Two-level: tier sub-packages (e.g. basic, intermediate, advanced).
            if is_tier_pkg:
                full_tier = f"{full_curriculum}.{tier_or_module_name}"
                tier_pkg = importlib.import_module(full_tier)
                tier_path = Path(tier_pkg.__file__).parent

                for _inner_finder, module_name, _inner_ispkg in pkgutil.iter_modules(
                    [str(tier_path)]
                ):
                    if not module_name.startswith("hint_"):
                        continue
                    full_name = f"{full_tier}.{module_name}"
                    _load_hint_module(full_name, registry)

            # Backward compatibility: flat ``hint_*.py`` directly under curriculum.
            elif tier_or_module_name.startswith("hint_"):
                full_name = f"{full_curriculum}.{tier_or_module_name}"
                _load_hint_module(full_name, registry)

    return registry


def _load_hint_module(full_module_name: str, registry: dict) -> None:
    """Import a single hint module and merge its entry into *registry*.

    Args:
        full_module_name: Fully-qualified Python module name (e.g.
            ``"hint.arraysmith.basic.hint_01_array_creation"``).
        registry: Mutable registry dict to update in-place.

    Raises:
        AttributeError: When the module is missing ``MODULE`` or ``HINT``.
    """
    mod = importlib.import_module(full_module_name)

    module_key: Optional[str] = getattr(mod, "MODULE", None)
    hint_data: Optional[dict] = getattr(mod, "HINT", None)

    if module_key is None:
        raise AttributeError(
            f"Hint file '{full_module_name}' is missing the required "
            f"'MODULE' constant. Add: MODULE = '<tier>/<lesson_directory_name>'"
        )
    if hint_data is None:
        raise AttributeError(
            f"Hint file '{full_module_name}' is missing the required "
            f"'HINT' dict. Add: HINT = {{...}}"
        )

    registry[module_key] = hint_data


HINTS_REGISTRY: dict = _build_registry()


def get_hint_str(deep: int = 1) -> Tuple[Dict[str, str], Path, str, str, str]:
    """Retrieve the hint dict for the calling context automatically.

    Inspects the call stack to determine the tier/lesson directory, class,
    and method of the caller, then looks up the composite key
    ``"<tier>/<lesson_dir>"`` in :data:`HINTS_REGISTRY`.

    The composite key is derived from the calling file's path by computing
    its two-directory-deep name relative to the ``arraysmith/`` root.  For a
    file at ``arraysmith/basic/01_array_creation/student_code.py`` the key
    is ``"basic/01_array_creation"``.

    Args:
        deep: Extra frames to skip above the default caller frame.  Pass ``1``
              (default) when calling directly from student code.

    Returns:
        Tuple of ``(hint_dict, caller_path, module_key, class_name, method_name)``.

    Raises:
        RuntimeError: When called outside a class method or when registry entries
                      are missing for the detected context.
        FileNotFoundError: When the calling file cannot be resolved.
    """
    stack = inspect.stack()

    if len(stack) < deep + 2:
        raise RuntimeError("Insufficient stack frame information to execute.")

    caller_frame_info = stack[deep + 1]
    frame = caller_frame_info.frame
    f_locals = frame.f_locals

    class_name: Optional[str] = None
    if "self" in f_locals:
        class_name = f_locals["self"].__class__.__name__
    elif "cls" in f_locals:
        class_name = f_locals["cls"].__name__

    if not class_name:
        raise RuntimeError(
            "show_hint() must be called from within a class method or instance method."
        )

    caller_path = Path(caller_frame_info.filename).resolve()
    if not caller_path.is_file():
        raise FileNotFoundError(f"Could not locate the calling file: {caller_path}")

    lesson_dir = caller_path.parent        # e.g. …/arraysmith/basic/01_array_creation
    tier_dir = lesson_dir.parent           # e.g. …/arraysmith/basic

    # Build the composite key: "<tier>/<lesson_dir_name>"
    module_key = f"{tier_dir.name}/{lesson_dir.name}"
    method_name = caller_frame_info.function

    if module_key not in HINTS_REGISTRY:
        # Fallback: try using only the immediate lesson directory name (legacy flat layout).
        fallback_key = lesson_dir.name
        if fallback_key in HINTS_REGISTRY:
            module_key = fallback_key
        else:
            raise RuntimeError(
                f"\n\nHint Error — DIRECTORY level:\n"
                f"  Composite key '{module_key}' (and legacy key '{fallback_key}') "
                f"have no entry in HINTS_REGISTRY.\n"
                f"  Fix: Create a 'hint_<name>.py' file in the appropriate\n"
                f"  hint tier sub-package with MODULE = '{module_key}' and a HINT dict."
            ) from None

    module_hints = HINTS_REGISTRY[module_key]

    if class_name not in module_hints:
        raise RuntimeError(
            f"\n\nHint Error — CLASS level:\n"
            f"  Class '{class_name}' not found in HINTS_REGISTRY['{module_key}'].\n"
            f"  Fix: Add '{class_name}' key to the HINT dict in the corresponding "
            f"hint file."
        ) from None

    class_hints = module_hints[class_name]

    if method_name not in class_hints:
        raise RuntimeError(
            f"\n\nHint Error — METHOD level:\n"
            f"  Method '{method_name}' not found in "
            f"HINTS_REGISTRY['{module_key}']['{class_name}'].\n"
            f"  Fix: Add '{method_name}' key with a dict containing 'major' and "
            f"'minor' strings."
        ) from None

    return class_hints[method_name], caller_path, module_key, class_name, method_name


def show_hint(deep: int = 1) -> None:
    """Print a formatted hint to stdout for the calling method.

    Automatically detects the caller's tier/lesson module, class, and method
    via :func:`get_hint_str`.  Students call this with no arguments from inside
    their exercise method.

    Args:
        deep: Additional stack frames to skip.  Leave at default ``1``.

    Example::

        class ArrayCreation:
            @classmethod
            def create_integer_range(cls):
                show_hint()  # prints the registered hint for this method
    """
    hint_data, file_path, module_key, class_name, method_name = get_hint_str(deep)

    sep = "=" * 70
    sub = "-" * 70

    print(f"\n{yellow(sep)}")
    print(yellow("💡  HINT"))
    print(yellow(sep))
    print(f"📂  Module  : {module_key}")
    print(f"📄  File    : {file_path}")
    print(f"📦  Class   : {class_name}")
    print(f"🔧  Method  : {method_name}()")
    print(yellow(sub))

    major_msg = hint_data.get("major", "No major hint provided.")
    minor_msg = hint_data.get("minor", "No minor hint provided.")

    print(yellow("🎯  Major Hint:"))
    for line in major_msg.splitlines():
        print(yellow(f"    {line}"))

    print(yellow("🔍  Minor Hint:"))
    for line in minor_msg.splitlines():
        print(yellow(f"    {line}"))

    print(yellow(sep) + "\n")
    raise NotImplementedError("Hint requested - Implementation pending")


__all__ = [
    "HINTS_REGISTRY",
    "get_hint_str",
    "show_hint",
]

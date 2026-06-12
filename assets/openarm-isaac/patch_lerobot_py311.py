#!/usr/bin/env python3
"""Make a pinned LeRobot source tree installable + importable on Python 3.11.

Why this exists
---------------
The NGC `isaac-lab` container ships Python 3.11.x (hard-locked to Isaac Sim's Kit
runtime). LeRobot @ d1b1c5c (v0.5.2) declares ``requires-python = ">=3.12"`` and uses
PEP-695 type-parameter syntax (``class X[T]``, ``def f[T]``, ``type X = ...``) in a
handful of modules. Both block us on 3.11:

  * the metadata gate makes ``pip install`` refuse outright, and
  * PEP-695 syntax is a *SyntaxError* on <3.12, which trips at import time AND during
    pip's install-time ``compileall`` pass over the whole package.

This script rewrites those constructs to their pre-695, semantics-identical
equivalents (module-level ``TypeVar`` + ``Generic[...]``; plain assignment for type
aliases) and relaxes the metadata gate to ``>=3.11``. No runtime behaviour changes.

It additionally lowers LeRobot's ``numpy``/``packaging`` *lower* bounds so the install
can be constrained to Isaac Sim's pinned versions (numpy<2, packaging<24) without an
empty-intersection ``ResolutionImpossible``. Without this, ``pip install lerobot[pi]``
drags numpy 1.x→2.x and packaging→25 into the Kit env and breaks Isaac's prebundle
torch (``import torch`` dies). The constraints file is applied at install time by the
userdata; see ``templates/openarm-isaac-userdata.sh.j2`` step 4. Upper bounds are kept.

Design notes
------------
* Line-oriented + exact full-line matching, so it is immune to substring relationships
  between an anchor and its replacement (e.g. ``from typing import Any`` is a substring
  of ``from typing import Any, TypeVar`` — a naive ``in`` check would re-apply it).
* Idempotent: re-running is a no-op (each edit is skipped if already applied).
* Fail-loud: if an anchor line is neither in its unpatched nor patched form, we raise —
  so a future LeRobot bump that moves these lines fails the deploy early (in this cheap
  step) instead of silently leaving PEP-695 syntax in place. Anchor lines are also
  required to be unique within their file.
* All five PEP-695 sites in the tree are patched, not just the two on our import chain,
  because pip compiles the entire package at install time.

Usage:  python patch_lerobot_py311.py /path/to/lerobot-src-root
"""

from __future__ import annotations

import sys
from pathlib import Path

# Each edit: (relative path, anchor_line, replacement_lines).
#   anchor_line       — the single unpatched source line to replace (exact, no newline).
#   replacement_lines — list of lines it becomes (length 1 = in-place swap; >1 = expand).
# "Already applied" is detected by the replacement block being present and the anchor
# absent — both compared as exact whole lines, so substring traps cannot occur.
EDITS: list[tuple[str, str, list[str]]] = [
    # ── metadata gate ────────────────────────────────────────────────────────────────
    (
        "pyproject.toml",
        'requires-python = ">=3.12"',
        ['requires-python = ">=3.11"'],
    ),
    # ── dependency-clobber guard: relax numpy/packaging *lower* bounds ─────────────────
    # The NGC isaac-lab Kit env pins numpy<2 and packaging<24 (isaaclab/-tasks/-rl
    # setup.py); its prebundle torch is built against that numpy 1.x ABI. LeRobot's
    # base pins (numpy>=2.0.0, packaging>=24.2) are hard *lower* bounds, so a plain
    # `pip install -c constraints.txt` pinning numpy<2 has an empty intersection and
    # fails with ResolutionImpossible. We lower the floors here so the install-time
    # constraints (numpy<2, packaging<24 — Isaac's installed versions) can hold.
    # Safe: pi05's only numpy call is np.digitize/np.linspace (1.x-stable), and lerobot
    # uses packaging only via .version.parse/Version (stable since packaging 14). The
    # numpy>=2.0.0 floor is a documented resolver hint ("helps the resolver converge"),
    # not an API requirement. Upper bounds (<2.3.0 / <26.0) are left intact.
    (
        "pyproject.toml",
        '    "numpy>=2.0.0,<2.3.0", # NOTE: Explicitly listing numpy helps the resolver converge faster. Upper bound imposed by opencv-python-headless.',
        ['    "numpy>=1.24.0,<2.3.0", # PATCHED(openarm-isaac): floor lowered 2.0.0->1.24.0 to coexist with Isaac Sim prebundle torch (numpy 1.x ABI). Upper bound imposed by opencv-python-headless.'],
    ),
    (
        "pyproject.toml",
        '    "packaging>=24.2,<26.0",',
        ['    "packaging>=21.3,<26.0", # PATCHED(openarm-isaac): floor lowered 24.2->21.3 to coexist with Isaac Lab packaging<24 pin.'],
    ),
    # ── site 1: processor/pipeline.py — generic class (TInput/TOutput already module-level TypeVars) ──
    (
        "src/lerobot/processor/pipeline.py",
        "from typing import Any, TypedDict, TypeVar, cast",
        ["from typing import Any, Generic, TypedDict, TypeVar, cast"],
    ),
    (
        "src/lerobot/processor/pipeline.py",
        "class DataProcessorPipeline[TInput, TOutput](HubMixin):",
        ["class DataProcessorPipeline(Generic[TInput, TOutput], HubMixin):"],
    ),
    # ── site 2: utils/io_utils.py — generic function (inject module-level T bound to JsonLike) ──
    (
        "src/lerobot/utils/io_utils.py",
        "from typing import Any",
        ["from typing import Any, TypeVar"],
    ),
    (
        "src/lerobot/utils/io_utils.py",
        "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:",
        [
            'T = TypeVar("T", bound=JsonLike)',
            "",
            "",
            "def deserialize_json_into_object(fpath: Path, obj: T) -> T:",
        ],
    ),
    # ── site 3: datasets/streaming_dataset.py — generic class (no typing import present) ──
    (
        "src/lerobot/datasets/streaming_dataset.py",
        "from pathlib import Path",
        ["from pathlib import Path", "from typing import Generic, TypeVar"],
    ),
    (
        "src/lerobot/datasets/streaming_dataset.py",
        "class Backtrackable[T]:",
        ['T = TypeVar("T")', "", "", "class Backtrackable(Generic[T]):"],
    ),
    # ── sites 4 & 5: motors/motors_bus.py — PEP-695 type aliases → plain assignment ──
    (
        "src/lerobot/motors/motors_bus.py",
        "type NameOrID = str | int",
        ["NameOrID = str | int"],
    ),
    (
        "src/lerobot/motors/motors_bus.py",
        "type Value = int | float",
        ["Value = int | float"],
    ),
]

# Files that must be free of PEP-695 syntax after patching (verified by AST below).
PEP695_FILES = [
    "src/lerobot/processor/pipeline.py",
    "src/lerobot/utils/io_utils.py",
    "src/lerobot/datasets/streaming_dataset.py",
    "src/lerobot/motors/motors_bus.py",
]


def _block_present(lines: list[str], block: list[str]) -> bool:
    """True if `block` appears as a run of consecutive exact lines in `lines`."""
    if not block:
        return True
    for i in range(len(lines) - len(block) + 1):
        if lines[i : i + len(block)] == block:
            return True
    return False


def apply_edits(root: Path) -> int:
    """Apply EDITS idempotently via exact full-line matching. Returns count applied."""
    applied = 0
    for rel, anchor, replacement in EDITS:
        fp = root / rel
        if not fp.exists():
            raise SystemExit(f"[patch] FATAL: expected file missing: {rel}")
        lines = fp.read_text().split("\n")
        n_anchor = lines.count(anchor)
        # Check "already applied" FIRST: some edits keep the anchor as the first line of
        # their replacement block (e.g. inserting an import after `from pathlib import
        # Path`), so the anchor still matches post-patch — the replacement block being
        # present is the reliable idempotency signal.
        if _block_present(lines, replacement):
            print(f"[patch] skip (already applied): {rel}  ({anchor[:48]}...)")
        elif n_anchor == 1:
            idx = lines.index(anchor)
            lines[idx : idx + 1] = replacement
            fp.write_text("\n".join(lines))
            print(f"[patch] applied: {rel}  ({anchor[:58]}...)")
            applied += 1
        elif n_anchor > 1:
            raise SystemExit(
                f"[patch] FATAL: anchor not unique in {rel} "
                f"(found {n_anchor}× of {anchor!r}); refusing ambiguous edit"
            )
        else:
            raise SystemExit(
                f"[patch] FATAL: anchor absent and patch not applied in {rel}.\n"
                f"        anchor={anchor!r}\n        Source layout changed — review patcher."
            )
    return applied


def verify_no_pep695(root: Path) -> None:
    """Parse each patched file and assert no PEP-695 AST nodes survive.

    PEP-695 introduces ast.TypeAlias (``type X = ...``) and the ``type_params``
    attribute on ClassDef/FunctionDef/AsyncFunctionDef (``class X[T]`` / ``def f[T]``).

    This patcher's verify step runs in-container on Python 3.11 (Isaac Sim's Kit
    runtime), where ``ast.TypeAlias`` does not exist — so we resolve it via getattr and
    skip that node-type check when absent. On 3.11, *parsing* a still-695 file would
    itself raise SyntaxError, so the check fails loud either way; the type_params sweep
    additionally covers any 3.12+ host running this for local verification.
    """
    import ast

    type_alias_cls = getattr(ast, "TypeAlias", None)  # 3.12+ only
    for rel in PEP695_FILES:
        tree = ast.parse((root / rel).read_text())
        offenders: list[str] = []
        for node in ast.walk(tree):
            if type_alias_cls is not None and isinstance(node, type_alias_cls):
                offenders.append(f"TypeAlias @ line {node.lineno}")
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and getattr(node, "type_params", None):
                offenders.append(f"{node.name} type_params @ line {node.lineno}")
        if offenders:
            raise SystemExit(
                f"[patch] FATAL: PEP-695 syntax still present in {rel}: {offenders}"
            )
        print(f"[patch] verified PEP-695-free: {rel}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_lerobot_py311.py <lerobot-src-root>")
    root = Path(sys.argv[1]).resolve()
    if not (root / "pyproject.toml").exists():
        raise SystemExit(f"[patch] FATAL: {root} does not look like a lerobot checkout")
    n = apply_edits(root)
    verify_no_pep695(root)
    print(f"[patch] OK — {n} edit(s) applied, all patched files PEP-695-free.")


if __name__ == "__main__":
    main()

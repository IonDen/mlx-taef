"""Guard: the banned future-annotations import must not appear.

ruff `FA` cannot enforce this on the py311 floor (it only *adds* the import for
older targets), so this offline test is the enforcement mechanism.
"""

from pathlib import Path

# Split so this file does not self-trigger the check.
BANNED = "from __future__" + " import annotations"


def test_no_future_annotations_import() -> None:
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for d in ("src", "tests", "scripts"):
        for p in (root / d).rglob("*.py"):
            if p.name == "_version.py":  # vcs-generated, exempt
                continue
            if BANNED in p.read_text(encoding="utf-8"):
                offenders.append(str(p.relative_to(root)))
    assert not offenders, f"banned `{BANNED}` in: {offenders}"

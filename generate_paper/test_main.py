# Copyright (c) 2026- Paschalis Bizopoulos
"""Check the generated manuscript against its expected SHA-256 hash."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


_EXPECTED_MS_PDF_SHA256 = (
    "d16183a513fef03691f51c57c6d38c9d276490431c74b21b62a3cf51490fae43"
)


def test_generated_manuscript_matches_the_reviewed_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate the manuscript and compare its bytes with the pinned baseline."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from generate_paper import main as subject  # noqa: PLC0415

    subject.main()
    actual = hashlib.sha256((subject.RUNTIME_DIR / "ms.pdf").read_bytes()).hexdigest()
    if actual != _EXPECTED_MS_PDF_SHA256:
        message = f"ms.pdf SHA-256: expected {_EXPECTED_MS_PDF_SHA256}, got {actual}"
        raise AssertionError(message)

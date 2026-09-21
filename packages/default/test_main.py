# Copyright (c) 2026- Paschalis Bizopoulos
"""Specify the offline EEG classification and manuscript workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_smoke_run_generates_classification_results_and_a_complete_manuscript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train all five module variants and compile their results into a PDF."""
    monkeypatch.chdir(tmp_path)
    from packages.default import main as subject  # noqa: PLC0415

    subject.main()
    results = (tmp_path / "tmp/results.tex").read_text()
    if "alexnet" not in results or "nan" in results.lower():
        message = "classification results must contain finite AlexNet accuracies"
        raise AssertionError(message)
    for name in ("eyes-open", "eyes-closed", "healthy-area", "tumor-area", "epilepsy"):
        for prefix in ("signal", "signal-as-image", "spectrogram", "cnn"):
            image = tmp_path / "tmp" / f"{prefix}-{name}.png"
            if not image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                message = "each EEG class must have all four visualizations"
                raise AssertionError(message)
    manuscript = (tmp_path / "tmp/ms.pdf").read_bytes()
    if not manuscript.startswith(b"%PDF-") or not manuscript.rstrip().endswith(
        b"%%EOF",
    ):
        message = "the manuscript must be a complete PDF"
        raise AssertionError(message)

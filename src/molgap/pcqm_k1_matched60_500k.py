"""Prospective matched60-v4 K1 reference for PairToken scale attribution."""
from __future__ import annotations

from pathlib import Path

from .pcqm_k1_pair_token_500k import run as run_matched60


MODE = "neural_atom_k1"
PARAMETERS = 3_658_817


def run(
    output: Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    resume: Path | None = None,
    preflight_only: bool = False,
) -> dict:
    """Train only the missing K1 reference under the exact PairToken contract."""
    return run_matched60(
        output,
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha256,
        resume=resume,
        preflight_only=preflight_only,
        mode=MODE,
        expected_parameters=PARAMETERS,
        reference_mae_eV=None,
    )

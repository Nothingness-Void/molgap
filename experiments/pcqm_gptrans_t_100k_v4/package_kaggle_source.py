"""Build the private Kaggle2 GPTrans-T source and frozen-state dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.pcqm_gptrans_v4 import (
    EXPECTED_INITIAL_MODEL_SHA256,
    EXPECTED_INITIAL_STATE_ARTIFACT_SHA256,
    _make_model,
    _state_sha256,
)
from molgap.training_reproducibility import configure_fp32_determinism, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "experiments/pcqm_gptrans_t_100k_v4"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("GPTrans packaging requires committed source and experiment files")
    output.mkdir(parents=True)
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT)
    sources = sorted(
        (REPO_ROOT / item.decode("utf-8") for item in raw.split(b"\0") if item),
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    sources = [path for path in sources if path.is_file() and "__pycache__" not in path.parts]
    archive = output / "source.tar.gz"
    inventory = []
    with tarfile.open(archive, "w:gz") as handle:
        for source in sources:
            relative = source.relative_to(REPO_ROOT).as_posix()
            handle.add(source, arcname=relative)
            inventory.append({"path": relative, "sha256": sha256_file(source)})
    archive_sha = sha256_file(archive)
    configure_fp32_determinism(42)
    model = _make_model()
    state_sha = _state_sha256(model)
    if state_sha != EXPECTED_INITIAL_MODEL_SHA256:
        raise RuntimeError(f"Frozen initial tensor state changed: {state_sha}")
    state_path = output / "initial_state.pt"
    torch.save(
        {"format": "molgap-gptrans-t-seed42-initial-state-v1", "model_state": model.state_dict(), "state_sha256": state_sha},
        state_path,
    )
    if sha256_file(state_path) != EXPECTED_INITIAL_STATE_ARTIFACT_SHA256:
        raise RuntimeError("Frozen initial-state serialization changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(archive_sha + "\n", encoding="utf-8")
    (output / "SOURCE_FILES.json").write_text(json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8")
    (output / "dataset-metadata.json").write_text(json.dumps({"title": "MolGap GPTrans T V4 Source", "id": "kaseichou/molgap-gptrans-t-v4-source", "licenses": [{"name": "other"}], "isPrivate": True}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

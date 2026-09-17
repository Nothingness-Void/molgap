"""Build the immutable Kaggle3 source bundle for the pair-normalization screen."""
from __future__ import annotations

import argparse
import json
import shutil
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


EXPERIMENT = "experiments/pcqm_gptrans_pair_norm_100k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", EXPERIMENT],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Packaging requires committed source and experiment files")

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
    shutil.copyfile(archive, output / "source_payload.bin")

    initial_state = args.initial_state.resolve()
    if sha256_file(initial_state) != EXPECTED_INITIAL_STATE_ARTIFACT_SHA256:
        raise RuntimeError("Frozen initial-state artifact identity changed")
    payload = torch.load(initial_state, map_location="cpu", weights_only=False)
    configure_fp32_determinism(42)
    model = _make_model()
    model.load_state_dict(payload["model_state"], strict=True)
    state_sha = _state_sha256(model)
    if state_sha != EXPECTED_INITIAL_MODEL_SHA256:
        raise RuntimeError(f"Frozen initial tensor state changed: {state_sha}")
    state_path = output / "initial_state.pt"
    shutil.copyfile(initial_state, state_path)
    if sha256_file(state_path) != EXPECTED_INITIAL_STATE_ARTIFACT_SHA256:
        raise RuntimeError("Frozen initial-state serialization changed")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(archive_sha + "\n", encoding="utf-8")
    (output / "SOURCE_FILES.json").write_text(
        json.dumps(
            {
                "format": "molgap-v4-source-inventory-v1",
                "source_commit": commit,
                "files": inventory,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MolGap GPTrans Pair Norm V5 Source",
                "id": "nvoid912/molgap-gptrans-pair-norm-v5-source-v2",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

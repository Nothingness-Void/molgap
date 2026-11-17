"""Kaggle2: evaluate one frozen validation winner on QM9 test exactly once."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working/pure2d_r3_test")
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
EXPECTED_SOURCE_COMMIT = "b56205967f12f517c7eea4428c0dfe8571c54996"
EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"
EXPECTED_RWSE_SHA256 = (
    "09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5"
)
VALIDATION_REFERENCE = {
    "average": 0.11006919294595718,
    "Gap": 0.1318935602903366,
}
TEST_REFERENCE = {
    "average": 0.11177898943424225,
    "Gap": 0.1340952217578888,
}
ALLOWED_CANDIDATES = {
    "edge_state_structural_gps",
    "edge_state_structural_orbital",
    "pair_gps_2d_r3_orbital",
    "pair_gps_2d_r3_triplet",
    "pair_gps_2d_r3_combined",
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/qm9_screen.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archive = find_one("src.zip")
    extracted = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, extracted)
    modules = list(extracted.rglob("molgap/qm9_screen.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def ensure_pascal_compatible_torch() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not allocate a GPU")
    if torch.cuda.get_device_capability(0) != (6, 0):
        return
    if "sm_60" in set(torch.cuda.get_arch_list()):
        return
    if os.environ.get(PASCAL_COMPAT_RESTART) == "1":
        raise RuntimeError("Compatibility install still lacks sm_60")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            "torch==2.7.1",
            "nvidia-cusparselt-cu12==0.6.3",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ]
    )
    os.environ[PASCAL_COMPAT_RESTART] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def install_dependencies() -> None:
    import torch

    version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "").replace(".", "")
    wheel_version = "2.7.0" if version.startswith("2.7.") else version
    index = f"https://data.pyg.org/whl/torch-{wheel_version}+cu{cuda}.html"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch-geometric==2.6.1",
            "torch_cluster",
            "torch_scatter",
            "-f",
            index,
        ]
    )


def locate_frozen_model(selected_model: str) -> Path:
    relative = Path(selected_model).relative_to("/kaggle/working")
    suffix = relative.as_posix()
    matches = [
        path
        for path in Path("/kaggle/input").rglob("model.pt")
        if path.as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected frozen model suffix {suffix}: {matches}")
    return matches[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        ensure_pascal_compatible_torch()
        install_dependencies()
        python_root = source_python_root()
        sys.path.insert(0, str(python_root))
        from molgap.qm9_screen import evaluate_structural_checkpoint_test
        from molgap.structural_encoding import sha256

        source_commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
        if source_commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError(f"Unexpected source commit: {source_commit}")
        selection = json.loads(find_one("selection.json").read_text())
        if selection.get("source_commit") != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("Validation selection used a different source commit")
        if selection.get("test_role_read") is not False:
            raise RuntimeError("Validation kernel did not keep QM9 test sealed")
        candidate = selection.get("selected_candidate")
        if candidate not in ALLOWED_CANDIDATES:
            raise RuntimeError(f"No permitted validation winner: {candidate}")
        selected_rows = [
            row for row in selection["completed"] if row["candidate"] == candidate
        ]
        if len(selected_rows) != 1:
            raise RuntimeError(f"Expected one selected row: {selected_rows}")
        selected = selected_rows[0]
        validation = selected["validation"]
        validation_pass = (
            selected.get("eligible") is True
            and selected["parameter_count"] <= 4_800_000
            and validation["average"]["mae"] < VALIDATION_REFERENCE["average"]
            and validation["Gap"]["mae"] < VALIDATION_REFERENCE["Gap"]
        )
        if not validation_pass:
            raise RuntimeError("Selected model does not pass the frozen validation gate")

        mounted_model = locate_frozen_model(selection["selected_model"])
        frozen_model = OUT / "frozen_validation_winner.pt"
        temporary_model = frozen_model.with_suffix(".tmp")
        shutil.copy2(mounted_model, temporary_model)
        os.replace(temporary_model, frozen_model)
        frozen_sha256 = sha256(frozen_model)
        result = evaluate_structural_checkpoint_test(
            candidate=candidate,
            checkpoint=frozen_model,
            output=OUT / "metrics.json",
            payload_output=OUT / "payload.pt",
            train_size=30_000,
            validation_size=3_000,
            test_size=3_000,
            split_seed=42,
            cache_dir=find_one("acceptance.json").parents[2],
        )
        if result["split_fingerprint"] != EXPECTED_SPLIT_FINGERPRINT:
            raise RuntimeError("QM9 test split fingerprint changed")
        if result["rwse_output_sha256"] != EXPECTED_RWSE_SHA256:
            raise RuntimeError("QM9 test RWSE cache identity changed")
        metrics = result["metrics"]
        passed = (
            metrics["average"]["mae"] < TEST_REFERENCE["average"]
            and metrics["Gap"]["mae"] < TEST_REFERENCE["Gap"]
        )
        summary = {
            "experiment": "pure2d_r3_qm9_frozen_test",
            "source_commit": source_commit,
            "candidate": candidate,
            "parameter_count": selected["parameter_count"],
            "validation": validation,
            "validation_gate_passed": validation_pass,
            "frozen_model_sha256": frozen_sha256,
            "test_role_read_once": True,
            "test": metrics,
            "test_reference": TEST_REFERENCE,
            "test_gate_passed": passed,
        }
        atomic_json(OUT / "test_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()

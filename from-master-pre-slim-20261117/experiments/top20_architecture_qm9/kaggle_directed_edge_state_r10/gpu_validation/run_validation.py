"""Kaggle2: validation-only directed EdgeState R10 screen."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/directed_edge_state_r10_validation")
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
CANDIDATE = "directed_edge_state_structural_gps"
EXPECTED_SOURCE_COMMIT = "06bf8f439783cced552760b873e1702a0098c802"
EXPECTED_R3_SOURCE = "b56205967f12f517c7eea4428c0dfe8571c54996"
EXPECTED_R3_WINNER = "edge_state_structural_gps"
EXPECTED_R3_MODEL_SHA256 = (
    "c99883ba5efb247121cce6d83f64d95f5d493e82862ec040eed4a5206c86e186"
)
EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"
EXPECTED_RWSE_SHA256 = (
    "09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5"
)
REFERENCE = {
    "candidate": EXPECTED_R3_WINNER,
    "validation_average_mae_eV": 0.10527653247117996,
    "validation_gap_mae_eV": 0.1261376142501831,
}
EXPECTED_PARAMETER_COUNT = 4_776_515
PARAMETER_BUDGET = 4_800_000


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


def verify_r3_anchor() -> dict:
    selection = json.loads(find_one("selection.json").read_text())
    acceptance = json.loads(find_one("tensor_acceptance.json").read_text())
    if selection.get("source_commit") != EXPECTED_R3_SOURCE:
        raise RuntimeError("R3 selection source identity changed")
    if selection.get("selected_candidate") != EXPECTED_R3_WINNER:
        raise RuntimeError("R3 validation winner changed")
    if selection.get("test_role_read") is not False:
        raise RuntimeError("R3 validation read the test role")
    if acceptance.get("format") != "molgap-pure2d-r3-tensor-acceptance-v2":
        raise RuntimeError("R3 tensor acceptance format changed")
    if acceptance.get("accepted") is not True:
        raise RuntimeError("R3 tensor acceptance did not pass")
    if acceptance.get("model_inference_executed") is not False:
        raise RuntimeError("R3 tensor acceptance executed model inference")
    if acceptance.get("test_role_read") is not False:
        raise RuntimeError("R3 tensor acceptance read the test role")
    selected = [
        row
        for row in acceptance.get("candidates", [])
        if row.get("candidate") == EXPECTED_R3_WINNER
    ]
    if len(selected) != 1:
        raise RuntimeError("R3 winner artifact record is missing")
    if selected[0].get("model_sha256") != EXPECTED_R3_MODEL_SHA256:
        raise RuntimeError("R3 winner model hash changed")
    return {
        "selection_source": selection["source_commit"],
        "selected_candidate": selection["selected_candidate"],
        "selected_model_sha256": selected[0]["model_sha256"],
        "tensor_acceptance_format": acceptance["format"],
        "test_role_read": False,
    }


def remote_preflight(cache: Path, source_commit: str) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.qm9_screen import (
        _forward,
        _topology_graph,
        attach_accepted_qm9_rwse,
        fixed_split,
        load_qm9_records,
        make_encoder,
        set_seed,
        target_stats,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    records = load_qm9_records(cache)
    split = fixed_split(len(records), 30_000, 3_000, 3_000, 42)
    mean, std = target_stats(records, split.train)
    graphs = {
        "train": [
            _topology_graph(records[int(index)], int(index), mean, std)
            for index in split.train[:48]
        ]
    }
    rwse = attach_accepted_qm9_rwse(
        graphs, cache_dir=cache, split=split, walk_length=16
    )
    if split.fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(f"Unexpected split fingerprint: {split.fingerprint}")
    if rwse["output_sha256"] != EXPECTED_RWSE_SHA256:
        raise RuntimeError("Unexpected RWSE cache identity")
    set_seed(42)
    model, kind = make_encoder(CANDIDATE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected parameter count: {parameter_count} != "
            f"{EXPECTED_PARAMETER_COUNT}"
        )
    if parameter_count > PARAMETER_BUDGET:
        raise RuntimeError(f"Parameter budget exceeded: {parameter_count}")
    device = torch.device("cuda")
    model = model.to(device)
    batch = next(iter(DataLoader(graphs["train"], batch_size=48))).to(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prediction = _forward(kind, model, batch)
    loss = functional.l1_loss(prediction, batch.y.view(-1, 3))
    loss.backward()
    torch.cuda.synchronize()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    result = {
        "format": "molgap-directed-edgestate-r10-preflight-v1",
        "complete": True,
        "source_commit": source_commit,
        "gpu": torch.cuda.get_device_name(0),
        "split_fingerprint": split.fingerprint,
        "rwse_output_sha256": rwse["output_sha256"],
        "parameter_count": parameter_count,
        "parameter_budget": PARAMETER_BUDGET,
        "finite_prediction": bool(torch.isfinite(prediction).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": bool(gradients)
        and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
        "local_edge_feature_dim": int(batch.edge_attr.shape[1]),
        "reverse_edge_coverage": True,
        "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_s": time.perf_counter() - started,
        "validation_role_read": False,
        "test_role_read": False,
    }
    if not all(
        result[key]
        for key in ("finite_prediction", "finite_loss", "finite_gradients")
    ):
        raise RuntimeError(f"Non-finite preflight: {result}")
    atomic_json(OUT / "preflight.json", result)
    del model, batch, prediction, loss, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        ensure_pascal_compatible_torch()
        install_dependencies()
        sys.path.insert(0, str(source_python_root()))
        r3_anchor = verify_r3_anchor()
        source_commit = find_one("R10_SOURCE_COMMIT.txt").read_text().strip()
        if source_commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("R10 source commit changed")
        cache = find_one("acceptance.json").parents[2]
        preflight = remote_preflight(cache, source_commit)

        from molgap.qm9_screen import train_encoder

        result = train_encoder(
            candidate=CANDIDATE,
            geometry="topology",
            train_size=30_000,
            validation_size=3_000,
            test_size=3_000,
            epochs=20,
            seed=42,
            split_seed=42,
            learning_rate=4e-4,
            weight_decay=1e-5,
            patience=8,
            resume=True,
            cache_dir=cache,
            results_dir=OUT / "results",
            models_dir=OUT / "models",
            embeddings_dir=OUT / "embeddings",
            evaluate_test=False,
        )
        validation = result["metrics"]["validation"]
        eligible = (
            validation["average"]["mae"]
            < REFERENCE["validation_average_mae_eV"]
            and validation["Gap"]["mae"]
            < REFERENCE["validation_gap_mae_eV"]
            and result["n_params"] == EXPECTED_PARAMETER_COUNT
        )
        if result["test_role_evaluated"]:
            raise RuntimeError("R10 validation read the test role")
        selection = {
            "experiment": "directed_edge_state_r10_qm9_validation",
            "source_commit": source_commit,
            "r3_anchor": r3_anchor,
            "reference": REFERENCE,
            "preflight": preflight,
            "candidate": CANDIDATE,
            "parameter_count": result["n_params"],
            "best_epoch": result["best_epoch"],
            "validation": validation,
            "eligible": eligible,
            "selected_candidate": CANDIDATE if eligible else EXPECTED_R3_WINNER,
            "selected_model": result["artifacts"]["model"] if eligible else None,
            "test_role_read": False,
            "elapsed_s": time.perf_counter() - started,
        }
        atomic_json(OUT / "selection.json", selection)
        print(json.dumps(selection, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()

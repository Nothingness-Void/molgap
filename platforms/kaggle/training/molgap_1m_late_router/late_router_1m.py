"""Fit a validation-only late blend for the original 1M base and dual heads."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

if importlib.util.find_spec("rdkit") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "rdkit==2025.3.6",
    ])


def ensure_pascal_compatible_torch() -> None:
    import torch as probe_torch

    if not probe_torch.cuda.is_available():
        return
    capability = probe_torch.cuda.get_device_capability(0)
    if capability != (6, 0) or "sm_60" in set(probe_torch.cuda.get_arch_list()):
        return
    if os.environ.get("MOLGAP_TORCH_COMPAT_RESTART") == "1":
        raise RuntimeError("cu126 compatibility install still lacks sm_60")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "--no-deps", "--force-reinstall", "torch==2.7.1",
        "nvidia-cusparselt-cu12==0.6.3",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    ])
    os.environ["MOLGAP_TORCH_COMPAT_RESTART"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


ensure_pascal_compatible_torch()

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
TAG = "original1m_late_router_seed42"
SEEDS = (42, 43, 44)
PASS_THRESHOLD = 0.001


def find_one(name: str) -> Path:
    paths = sorted(INPUT.rglob(name))
    if not paths:
        raise FileNotFoundError(name)
    return paths[0]


runtime = find_one("late_router.py").parent
sys.path.insert(0, str(runtime))
from fusion import FusionHead  # noqa: E402
from late_router import (  # noqa: E402
    apply_binned_alpha,
    binned_alpha,
    blend,
    build_router_features,
    grid_alpha,
    metric_block,
    optimal_alpha,
    scaffold_partition,
)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_state(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.inference_mode()
def predict(model, left, right, indices, device):
    output = []
    for start in range(0, len(indices), 16384):
        idx = indices[start:start + 16384]
        output.append(model(left[idx].to(device), right[idx].to(device)).float().cpu())
    return torch.cat(output).numpy()


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid:{smiles}"
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return value or Chem.MolToSmiles(mol, canonical=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)
    gps7 = torch.load(find_one("gps_expansion_1m_embeddings.pt"), map_location="cpu", weights_only=False)
    gps9 = torch.load(find_one("gps_expansion_1m_depth9_embeddings.pt"), map_location="cpu", weights_only=False)
    geo = torch.load(find_one("schnet_expansion_1m_embeddings.pt"), map_location="cpu", weights_only=False)
    source = geo["source_idx"].long()
    position = torch.searchsorted(gps7["source_idx"].long(), source)
    if not torch.equal(gps7["source_idx"].long()[position], source):
        raise RuntimeError("GPS7/SchNet source alignment failed")
    if not torch.equal(gps7["source_idx"], gps9["source_idx"]):
        raise RuntimeError("GPS7/GPS9 source alignment failed")
    h7 = gps7["embeddings"][position].float()
    h9 = gps9["embeddings"][position].float()
    h3 = geo["embeddings"].float()
    table = pd.read_csv(find_one("phase8_expansion_1m.csv"), usecols=["smiles", "homo", "lumo", "gap"])
    labels = table[["homo", "lumo", "gap"]].to_numpy(np.float32)[source.numpy()]
    permutation = np.random.RandomState(42).permutation(len(source))
    n_train, n_val = int(0.8 * len(source)), int(0.1 * len(source))
    val_idx = permutation[n_train:n_train + n_val]
    test_idx = permutation[n_train + n_val:]
    split_hash = hashlib.sha256(permutation.astype(np.int64).tobytes()).hexdigest()

    base = load_state(
        FusionHead("gate", 192, 0.0, dim_2d=192, dim_3d=192).to(device),
        find_one("routed_1m_dualgps_seed42_base_best.pt"), device,
    )
    expert = load_state(
        FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192).to(device),
        find_one("gate_2gps_expansion_1m_n997445_best.pt"), device,
    )
    all_idx = np.concatenate([val_idx, test_idx])
    base_prediction = predict(base, h7, h3, all_idx, device)
    expert_prediction = predict(expert, torch.cat([h7, h9], dim=1), h3, all_idx, device)
    features = build_router_features(
        base_prediction, expert_prediction,
        h7[all_idx].numpy(), h9[all_idx].numpy(), h3[all_idx].numpy(),
    )
    n_validation = len(val_idx)
    y_validation = labels[val_idx]
    base_validation, expert_validation = base_prediction[:n_validation], expert_prediction[:n_validation]
    scaffolds = [scaffold(value) for value in table["smiles"].iloc[source[val_idx].numpy()].tolist()]
    fit_idx, selection_idx = scaffold_partition(scaffolds)
    atomic_json({
        "stage": "predictions_complete", "split_sha256": split_hash,
        "n_aligned": len(source), "validation": len(val_idx), "test_locked": len(test_idx),
        "router_fit": len(fit_idx), "router_selection": len(selection_idx),
    }, OUTPUT / f"{TAG}_progress.json")

    candidates = {}
    global_alpha = grid_alpha(y_validation[fit_idx], base_validation[fit_idx], expert_validation[fit_idx])
    candidates["fixed_alpha"] = blend(base_validation[selection_idx], expert_validation[selection_idx], global_alpha)
    bin_alpha = binned_alpha(y_validation[fit_idx], base_validation[fit_idx], expert_validation[fit_idx])
    candidates["gap_bin_alpha"] = apply_binned_alpha(base_validation[selection_idx], expert_validation[selection_idx], bin_alpha)
    alpha_target = optimal_alpha(y_validation[fit_idx], base_validation[fit_idx], expert_validation[fit_idx])
    seed_models, seed_predictions = [], []
    for seed in SEEDS:
        models, columns = [], []
        for column in range(3):
            model = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=0.1,
                early_stopping=True, validation_fraction=0.1, random_state=seed,
            )
            model.fit(features[fit_idx], alpha_target[:, column])
            models.append(model)
            columns.append(np.clip(model.predict(features[selection_idx]), 0.0, 1.0))
        seed_models.append(models)
        alpha = np.stack(columns, axis=1)
        seed_predictions.append(blend(base_validation[selection_idx], expert_validation[selection_idx], alpha))
        joblib.dump(models, OUTPUT / f"{TAG}_hgb_seed{seed}.joblib")
        atomic_json({"stage": "hgb_seed_complete", "seed": seed}, OUTPUT / f"{TAG}_progress.json")
    candidates["hgb_alpha_3seed"] = np.mean(seed_predictions, axis=0)

    baseline_selection = metric_block(y_validation[selection_idx], expert_validation[selection_idx])
    selection_metrics = {name: metric_block(y_validation[selection_idx], value) for name, value in candidates.items()}
    eligible = [
        name for name, metrics in selection_metrics.items()
        if metrics["average"]["mae_eV"] <= baseline_selection["average"]["mae_eV"] - PASS_THRESHOLD
        and metrics["Gap"]["mae_eV"] <= baseline_selection["Gap"]["mae_eV"] - PASS_THRESHOLD
    ]
    selected = min(eligible, key=lambda name: selection_metrics[name]["average"]["mae_eV"]) if eligible else None
    result = {
        "tag": TAG, "split_sha256": split_hash, "selection_baseline": baseline_selection,
        "selection_candidates": selection_metrics, "pass_threshold_eV": PASS_THRESHOLD,
        "selected": selected, "test_opened": bool(selected),
    }
    if selected:
        offset = n_validation
        y_test = labels[test_idx]
        base_test, expert_test = base_prediction[offset:], expert_prediction[offset:]
        if selected == "fixed_alpha":
            chosen = blend(base_test, expert_test, global_alpha)
        elif selected == "gap_bin_alpha":
            chosen = apply_binned_alpha(base_test, expert_test, bin_alpha)
        else:
            per_seed = []
            for models in seed_models:
                alpha = np.stack([np.clip(model.predict(features[offset:]), 0.0, 1.0) for model in models], axis=1)
                per_seed.append(blend(base_test, expert_test, alpha))
            chosen = np.mean(per_seed, axis=0)
        result["test_baseline"] = metric_block(y_test, expert_test)
        result["test_selected"] = metric_block(y_test, chosen)
        result["test_delta"] = {
            key: result["test_selected"][key]["mae_eV"] - result["test_baseline"][key]["mae_eV"]
            for key in ("HOMO", "LUMO", "Gap", "average")
        }
        np.savez_compressed(
            OUTPUT / f"{TAG}_test_predictions.npz", source_idx=source[test_idx].numpy(),
            target=y_test, base=base_test, expert=expert_test, selected=chosen,
        )
    atomic_json(result, OUTPUT / f"{TAG}_metrics.json")
    atomic_json({"complete": True, "selected": selected, "outputs": sorted(path.name for path in OUTPUT.glob(f"{TAG}*"))}, OUTPUT / f"{TAG}_manifest.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

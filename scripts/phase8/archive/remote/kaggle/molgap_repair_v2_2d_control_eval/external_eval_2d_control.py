"""Evaluate the controlled v1/v2 dual-GPS heads on fixed external labels only."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


SEED = 42
TARGETS = ("homo", "lumo", "gap")


def pip_install(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=True)


def install_runtime() -> None:
    # Kaggle can assign a P100, which needs a CUDA build retaining sm_60 support.
    pip_install("--upgrade", "--force-reinstall", "torch==2.7.1+cu126", "--index-url", "https://download.pytorch.org/whl/cu126")
    pip_install("--upgrade", "torch_geometric")
    pip_install("pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "-f", "https://data.pyg.org/whl/torch-2.7.0+cu126.html")
    # Keep the compiled RDKit extension compatible with NumPy after Torch install.
    pip_install("--upgrade", "--force-reinstall", "numpy==1.26.4", "rdkit==2024.9.6")
    from rdkit import Chem
    if Chem.MolFromSmiles("CC") is None:
        raise RuntimeError("RDKit smoke test failed after runtime installation")


def atomic_write_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(table, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def find_input(required: set[str]) -> Path:
    for root, _, names in os.walk("/kaggle/input"):
        if required.issubset(names):
            return Path(root)
    raise FileNotFoundError(f"Could not find Kaggle input with: {sorted(required)}")


def find_runtime() -> Path:
    for root, dirs, _ in os.walk("/kaggle/input"):
        candidate = Path(root) / "molgap"
        if "molgap" in dirs and (candidate / "gps.py").is_file():
            return Path(root)
    for root, _, names in os.walk("/kaggle/input"):
        if "molgap_runtime_source_v2.zip" in names:
            destination = Path("/kaggle/working/runtime_source")
            with zipfile.ZipFile(Path(root) / "molgap_runtime_source_v2.zip") as archive:
                archive.extractall(destination)
            if (destination / "molgap" / "gps.py").is_file():
                return destination
    raise FileNotFoundError("MolGap runtime source is not mounted")


def load_model(model, path: Path, torch, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def metric_block(y_true, prediction, np, r2_score, targets=TARGETS) -> dict:
    result = {}
    for target_index, target in enumerate(targets):
        error = np.abs(prediction[:, target_index] - y_true[:, target_index])
        result[target] = {"mae_eV": float(error.mean()), "r2": float(r2_score(y_true[:, target_index], prediction[:, target_index]))}
    result["average"] = {"mae_eV": float(np.abs(prediction - y_true).mean()), "r2": float(np.mean([result[target]["r2"] for target in targets]))}
    return result


def bootstrap_delta(y_true, baseline, candidate, np, targets=TARGETS, draws: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    result = {}
    for target_index, target in enumerate((*targets, "average")):
        if target == "average":
            delta = np.abs(candidate - y_true).mean(axis=1) - np.abs(baseline - y_true).mean(axis=1)
        else:
            delta = np.abs(candidate[:, target_index] - y_true[:, target_index]) - np.abs(baseline[:, target_index] - y_true[:, target_index])
        means = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            means[draw] = delta[rng.integers(0, len(delta), len(delta))].mean()
        result[target] = {"mae_delta_eV": float(delta.mean()), "ci95_eV": [float(value) for value in np.quantile(means, [0.025, 0.975])], "p_v2_better": float((means < 0).mean())}
    return result


def predict_models(table, gps7_models, gps9_models, heads, smiles_to_2d_pyg, GeometricDataLoader, TensorDataset, TorchDataLoader, torch, device):
    graphs, kept = [], []
    for row_index, row in table.iterrows():
        graph = smiles_to_2d_pyg(row.smiles)
        if graph is not None:
            graphs.append(graph)
            kept.append(row_index)
    table = table.iloc[kept].reset_index(drop=True)
    if not len(table):
        raise RuntimeError("No valid 2D graphs were constructed")
    encoded = [[] for _ in (*gps7_models, *gps9_models)]
    with torch.no_grad():
        for batch in GeometricDataLoader(graphs, batch_size=256, shuffle=False):
            batch = batch.to(device)
            with torch.autocast("cuda"):
                for output, model in zip(encoded, (*gps7_models, *gps9_models)):
                    output.append(model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch).float().cpu())
    h_v1_7, h_v2_7, h_v1_9, h_v2_9 = (torch.cat(parts) for parts in encoded)
    predictions = []
    with torch.no_grad():
        for head, h7, h9 in zip(heads, (h_v1_7, h_v2_7), (h_v1_9, h_v2_9)):
            chunks = []
            for batch7, batch9 in TorchDataLoader(TensorDataset(h7, h9), batch_size=4096, shuffle=False):
                chunks.append(head(batch7.to(device), batch9.to(device)).float().cpu())
            predictions.append(torch.cat(chunks).numpy())
    return table, predictions[0], predictions[1]


def main() -> None:
    install_runtime()
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import r2_score
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import TensorDataset
    from torch_geometric.loader import DataLoader as GeometricDataLoader

    sys.path.insert(0, str(find_runtime()))
    from molgap.fusion import DualGPSFusionHead
    from molgap.gps import GPSWrapper
    from molgap.graphs import smiles_to_2d_pyg

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not attach a GPU")
    if "sm_60" not in torch.cuda.get_arch_list():
        raise RuntimeError("P100-compatible Torch was not installed")
    device = torch.device("cuda")
    work = Path("/kaggle/working")
    atomic_write_json({"status": "runtime_ready"}, work / "repair_v2_2d_progress.json")
    common_root = find_input({"external_common_labels.csv"})
    pcqm_root = find_input({"pcqm4mv2_valid_5k.csv"})
    model_root = find_input({"phase8_gps_expansion_1m.pt", "phase8_gps_expansion_1m_depth9.pt", "phase8_repair_v2_1m_gps7_control.pt", "phase8_repair_v2_1m_gps9_control.pt", "phase8_dual_gps_2d_v1_1m.pt", "phase8_dual_gps_2d_v2_1m.pt"})
    gps_args = dict(hidden_channels=192, num_heads=4, dropout=0.05)
    v1_gps7 = load_model(GPSWrapper(num_layers=7, **gps_args), model_root / "phase8_gps_expansion_1m.pt", torch, device)
    v1_gps9 = load_model(GPSWrapper(num_layers=9, **gps_args), model_root / "phase8_gps_expansion_1m_depth9.pt", torch, device)
    v2_gps7 = load_model(GPSWrapper(num_layers=7, **gps_args), model_root / "phase8_repair_v2_1m_gps7_control.pt", torch, device)
    v2_gps9 = load_model(GPSWrapper(num_layers=9, **gps_args), model_root / "phase8_repair_v2_1m_gps9_control.pt", torch, device)
    v1_head = load_model(DualGPSFusionHead(hidden=192), model_root / "phase8_dual_gps_2d_v1_1m.pt", torch, device)
    v2_head = load_model(DualGPSFusionHead(hidden=192), model_root / "phase8_dual_gps_2d_v2_1m.pt", torch, device)

    common = pd.read_csv(common_root / "external_common_labels.csv")
    common, v1_common, v2_common = predict_models(common, (v1_gps7, v2_gps7), (v1_gps9, v2_gps9), (v1_head, v2_head), smiles_to_2d_pyg, GeometricDataLoader, TensorDataset, TorchDataLoader, torch, device)
    y_common = common.loc[:, TARGETS].to_numpy(dtype=np.float64)
    common_metrics = {"n_valid": int(len(common)), "v1": metric_block(y_common, v1_common, np, r2_score), "v2": metric_block(y_common, v2_common, np, r2_score), "v2_minus_v1": {}}
    for scope in ("all", "ood1000", "p8_targeted_hard"):
        mask = np.ones(len(common), dtype=bool) if scope == "all" else common.eval_set.eq(scope).to_numpy()
        common_metrics["v2_minus_v1"][scope] = {"n": int(mask.sum()), **bootstrap_delta(y_common[mask], v1_common[mask], v2_common[mask], np)}
    common_output = common.loc[:, ["eval_set", "cid", "smiles", *TARGETS]].copy()
    for index, target in enumerate(TARGETS):
        common_output[f"v1_{target}"] = v1_common[:, index]
        common_output[f"v2_{target}"] = v2_common[:, index]
        common_output[f"abs_error_delta_{target}"] = np.abs(v2_common[:, index] - y_common[:, index]) - np.abs(v1_common[:, index] - y_common[:, index])
    atomic_write_json(common_metrics, work / "repair_v2_2d_common_metrics.json")
    atomic_write_csv(common_output, work / "repair_v2_2d_common_predictions.csv")
    atomic_write_json({"status": "common_complete", "n_common": int(len(common))}, work / "repair_v2_2d_progress.json")

    pcqm = pd.read_csv(pcqm_root / "pcqm4mv2_valid_5k.csv")
    pcqm, v1_pcqm, v2_pcqm = predict_models(pcqm, (v1_gps7, v2_gps7), (v1_gps9, v2_gps9), (v1_head, v2_head), smiles_to_2d_pyg, GeometricDataLoader, TensorDataset, TorchDataLoader, torch, device)
    y_pcqm = pcqm.loc[:, ["gap_true"]].to_numpy(dtype=np.float64)
    pcqm_metrics = {"n_valid": int(len(pcqm)), "v1_gap": metric_block(y_pcqm, v1_pcqm[:, 2:3], np, r2_score, targets=("gap",)), "v2_gap": metric_block(y_pcqm, v2_pcqm[:, 2:3], np, r2_score, targets=("gap",)), "v2_minus_v1_gap": bootstrap_delta(y_pcqm, v1_pcqm[:, 2:3], v2_pcqm[:, 2:3], np, targets=("gap",))}
    pcqm_output = pcqm.loc[:, ["idx", "smiles", "gap_true"]].copy()
    pcqm_output["v1_gap"] = v1_pcqm[:, 2]
    pcqm_output["v2_gap"] = v2_pcqm[:, 2]
    pcqm_output["abs_error_delta_gap"] = np.abs(v2_pcqm[:, 2] - y_pcqm[:, 0]) - np.abs(v1_pcqm[:, 2] - y_pcqm[:, 0])
    atomic_write_json(pcqm_metrics, work / "repair_v2_2d_pcqm_metrics.json")
    atomic_write_csv(pcqm_output, work / "repair_v2_2d_pcqm_predictions.csv")
    atomic_write_json({"status": "complete", "n_common": int(len(common)), "n_pcqm": int(len(pcqm))}, work / "repair_v2_2d_progress.json")
    print(json.dumps({"common": common_metrics, "pcqm": pcqm_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()

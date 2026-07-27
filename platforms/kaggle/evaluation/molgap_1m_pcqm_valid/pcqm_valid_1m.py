"""Compare the 1M candidate and routed-v4 on PCQM4Mv2 public validation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RUN_MODE = "full"
SEED = 42
TARGETS = ("gap",)


def pip_install(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=True)


def install_runtime() -> None:
    # Kaggle currently assigns a P100 (sm_60); its stock CUDA 12.8 torch cannot run there.
    pip_install(
        "--upgrade", "--force-reinstall", "torch==2.7.1+cu126",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    )
    pip_install("--upgrade", "torch_geometric")
    pip_install(
        "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
        "-f", "https://data.pyg.org/whl/torch-2.7.0+cu126.html",
    )
    # The Torch reinstall can upgrade NumPy beyond the ABI supported by Kaggle's
    # preinstalled RDKit. Pin the pair before importing MolGap graph utilities.
    pip_install("--upgrade", "--force-reinstall", "numpy==1.26.4", "rdkit==2024.9.6")
    from rdkit import Chem

    if Chem.MolFromSmiles("CC") is None:
        raise RuntimeError("RDKit smoke test failed after runtime installation")


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
    raise FileNotFoundError("MolGap runtime source is not mounted")


def load_model(model, path: Path, torch, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def metric_block(y_true, prediction, np, r2_score):
    result = {}
    for target_index, target in enumerate(TARGETS):
        error = np.abs(prediction[:, target_index] - y_true[:, target_index])
        result[target] = {
            "mae_eV": float(error.mean()),
            "r2": float(r2_score(y_true[:, target_index], prediction[:, target_index])),
        }
    result["average"] = {
        "mae_eV": float(np.abs(prediction - y_true).mean()),
        "r2": float(np.mean([result[target]["r2"] for target in TARGETS])),
    }
    return result


def bootstrap_delta(y_true, baseline, candidate, np, draws: int = 10000):
    rng = np.random.default_rng(SEED)
    result = {}
    for target_index, target in enumerate((*TARGETS, "average")):
        if target == "average":
            delta = np.abs(candidate - y_true).mean(axis=1) - np.abs(baseline - y_true).mean(axis=1)
        else:
            delta = np.abs(candidate[:, target_index] - y_true[:, target_index]) - np.abs(baseline[:, target_index] - y_true[:, target_index])
        means = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            means[draw] = delta[rng.integers(0, len(delta), len(delta))].mean()
        result[target] = {
            "mae_delta_eV": float(delta.mean()),
            "ci95_eV": [float(value) for value in np.quantile(means, [0.025, 0.975])],
            "p_candidate_better": float((means < 0).mean()),
        }
    return result


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
    from molgap.fusion import FusionHead
    from molgap.gps import GPSWrapper
    from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg
    from molgap.schnet import SchNetWrapper

    assert torch.cuda.is_available(), "Kaggle did not attach a GPU"
    assert "sm_60" in torch.cuda.get_arch_list(), "P100-compatible torch was not installed"
    device = torch.device("cuda")
    labels_root = find_input({"pcqm4mv2_valid_5k.csv"})
    model_root = find_input({"phase8_gps_expansion_1m.pt", "gate_2gps_expansion_1m_n997445_best.pt"})
    df = pd.read_csv(labels_root / "pcqm4mv2_valid_5k.csv")
    # Reuse the shared evaluator while reporting only its physically relevant Gap field.
    df["cid"] = df["idx"]
    df["eval_set"] = "pcqm4mv2_valid"
    df["gap"] = df["gap_true"]
    if RUN_MODE == "preflight":
        df = df.groupby("eval_set", group_keys=False).head(2).reset_index(drop=True)

    graphs_2d, graphs_3d, kept = [], [], []
    for index, row in df.iterrows():
        graph_2d = smiles_to_2d_pyg(row.smiles)
        graph_3d = smiles_to_pyg(row.smiles, random_seed=SEED + int(index))
        if graph_2d is not None and graph_3d is not None:
            graphs_2d.append(graph_2d)
            graphs_3d.append(graph_3d)
            kept.append(index)
    df = df.iloc[kept].reset_index(drop=True)
    if len(df) != len(kept):
        raise RuntimeError("Invalid external graph alignment")

    common_gps = dict(hidden_channels=192, num_heads=4, dropout=0.05)
    schnet_args = dict(hidden_channels=192, num_filters=192, num_interactions=6, num_gaussians=50, cutoff=6.0, dropout=0.0, use_charges=True)
    candidate_gps7 = load_model(GPSWrapper(num_layers=7, **common_gps), model_root / "phase8_gps_expansion_1m.pt", torch, device)
    candidate_gps9 = load_model(GPSWrapper(num_layers=9, **common_gps), model_root / "phase8_gps_expansion_1m_depth9.pt", torch, device)
    candidate_schnet = load_model(SchNetWrapper(**schnet_args), model_root / "extend_1m_n997445_best.pt", torch, device)
    candidate_fusion = load_model(FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192), model_root / "gate_2gps_expansion_1m_n997445_best.pt", torch, device)
    base_gps7 = load_model(GPSWrapper(num_layers=7, **common_gps), model_root / "phase8_gps_expansion_500k.pt", torch, device)
    base_gps9 = load_model(GPSWrapper(num_layers=9, **common_gps), model_root / "phase8_gps_expansion_500k_depth9.pt", torch, device)
    base_schnet = load_model(SchNetWrapper(**schnet_args), model_root / "phase8_schnet_expansion_500k.pt", torch, device)
    base_fusion = load_model(FusionHead("gate", 192, 0.0, dim_2d=192, dim_3d=192), model_root / "phase8_hybrid_fusion_expansion_500k.pt", torch, device)
    base_dual_fusion = load_model(FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192), model_root / "phase8_hybrid_fusion_expansion_500k_dualgps.pt", torch, device)

    def encode_2d(models):
        outputs = [[] for _ in models]
        with torch.no_grad():
            for batch in GeometricDataLoader(graphs_2d, batch_size=192, shuffle=False):
                batch = batch.to(device)
                with torch.autocast("cuda"):
                    for output, model in zip(outputs, models):
                        output.append(model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch).float().cpu())
        return [torch.cat(output) for output in outputs]

    def encode_3d(models):
        outputs = [[] for _ in models]
        with torch.no_grad():
            for batch in GeometricDataLoader(graphs_3d, batch_size=96, shuffle=False):
                batch = batch.to(device)
                with torch.autocast("cuda"):
                    for output, model in zip(outputs, models):
                        output.append(model.encode(batch.z, batch.pos, batch.batch, charges=batch.charges).float().cpu())
        return [torch.cat(output) for output in outputs]

    candidate_h7, candidate_h9, base_h7, base_h9 = encode_2d([candidate_gps7, candidate_gps9, base_gps7, base_gps9])
    candidate_h3, base_h3 = encode_3d([candidate_schnet, base_schnet])

    def predict(fusion, h2, h3):
        chunks = []
        with torch.no_grad():
            for h2_batch, h3_batch in TorchDataLoader(TensorDataset(h2, h3), batch_size=2048, shuffle=False):
                chunks.append(fusion(h2_batch.to(device), h3_batch.to(device)).float().cpu())
        return torch.cat(chunks).numpy()

    def predict_head(model, embedding):
        """Evaluate each encoder's trained readout without rerunning graph work."""
        chunks = []
        with torch.no_grad():
            for (embedding_batch,) in TorchDataLoader(TensorDataset(embedding), batch_size=2048, shuffle=False):
                chunks.append(model.head(embedding_batch.to(device)).float().cpu())
        return torch.cat(chunks).numpy()

    candidate = predict(candidate_fusion, torch.cat([candidate_h7, candidate_h9], dim=1), candidate_h3)
    base_single = predict(base_fusion, base_h7, base_h3)
    base_dual = predict(base_dual_fusion, torch.cat([base_h7, base_h9], dim=1), base_h3)
    component_predictions = {
        "base_gps7": predict_head(base_gps7, base_h7),
        "candidate_gps7": predict_head(candidate_gps7, candidate_h7),
        "base_gps9": predict_head(base_gps9, base_h9),
        "candidate_gps9": predict_head(candidate_gps9, candidate_h9),
        "base_schnet": predict_head(base_schnet, base_h3),
        "candidate_schnet": predict_head(candidate_schnet, candidate_h3),
        "base_fusion_single": base_single,
        "base_fusion_dualgps": base_dual,
        "candidate_fusion_dualgps": candidate,
    }
    route = base_single[:, 2] < 4.0
    baseline = base_single.copy()
    baseline[route] = base_dual[route]
    y_true = df.loc[:, ["gap"]].to_numpy(dtype=np.float64)
    baseline_gap = baseline[:, 2:3]
    candidate_gap = candidate[:, 2:3]

    metrics = {"run_mode": RUN_MODE, "n_valid": int(len(df)), "route_n": int(route.sum()), "baseline_routed_v4": metric_block(y_true, baseline_gap, np, r2_score), "candidate_1m": metric_block(y_true, candidate_gap, np, r2_score)}
    metrics["components_gap"] = {
        name: metric_block(y_true, prediction[:, 2:3], np, r2_score)
        for name, prediction in component_predictions.items()
    }
    metrics["continuation_minus_base_gap"] = {
        "gps7": bootstrap_delta(y_true, component_predictions["base_gps7"][:, 2:3], component_predictions["candidate_gps7"][:, 2:3], np),
        "gps9": bootstrap_delta(y_true, component_predictions["base_gps9"][:, 2:3], component_predictions["candidate_gps9"][:, 2:3], np),
        "schnet": bootstrap_delta(y_true, component_predictions["base_schnet"][:, 2:3], component_predictions["candidate_schnet"][:, 2:3], np),
    }
    blocks = {}
    for scope in ("all",):
        mask = np.ones(len(df), dtype=bool) if scope == "all" else df.eval_set.eq(scope).to_numpy()
        blocks[scope] = {"n": int(mask.sum()), **bootstrap_delta(y_true[mask], baseline_gap[mask], candidate_gap[mask], np)}
    metrics["candidate_minus_routed_v4"] = blocks
    output = df.loc[:, ["eval_set", "cid", "smiles", *TARGETS]].copy()
    for index, target in enumerate(TARGETS):
        output[f"routed_v4_{target}"] = baseline_gap[:, index]
        output[f"candidate_1m_{target}"] = candidate_gap[:, index]
        output[f"abs_error_delta_{target}"] = np.abs(candidate_gap[:, index] - y_true[:, index]) - np.abs(baseline_gap[:, index] - y_true[:, index])
    for name, prediction in component_predictions.items():
        output[f"{name}_gap"] = prediction[:, 2]
    Path("/kaggle/working/pcqm4mv2_valid_5k_metrics.json").write_text(json.dumps(metrics, indent=2))
    output.to_csv("/kaggle/working/pcqm4mv2_valid_5k_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

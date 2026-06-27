"""Unified model loading and inference pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .constants import (
    MODEL_PHASE6, GRAPHS_PHASE6, PARAMS_PHASE6, TARGET_COLS, SEED,
    MODEL_REGISTRY,
)
from .fusion import FusionHead
from .gps import GPSWrapper
from .graphs import (
    smiles_to_pyg, smiles_list_to_pyg, smiles_to_pyg_ensemble, smiles_to_2d_pyg,
)
from .schnet import SchNetWrapper
from .tensornet import TensorNetWrapper
from .utils import create_split_indices


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def load_normalization_stats(
    graphs_path: str | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Load y_mean and y_std from a graph cache's training split."""
    graphs_path = graphs_path or GRAPHS_PHASE6
    graphs = torch.load(graphs_path, weights_only=False)
    train_idx, _, _ = create_split_indices(len(graphs), random_state=seed)
    train_y = np.stack([graphs[i].y.squeeze(0).numpy() for i in train_idx])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    del graphs
    return y_mean, y_std


def load_model(
    model_path: str | None = None,
    params: dict | None = None,
    graphs_path: str | None = None,
    *,
    key: str | None = None,
    normalized: bool | None = None,
    use_charges: bool = True,
    n_desc: int = 0,
    device: torch.device | str | None = None,
) -> tuple[SchNetWrapper | TensorNetWrapper, np.ndarray, np.ndarray, torch.device]:
    """Load a 3D encoder model with its normalization stats.

    With no arguments, defaults to the Phase 6 best model (normalized).
    Pass ``key=`` a registry name ("phase6_schnet" / "phase7_schnet_300k" /
    "tensornet_300k") to load that model with the right checkpoint, params, and
    normalization. For the GPS 2D or hybrid models use ``load_hybrid`` instead.

    For non-normalized models (P7 raw eV), y_mean/y_std are 0/1 so the same
    ``predict_graphs`` denorm step is a no-op.
    """
    kind = "schnet"
    if key is not None:
        spec = MODEL_REGISTRY[key]
        kind = spec["kind"]
        if kind not in ("schnet", "tensornet"):
            raise ValueError(
                f"load_model is for 3D encoder keys; '{key}' is kind='{kind}'. "
                "Use load_hybrid() for the 2D/hybrid models."
            )
        model_path = spec["checkpoint"]
        params = spec["params"]
        graphs_path = spec.get("graphs", graphs_path)
        use_charges = spec.get("use_charges", use_charges)
        if normalized is None:
            normalized = spec["normalized"]

    model_path = model_path or MODEL_PHASE6
    params = params or PARAMS_PHASE6
    graphs_path = graphs_path or GRAPHS_PHASE6
    if normalized is None:
        normalized = True

    device = _resolve_device(device)

    if normalized:
        y_mean, y_std = load_normalization_stats(graphs_path)
    else:
        y_mean = np.zeros(len(TARGET_COLS), dtype=np.float32)
        y_std = np.ones(len(TARGET_COLS), dtype=np.float32)

    if kind == "tensornet":
        model = TensorNetWrapper(**params, use_charges=use_charges).to(device)
    else:
        model = SchNetWrapper(**params, use_charges=use_charges, n_desc=n_desc).to(device)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    model.eval()
    return model, y_mean, y_std, device


def load_hybrid(
    device: torch.device | str | None = None,
    *,
    key: str = "phase8_replacement_hybrid",
) -> tuple[GPSWrapper, SchNetWrapper | TensorNetWrapper, FusionHead, torch.device]:
    """Load a hybrid trio: (gps_2d, encoder_3d, fusion_head, device).

    Default is the selected Phase 8 v2 base, ``"phase8_replacement_hybrid"``.
    Pass ``key="phase7_hybrid"`` to load the frozen v1 fallback/control.

    All are raw-eV (no normalization). The fusion head's architecture
    (fusion_type, hidden) is read from its Optuna metrics file so it always
    matches the saved checkpoint. Encoders expose ``encode()`` for the
    embeddings the fusion head consumes.
    """
    import json

    device = _resolve_device(device)
    hspec = MODEL_REGISTRY[key]
    comp_2d_key, comp_3d_key = hspec["components"]
    gspec = MODEL_REGISTRY[comp_2d_key]
    tspec = MODEL_REGISTRY[comp_3d_key]

    gps = GPSWrapper(**gspec["params"]).to(device)
    gps.load_state_dict(
        torch.load(gspec["checkpoint"], weights_only=True, map_location=device)
    )
    gps.eval()

    if tspec["kind"] == "tensornet":
        encoder_3d = TensorNetWrapper(**tspec["params"], use_charges=tspec.get("use_charges", False)).to(device)
    else:
        encoder_3d = SchNetWrapper(**tspec["params"], use_charges=tspec.get("use_charges", True)).to(device)
    encoder_3d.load_state_dict(
        torch.load(tspec["checkpoint"], weights_only=True, map_location=device)
    )
    encoder_3d.eval()

    with open(hspec["metrics"]) as f:
        metrics = json.load(f)
    bp = metrics.get(
        "best_params",
        {
            "fusion_type": hspec.get("fusion_type", "gate"),
            "hidden": hspec.get("hidden", 192),
            "dropout": hspec.get("dropout", 0.0),
        },
    )
    fusion = FusionHead(
        bp["fusion_type"],
        bp["hidden"],
        bp.get("dropout", hspec.get("dropout", 0.0)),
    ).to(device)
    fusion.load_state_dict(
        torch.load(hspec["checkpoint"], weights_only=True, map_location=device)
    )
    fusion.eval()

    return gps, encoder_3d, fusion, device


def predict_smiles_batch_hybrid(
    smiles_list: list[str],
    models: tuple | None = None,
    *,
    bs_2d: int = 256,
    bs_3d: int = 128,
    return_embeddings: bool = False,
    device: torch.device | str | None = None,
    hybrid_key: str = "phase8_replacement_hybrid",
):
    """Batch-predict B3LYP HOMO/LUMO/Gap with the hybrid model (raw eV).

    Builds both 2D and 3D graphs, keeps only molecules where BOTH succeed (3D
    ETKDG can fail), encodes each with its frozen encoder, and fuses. Returns
    ``(valid_idx, preds)`` — preds[i] aligns with smiles_list[valid_idx[i]]. With
    ``return_embeddings=True`` also returns the ``emb_2d, emb_3d`` arrays
    (the features the Δ model will consume), so one forward pass yields both the
    B3LYP baseline and the Δ features.

    Pass ``models=(gps, encoder_3d, fusion, device)`` from ``load_hybrid`` to
    reuse a loaded trio across calls.
    """
    from torch_geometric.loader import DataLoader

    if models is None:
        gps, encoder_3d, fusion, device = load_hybrid(device, key=hybrid_key)
    else:
        gps, encoder_3d, fusion, device = models

    g2d_list, g3d_list, valid_idx = [], [], []
    for i, smi in enumerate(smiles_list):
        g3d = smiles_to_pyg(smi)
        if g3d is None:
            continue
        g2d = smiles_to_2d_pyg(smi)
        if g2d is None:
            continue
        g3d_list.append(g3d)
        g2d_list.append(g2d)
        valid_idx.append(i)

    if not valid_idx:
        emb_dim_3d = 192
        empty = np.empty((0, len(TARGET_COLS)), dtype=np.float32)
        return (np.array([], dtype=int), empty) + (
            (np.empty((0, 192)), np.empty((0, emb_dim_3d))) if return_embeddings else ()
        )

    emb_2d = []
    with torch.no_grad():
        for b in DataLoader(g2d_list, batch_size=bs_2d):
            b = b.to(device)
            emb_2d.append(gps.encode(b.x, b.edge_index, b.edge_attr, b.batch).cpu())
    emb_2d = torch.cat(emb_2d)

    emb_3d = []
    with torch.no_grad():
        for b in DataLoader(g3d_list, batch_size=bs_3d):
            b = b.to(device)
            charges = b.charges if hasattr(b, "charges") else None
            emb_3d.append(encoder_3d.encode(b.z, b.pos, b.batch, charges=charges).cpu())
    emb_3d = torch.cat(emb_3d)

    with torch.no_grad():
        preds = fusion(emb_2d.to(device), emb_3d.to(device)).cpu().numpy()

    valid_idx = np.array(valid_idx, dtype=int)
    if return_embeddings:
        return valid_idx, preds, emb_2d.numpy(), emb_3d.numpy()
    return valid_idx, preds


def predict_graphs(
    model: SchNetWrapper,
    pyg_list: list,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Run inference on a list of PyG graphs. Returns denormalized predictions."""
    from torch_geometric.loader import DataLoader

    loader = DataLoader(pyg_list, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            desc = batch.desc if hasattr(batch, "desc") else None
            out = model(batch.z, batch.pos, batch.batch, charges=charges, desc=desc)
            preds.append(out.cpu().numpy() * y_std + y_mean)
    return np.vstack(preds)


def predict_smiles(
    smiles: str,
    model: SchNetWrapper | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict[str, float] | None:
    """Predict HOMO/LUMO/Gap for a single SMILES. Returns dict or None on failure."""
    if model is None:
        model, y_mean, y_std, device = load_model()

    data = smiles_to_pyg(smiles)
    if data is None:
        return None

    preds = predict_graphs(model, [data], y_mean, y_std, device)
    return {t: float(preds[0, i]) for i, t in enumerate(TARGET_COLS)}


def predict_smiles_batch(
    smiles_list: list[str],
    model: SchNetWrapper | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    device: torch.device | None = None,
    *,
    batch_size: int = 64,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Predict HOMO/LUMO/Gap for a list of SMILES. Returns DataFrame with results."""
    if model is None:
        model, y_mean, y_std, device = load_model()

    graphs, valid_idx = smiles_list_to_pyg(
        smiles_list, show_progress=show_progress,
    )

    if not graphs:
        return pd.DataFrame(columns=["smiles", "homo", "lumo", "gap", "success"])

    preds = predict_graphs(model, graphs, y_mean, y_std, device, batch_size=batch_size)

    rows = []
    pred_i = 0
    for i, smi in enumerate(smiles_list):
        if i in valid_idx:
            rows.append({
                "smiles": smi,
                "homo": float(preds[pred_i, 0]),
                "lumo": float(preds[pred_i, 1]),
                "gap": float(preds[pred_i, 2]),
                "success": True,
            })
            pred_i += 1
        else:
            rows.append({
                "smiles": smi,
                "homo": np.nan,
                "lumo": np.nan,
                "gap": np.nan,
                "success": False,
            })
    return pd.DataFrame(rows)


def predict_smiles_ensemble(
    smiles: str,
    k: int = 8,
    model: SchNetWrapper | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict[str, float] | None:
    """Predict with k conformers and average. Returns dict with mean and std."""
    if model is None:
        model, y_mean, y_std, device = load_model()

    graphs = smiles_to_pyg_ensemble(smiles, k=k)
    if not graphs:
        return None

    preds = predict_graphs(model, graphs, y_mean, y_std, device)
    result = {}
    for i, t in enumerate(TARGET_COLS):
        result[t] = float(preds[:, i].mean())
        result[f"{t}_std"] = float(preds[:, i].std())
    return result


# ── M1: GW prediction with uncertainty (Δ-ensemble + calibration + OOD) ──
# The frozen-embedding Δ-learning path (Phase 9/10), wrapped so a single SMILES
# yields a GW-level (value, σ, ood_flag) instead of a bare number. Artifacts come
# from scripts/phase10: the 10 LightGBM members per target, the σ-recalibration
# scales, and the OOD reference bundle (standardized fit embeddings + threshold).


def load_uq_bundle(device: torch.device | str | None = None) -> dict:
    """Load everything predict_smiles_with_uq needs, once, for reuse across calls.

    Returns a dict with: the SchNet hybrid trio (for B3LYP + 384-d features —
    MUST be the SAME hybrid that produced the Δ-model's training embeddings,
    i.e. phase7_hybrid, 192+192-d), the per-target LightGBM members, the
    calibration scales, and the OOD reference arrays.
    """
    import json
    import lightgbm as lgb

    from .constants import RESULTS_DIR

    phase10 = RESULTS_DIR / "phase10"
    # SchNet hybrid: the Δ embeddings (delta_oe62_embeddings.npz) are 192+192-d,
    # produced by phase7_hybrid — not the 128-d tensornet hybrid. Train/inference
    # feature consistency requires the same encoder here.
    hybrid = load_hybrid(device, key="phase7_hybrid")

    members = {}
    for t in TARGET_COLS:
        m, k = [], 0
        while (phase10 / "ensemble_lgbm" / f"{t}_m{k}.txt").exists():
            s = (phase10 / "ensemble_lgbm" / f"{t}_m{k}.txt").read_text(encoding="utf-8")
            m.append(lgb.Booster(model_str=s))
            k += 1
        if not m:
            raise FileNotFoundError(
                f"No ensemble boosters for '{t}' in {phase10/'ensemble_lgbm'}. "
                "Run scripts/phase10/train_ensemble.py first."
            )
        members[t] = m

    calib = json.loads((phase10 / "ensemble_calibration.json").read_text())
    ood = np.load(phase10 / "ood_reference.npz")
    return {
        "hybrid": hybrid, "members": members, "calib": calib,
        "ref_std": ood["ref_std"], "ood_mu": ood["mu"], "ood_sd": ood["sd"],
        "ood_threshold": float(ood["threshold"][0]), "ood_k": int(ood["k"][0]),
    }


def predict_smiles_with_uq(
    smiles: str,
    bundle: dict | None = None,
    device: torch.device | str | None = None,
) -> dict | None:
    """Predict GW-level HOMO/LUMO/Gap for one SMILES, with uncertainty.

    Returns, per target, the calibrated GW value, its 1σ uncertainty (eV), plus a
    single molecule-level ``ood`` flag and the embedding-distance that triggered
    it. ``None`` if the 3D conformer or 2D graph could not be built.

    Pipeline: SchNet-hybrid B3LYP + 384-d embedding → 10 LightGBM members predict
    Δ → GW = B3LYP + mean(Δ); σ = std(Δ) × calibration_scale → OOD by k-NN
    distance to the training embeddings (euclidean, standardized).

    Result shape::

        {"homo": {"value": .., "sigma": .., "b3lyp": ..},   # GW eV, 1σ eV, raw B3LYP
         "lumo": {...}, "gap": {...},
         "ood": bool, "ood_distance": float, "ood_threshold": float}

    Pass ``bundle=load_uq_bundle()`` to reuse loaded artifacts across many calls.
    """
    from sklearn.neighbors import NearestNeighbors

    if bundle is None:
        bundle = load_uq_bundle(device)

    # B3LYP prediction + the 384-d fusion embedding (one forward pass).
    vi, preds, e2d, e3d = predict_smiles_batch_hybrid(
        [smiles], models=bundle["hybrid"], return_embeddings=True,
    )
    if len(vi) == 0:
        return None
    b3lyp = preds[0]                       # [homo, lumo, gap] in eV
    feat = np.hstack([e2d[0], e3d[0]]).astype(np.float32)[None, :]  # [1, 384]

    out: dict = {}
    for i, t in enumerate(TARGET_COLS):
        P = np.array([mb.predict(feat)[0] for mb in bundle["members"][t]])  # member Δs
        delta_mu, delta_sd = float(P.mean()), float(P.std())
        sigma = delta_sd * bundle["calib"][t]["scale"]   # calibrated 1σ
        out[t] = {
            "value": float(b3lyp[i] + delta_mu),         # GW-level prediction
            "sigma": float(sigma),
            "b3lyp": float(b3lyp[i]),
        }

    # OOD: standardized k-NN distance to the training embeddings.
    fz = (feat - bundle["ood_mu"]) / bundle["ood_sd"]
    nn = NearestNeighbors(n_neighbors=bundle["ood_k"], metric="euclidean").fit(bundle["ref_std"])
    dist, _ = nn.kneighbors(fz)
    d = float(dist.mean())
    out["ood"] = bool(d > bundle["ood_threshold"])
    out["ood_distance"] = d
    out["ood_threshold"] = bundle["ood_threshold"]
    return out

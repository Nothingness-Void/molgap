"""Zero-training post-hoc analysis of saved development predictions.

Output is exploratory evidence only (`posthoc` is always true); it is never a
gate decision and never touches RML records.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .paths import file_digest

PROTECTED_ROLES = frozenset({"official_validation", "test_dev", "test_challenge"})
OPS = ("paired", "ensemble-curve", "residual-corr")
_MEV = 1000.0


def load_aligned(paths):
    """Return (names, signed errors in eV, source_idx); every file must share rows exactly."""
    import torch

    names, errors, ids, target = [], [], None, None
    for path in map(Path, paths):
        value = torch.load(path, map_location="cpu", weights_only=True)
        source = value["source_idx"].reshape(-1).numpy()
        truth = value["target_eV"].reshape(-1).double().numpy()
        pred = value["prediction_eV"].reshape(-1).double().numpy()
        if not (np.isfinite(truth).all() and np.isfinite(pred).all()) or len(source) != len(pred):
            raise ValueError(f"{path}: non-finite values or length mismatch")
        order = np.argsort(source)
        source, truth, pred = source[order], truth[order], pred[order]
        if ids is None:
            ids, target = source, truth
        elif not np.array_equal(ids, source) or not np.allclose(target, truth, atol=1e-6):
            raise ValueError(f"{path}: source_idx or targets differ from the first file")
        names.append(str(path))
        errors.append(pred - target)
    if len(names) < 2:
        raise ValueError("Need at least two prediction files")
    return names, np.stack(errors), ids


def _mae(error) -> float:
    return float(np.abs(error).mean() * _MEV)


def run(op, paths, *, role_id, n_bootstrap=2000, seed=42):
    if role_id in PROTECTED_ROLES or not role_id:
        raise ValueError("Refusing protected or missing role: pass an unprotected --role-id")
    if op not in OPS:
        raise ValueError(f"Unknown op {op!r}")
    names, errors, ids = load_aligned(paths)
    singles = {n: _mae(e) for n, e in zip(names, errors)}
    if op == "paired":
        from molgap.router import paired_bootstrap_mean

        # first file is the reference; delta < 0 means the candidate is better
        results = {"reference": names[0], "reference_mae_meV": singles[names[0]], "candidates": {
            n: {"mae_meV": singles[n], **paired_bootstrap_mean(
                (np.abs(e) - np.abs(errors[0])) * _MEV, n_bootstrap=n_bootstrap, seed=seed)}
            for n, e in zip(names[1:], errors[1:])}}
    elif op == "ensemble-curve":
        curve = {str(k): float(np.mean([_mae(errors[list(c)].mean(0)) for c in
                                        itertools.combinations(range(len(names)), k)]))
                 for k in range(1, len(names) + 1)}
        results = {"mean_single_mae_meV": float(np.mean(list(singles.values()))), "singles_meV": singles,
                   "mae_by_n_members_meV": curve, "gain_all_members_meV":
                   curve["1"] - curve[str(len(names))]}
    else:
        corr = np.corrcoef(errors)
        pairs = {f"{i}-{j}": float(corr[i, j]) for i, j in itertools.combinations(range(len(names)), 2)}
        results = {"members": names, "pair_correlation": pairs,
                   "mean_pair_correlation": float(np.mean(list(pairs.values())))}
    return {"op": op, "role_id": role_id, "inputs": [{"path": n, "sha256": file_digest(Path(n))} for n in names],
            "row_count": int(len(ids)), "source_idx_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            "results": results, "posthoc": True}


def write_output(result, out, repo_root):
    out, root = Path(out).resolve(), Path(repo_root).resolve()
    if any(out.is_relative_to(root / name) for name in ("research_memory", "experiments")):
        raise ValueError("analyze writes only to a scratch path, never to RML or experiments")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

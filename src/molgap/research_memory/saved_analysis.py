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
OPS = ("paired", "ensemble-curve", "residual-corr", "summary", "floor")
_MEV = 1000.0


def load_aligned(paths):
    """Return (names, signed errors in eV, source_idx); every file must share rows exactly."""
    import torch

    names, errors, ids, target = [], [], None, None
    for name, path in (paths.items() if isinstance(paths, dict) else ((str(p), p) for p in paths)):
        path = Path(path)
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
        names.append(name)
        errors.append(pred - target)
    if len(names) < 2:
        raise ValueError("Need at least two prediction files")
    return names, np.stack(errors), ids


def _mae(error) -> float:
    return float(np.abs(error).mean() * _MEV)


def _curve(errors):
    """Mean-of-k MAE for every member subset: [(k, mae_meV)]; used for the E_N gain and the a + b/k fit."""
    return [(k, _mae(errors[list(c)].mean(0))) for k in range(1, len(errors) + 1)
            for c in itertools.combinations(range(len(errors)), k)]


def _summary(names, errors, singles, pairs, n_bootstrap, seed):
    from molgap.router import paired_bootstrap_mean

    points = _curve(errors) if len(names) <= 12 else None
    full = errors.mean(0)
    out = {"singles_meV": singles, "n_inits": len(names), "D_meV": max(singles.values()) - min(singles.values()),
           "ensemble_mae_meV": _mae(full), "ensemble_gain_meV": float(np.mean(list(singles.values()))) - _mae(full),
           "fit": None, "pairs": [], "m_meV": None}
    if points and len(names) >= 3:
        k = np.array([x for x, _ in points], dtype=float)
        a, b = np.linalg.lstsq(np.stack([np.ones_like(k), 1 / k], 1), np.array([y for _, y in points]), rcond=None)[0]
        out["fit"] = {"form": "a + b/N over all member subsets", "a_meV": float(a), "b_meV": float(b)}
    at = {n: i for i, n in enumerate(names)}
    for cand, ref in pairs or ():
        # delta < 0 means the candidate is better; the CI is row-bootstrap noise only
        delta = (np.abs(errors[at[cand]]) - np.abs(errors[at[ref]])) * _MEV
        out["pairs"].append({"candidate": cand, "reference": ref, **paired_bootstrap_mean(delta, n_bootstrap=n_bootstrap, seed=seed)})
    if out["pairs"]:
        out["m_meV"] = float(np.mean([p["delta"] for p in out["pairs"]]))
    return out


def _floor(errors):
    _, keep = np.unique(errors.round(9), axis=0, return_index=True)  # bit-identical reruns count once
    e = errors[np.sort(keep)]
    mean = e.mean(0)
    wrong = (np.sign(e) == np.sign(e[0])).all(0) & (np.abs(e) > 0.3).all(0)
    return {"members_used": int(len(e)), "shared_fraction_of_mse": float((mean ** 2).mean() / (e ** 2).mean()),
            "all_wrong_rows": int(wrong.sum()), "oracle_mae_meV": float(np.abs(e).min(0).mean() * _MEV)}


def discover_predictions(repo_root):
    """{row_set_sha16: [{trajectory_id, path, sha256}]} from RML-bound, hash-verified, unprotected predictions."""
    import torch

    root, pools, seen = Path(repo_root).resolve(), {}, set()
    for row in json.loads((root / "research_memory/derived/trajectory_index.json").read_text(encoding="utf-8"))["trajectories"]:
        record = json.loads((root / row["record_path"]).read_text(encoding="utf-8"))
        for ref in (record.get("result") or {}).get("evidence_refs") or []:
            path = root / ref
            if path.suffix != ".json" or not path.is_file():
                continue
            evidence = json.loads(path.read_text(encoding="utf-8"))
            if any((evidence.get("role_use") or {}).get(r) not in (None, "untouched") for r in PROTECTED_ROLES):
                continue
            for item in evidence.get("artifacts") or []:
                loc = Path(item["locator"].replace("external:///", ""))
                if item.get("name") != "predictions" or loc in seen or not loc.is_file() or file_digest(loc) != item.get("sha256"):
                    continue
                seen.add(loc)
                ids = torch.load(loc, map_location="cpu", weights_only=True)["source_idx"].reshape(-1).numpy()
                key = hashlib.sha256(np.sort(ids).astype("<i8").tobytes()).hexdigest()[:16]
                pools.setdefault(key, []).append({"trajectory_id": row["trajectory_id"], "path": str(loc), "sha256": item["sha256"]})
    return pools


def run(op, paths, *, role_id, n_bootstrap=2000, seed=42, pairs=None):
    if role_id in PROTECTED_ROLES or not role_id:
        raise ValueError("Refusing protected or missing role: pass an unprotected --role-id")
    if op not in OPS:
        raise ValueError(f"Unknown op {op!r}")
    names, errors, ids = load_aligned(paths)
    files = dict(paths) if isinstance(paths, dict) else {str(p): p for p in paths}
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
    elif op == "summary":
        results = _summary(names, errors, singles, pairs, n_bootstrap, seed)
    elif op == "floor":
        results = _floor(errors)
    else:
        corr = np.corrcoef(errors)
        pairs = {f"{i}-{j}": float(corr[i, j]) for i, j in itertools.combinations(range(len(names)), 2)}
        results = {"members": names, "pair_correlation": pairs,
                   "mean_pair_correlation": float(np.mean(list(pairs.values())))}
    return {"op": op, "role_id": role_id, "inputs": [{"path": str(files[n]), "sha256": file_digest(Path(files[n]))} for n in names],
            "row_count": int(len(ids)), "source_idx_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            "results": results, "posthoc": True}


def write_output(result, out, repo_root):
    out, root = Path(out).resolve(), Path(repo_root).resolve()
    if any(out.is_relative_to(root / name) for name in ("research_memory", "experiments")):
        raise ValueError("analyze writes only to a scratch path, never to RML or experiments")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

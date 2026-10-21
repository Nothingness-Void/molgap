"""Reusable post-training diagnostic and failure-mode attribution for family runs.

Replaces one-off experiment analysis scripts with a deterministic, read-only
analyzer over retained ``molgap-family-output-v1`` artifacts:
- Contract and 50K development row alignment verification
- Paired bootstrap mean/CI, material & noninferiority gates, row win/loss/tie
- Global signed bias and post-hoc same-role optimal scalar shift decomposition
- Tail absolute error quantiles (p50/p75/p90/p95/p99/max) and worst-row table
- Target HOMO-LUMO Gap regime stratification (physical eV bins + quartiles)
- Complementarity: residual correlation, 50/50 blend, and grid-oracle blend
- Trace dynamics: early/mid/late pass windows, best-vs-final, Live-vs-EMA,
  and online train MAE (in eV) vs. fixed-weight development MAE
- Runtime cost & profiler summary
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .common.io import _safe_local
from .router import paired_bootstrap_mean
from .training_reproducibility import atomic_json, sha256_file

ATTRIBUTION_FORMAT = "molgap-family-attribution-v1"
PRIMARY_METRICS = {
    "live": "live_dev_metric",
    "ema": "ema_dev_metric",
    "clean-ema": "ema_dev_metric",
}


def _read_json(path: Path) -> Any:
    _safe_local(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _pearson(a, b) -> float:
    import numpy as np
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = float(np.sqrt(np.dot(vx, vx) * np.dot(vy, vy)))
    return float(np.dot(vx, vy) / denom) if denom > 0.0 else 0.0


def _compare_paired_errors(
    baseline_abs_error,
    candidate_abs_error,
    *,
    threshold_eV: float = 0.003,
    noninferiority_margin_eV: float = 0.0005,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    base = np.asarray(baseline_abs_error, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate_abs_error, dtype=np.float64).reshape(-1)
    if base.size != cand.size or base.size == 0:
        raise ValueError("Paired error arrays must be non-empty and identical length")
    if not np.isfinite(base).all() or not np.isfinite(cand).all():
        raise ValueError("Paired errors must be finite")
    if type(n_bootstrap) is not int or n_bootstrap < 0:
        raise ValueError("Bootstrap samples must be a nonnegative integer")
    for value in (threshold_eV, noninferiority_margin_eV):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Comparison margins must be finite and nonnegative")
    delta = cand - base
    boot = paired_bootstrap_mean(delta, n_bootstrap=n_bootstrap, seed=seed)
    ci95 = boot["ci95"]
    lo, hi = (float(ci95[0]), float(ci95[1])) if ci95 is not None else (None, None)
    mean_delta = float(boot["delta"])
    return {
        "n_molecules": int(base.size),
        "baseline_mae_eV": float(base.mean()),
        "candidate_mae_eV": float(cand.mean()),
        "mean_delta_eV": mean_delta,
        "lower_95_eV": lo,
        "upper_95_eV": hi,
        "probability_candidate_better": boot["probability_better"],
        "threshold_eV": float(threshold_eV),
        "noninferiority_margin_eV": float(noninferiority_margin_eV),
        "passes_material_gate": bool(hi is not None and hi <= -float(threshold_eV)),
        "passes_noninferiority_gate": bool(hi is not None and hi <= float(noninferiority_margin_eV)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def _normalize_trace_rows(trace: dict[str, Any], std_eV: float) -> list[dict[str, Any]]:
    """Normalize canonical RML trace observations or legacy evaluation rows."""
    if "observations" in trace:
        semantics = trace.get("metric_semantics") or {}
        train_unit = (semantics.get("live_train_metric") or {}).get("unit", "normalized-Gap")
        rows = []
        for item in trace["observations"]:
            if item.get("event") != "observation":
                continue
            raw_train = item.get("live_train_metric")
            train_eV = (
                None if raw_train is None else
                float(raw_train) if train_unit == "eV" else float(raw_train) * std_eV
            )
            rows.append({
                "pass_index": float(item["epoch_or_pass"]),
                "optimizer_steps": int(item["optimizer_step"]),
                "sample_presentations": int(item["sample_presentations"]),
                "online_train_mae_eV": train_eV,
                "live_dev_metric": float(item["live_dev_metric"]) if item.get("live_dev_metric") is not None else None,
                "ema_dev_metric": float(item["ema_dev_metric"]) if item.get("ema_dev_metric") is not None else None,
                "learning_rate": float(item["learning_rate"]) if item.get("learning_rate") is not None else None,
                "wall_time_seconds": float(item["wall_time_seconds"]) if item.get("wall_time_seconds") is not None else None,
            })
        return rows
    rows = []
    for item in trace.get("evaluations", []):
        raw_train = item.get("train_loss_normalized_mae")
        rows.append({
            "pass_index": float(item["pass_index"]),
            "optimizer_steps": int(item["optimizer_steps"]),
            "sample_presentations": int(item.get("sample_presentations", 0)),
            "online_train_mae_eV": float(raw_train) * std_eV if raw_train is not None else None,
            "live_dev_metric": float(item["live_mae_eV"]) if item.get("live_mae_eV") is not None else None,
            "ema_dev_metric": (
                float(item["ema_mae_eV"]) if item.get("ema_mae_eV") is not None else
                float(item["clean_ema_mae_eV"]) if item.get("clean_ema_mae_eV") is not None else None
            ),
            "learning_rate": float(item["learning_rate"]) if item.get("learning_rate") is not None else None,
            "wall_time_seconds": None,
        })
    return rows


def _load_arm_artifacts(arm_dir: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from .experiment_family_workflow import RunContext, _artifact_path, _json, inspect_output
    from .family_training_runner import INTERFACES, read_recipe

    root = Path(arm_dir).absolute()
    _safe_local(root)
    manifest_path = root / "output_manifest.json"
    manifest = _json(manifest_path)
    try:
        context = RunContext(**manifest["context"])
        contract_path = _artifact_path(root, manifest["artifacts"]["contract"]["path"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Family attribution requires a bound output manifest") from exc
    contract_raw, recipe = read_recipe(contract_path)
    if INTERFACES.get(recipe.family) != (context.family_name, context.family_version):
        raise ValueError("Output family differs from its executable contract")
    if not recipe.dataset.development:
        raise ValueError("Attribution requires explicitly selected development outputs")
    acceptance = inspect_output(root, context=context, expected=contract_raw["acceptance_requirements"])
    if acceptance["status"] != "MECHANICALLY_VERIFIED":
        raise ValueError("Family output failed mechanical inspection: " + "; ".join(acceptance["blockers"]))
    # The existing owner verifies artifact/context/row/target/selection bindings.
    paths = {role: _artifact_path(root, entry["path"]) for role, entry in manifest["artifacts"].items()}
    trace_path, predictions_path = paths["trace"], paths["predictions"]
    trace = _json(trace_path)
    payload = torch.load(predictions_path, map_location="cpu", weights_only=True)
    source_idx, target, pred = (payload[key].detach().cpu() for key in ("source_idx", "target_eV", "prediction_eV"))
    observed = acceptance["observed"]
    selected_step = observed["selected_optimizer_step"]
    selected_info = {
        "pass_index": selected_step / recipe.loop_plan.steps_per_pass,
        "optimizer_steps": selected_step,
        "mae_eV": observed["development_mae_eV"],
        "weights": observed["selected_weights"],
    }

    timings_path = root / "epoch_step_timings.json"
    timings = _read_json(timings_path) if timings_path.is_file() else None
    profile_path = root / "profile_summary.json"
    profile = _read_json(profile_path) if profile_path.is_file() else None

    wall_cost = next((item for item in manifest["costs"]
                      if item["metric"] == "wall_seconds" and item["semantics"] == "process_wall"), None)
    wall_seconds = wall_cost["value"] if wall_cost is not None else None

    return {
        "root": root,
        "arm_id": context.arm_id,
        "context": context,
        "wall_seconds": float(wall_seconds) if wall_seconds is not None else None,
        "wall_cost": wall_cost,
        "manifest": manifest,
        "contract": contract_raw,
        "recipe": recipe.to_dict(),
        "selection_weights": recipe.selection_weights,
        "trace": trace,
        "selected_checkpoint": selected_info,
        "payload": payload,
        "source_idx": source_idx.numpy().astype(np.int64, copy=False),
        "target_eV": target.numpy().astype(np.float64, copy=False),
        "prediction_eV": pred.numpy().astype(np.float64, copy=False),
        "timings": timings,
        "profile": profile,
        "hashes": {
            "output_manifest_sha256": sha256_file(manifest_path),
            "training_contract_sha256": sha256_file(contract_path),
            "canonical_trace_sha256": sha256_file(trace_path),
            "development_predictions_sha256": sha256_file(predictions_path),
        },
    }


def _optimal_constant_shift(signed_error) -> tuple[float, float]:
    """Return (optimal_shift_eV, shifted_mae_eV) minimizing L1 error on this role."""
    import numpy as np
    shift = -float(np.median(signed_error))
    shifted_mae = float(np.mean(np.abs(signed_error + shift)))
    return shift, shifted_mae


def _quantiles(abs_err) -> dict[str, float]:
    import numpy as np
    return {
        "p50_eV": float(np.quantile(abs_err, 0.50)),
        "p75_eV": float(np.quantile(abs_err, 0.75)),
        "p90_eV": float(np.quantile(abs_err, 0.90)),
        "p95_eV": float(np.quantile(abs_err, 0.95)),
        "p99_eV": float(np.quantile(abs_err, 0.99)),
        "max_eV": float(np.max(abs_err)),
    }


def _strata_table(target, ctrl_abs, cand_abs) -> dict[str, list[dict[str, Any]]]:
    import numpy as np

    def summarize(label: str, mask) -> dict[str, Any]:
        count = int(np.sum(mask))
        if count == 0:
            return {
                "stratum": label,
                "rows": 0,
                "target_mean_eV": None,
                "control_mae_eV": None,
                "candidate_mae_eV": None,
                "delta_mae_eV": None,
                "candidate_win_fraction": None,
            }
        c_err = ctrl_abs[mask]
        x_err = cand_abs[mask]
        c_mae = float(np.mean(c_err))
        x_mae = float(np.mean(x_err))
        return {
            "stratum": label,
            "rows": count,
            "target_mean_eV": float(np.mean(target[mask])),
            "control_mae_eV": c_mae,
            "candidate_mae_eV": x_mae,
            "delta_mae_eV": x_mae - c_mae,
            "candidate_win_fraction": float(np.mean(x_err < c_err)),
        }

    physical_bins = [
        ("<4.0_eV", target < 4.0),
        ("4.0_to_6.0_eV", (target >= 4.0) & (target < 6.0)),
        ("6.0_to_8.0_eV", (target >= 6.0) & (target < 8.0)),
        (">=8.0_eV", target >= 8.0),
    ]
    q25, q50, q75 = [float(v) for v in np.quantile(target, [0.25, 0.50, 0.75])]
    quartile_bins = [
        ("Q1_0_25", target <= q25),
        ("Q2_25_50", (target > q25) & (target <= q50)),
        ("Q3_50_75", (target > q50) & (target <= q75)),
        ("Q4_75_100", target > q75),
    ]
    return {
        "physical_gap_bins": [summarize(label, mask) for label, mask in physical_bins],
        "target_quartiles": [summarize(label, mask) for label, mask in quartile_bins],
    }


def _trace_dynamics(ctrl: dict[str, Any], cand: dict[str, Any], metric_key: str) -> dict[str, Any]:
    import numpy as np

    std_eV = float(ctrl["recipe"]["target_std_eV"])
    ctrl_evals = _normalize_trace_rows(ctrl["trace"], std_eV)
    cand_evals = _normalize_trace_rows(cand["trace"], std_eV)

    def arm_curve_summary(arm: dict[str, Any], evals: list[dict[str, Any]]) -> dict[str, Any]:
        if not evals:
            return {"evaluations": 0}
        valid = [row for row in evals if row.get(metric_key) is not None]
        if not valid:
            return {"evaluations": len(evals)}
        best_row = min(valid, key=lambda r: float(r[metric_key]))
        final_row = valid[-1]
        best_val = float(best_row[metric_key])
        final_val = float(final_row[metric_key])
        selected = arm.get("selected_checkpoint") or {}
        sel_pass = selected.get("pass_index", best_row["pass_index"])
        sel_step = selected.get("optimizer_steps", best_row["optimizer_steps"])
        sel_mae = selected.get("mae_eV", best_val)
        best_train_eV = best_row.get("online_train_mae_eV")
        final_train_eV = final_row.get("online_train_mae_eV")
        live_minus_ema_selected = (
            float(best_row["live_dev_metric"]) - float(best_row["ema_dev_metric"])
            if best_row.get("live_dev_metric") is not None and best_row.get("ema_dev_metric") is not None
            else None
        )
        live_minus_ema_final = (
            float(final_row["live_dev_metric"]) - float(final_row["ema_dev_metric"])
            if final_row.get("live_dev_metric") is not None and final_row.get("ema_dev_metric") is not None
            else None
        )
        return {
            "evaluations": len(valid),
            "selected_pass": float(sel_pass) if sel_pass is not None else float(best_row["pass_index"]),
            "selected_step": int(sel_step) if sel_step is not None else int(best_row["optimizer_steps"]),
            "selected_mae_eV": float(sel_mae) if sel_mae is not None else best_val,
            "final_pass": float(final_row["pass_index"]),
            "final_step": int(final_row["optimizer_steps"]),
            "final_mae_eV": final_val,
            "best_minus_final_eV": best_val - final_val,
            "online_train_mae_at_selected_eV": best_train_eV,
            "online_train_mae_at_final_eV": final_train_eV,
            "dev_minus_online_train_at_selected_eV": (
                best_val - best_train_eV if best_train_eV is not None else None
            ),
            "dev_minus_online_train_at_final_eV": (
                final_val - final_train_eV if final_train_eV is not None else None
            ),
            "live_minus_ema_at_selected_eV": live_minus_ema_selected,
            "live_minus_ema_at_final_eV": live_minus_ema_final,
        }

    paired_deltas = []
    if len(ctrl_evals) == len(cand_evals) and ctrl_evals:
        for c_item, x_item in zip(ctrl_evals, cand_evals):
            if (c_item.get("optimizer_steps") == x_item.get("optimizer_steps")
                    and c_item.get(metric_key) is not None
                    and x_item.get(metric_key) is not None):
                paired_deltas.append({
                    "pass_index": float(c_item["pass_index"]),
                    "optimizer_steps": int(c_item["optimizer_steps"]),
                    "control_mae_eV": float(c_item[metric_key]),
                    "candidate_mae_eV": float(x_item[metric_key]),
                    "delta_mae_eV": float(x_item[metric_key]) - float(c_item[metric_key]),
                })

    def window_mean(items: list[dict[str, Any]]) -> float | None:
        if not items:
            return None
        return float(np.mean([row["delta_mae_eV"] for row in items]))

    n = len(paired_deltas)
    windows = {
        "early_passes_1_to_10_mean_delta_eV": window_mean(paired_deltas[: min(10, n)]),
        "mid_passes_11_to_20_mean_delta_eV": window_mean(paired_deltas[10: min(20, n)]) if n > 10 else None,
        "late_last_10_passes_mean_delta_eV": window_mean(paired_deltas[max(0, n - 10):]),
        "late_last_5_passes_mean_delta_eV": window_mean(paired_deltas[max(0, n - 5):]),
        "candidate_ahead_evaluations": int(sum(1 for row in paired_deltas if row["delta_mae_eV"] < 0.0)),
        "total_paired_evaluations": n,
    }
    return {
        "primary_metric_key": metric_key,
        "control": arm_curve_summary(ctrl, ctrl_evals),
        "candidate": arm_curve_summary(cand, cand_evals),
        "paired_windows": windows,
        "paired_curve": paired_deltas,
        "online_train_note": (
            "online_train_mae_eV is an in-pass online running average "
            "under active optimization updates, not a fixed-weight evaluation pass."
        ),
    }


def _cost_and_profile(ctrl: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    def arm_cost(arm: dict[str, Any]) -> dict[str, Any]:
        wall = arm["wall_seconds"]
        info: dict[str, Any] = {"process_wall_seconds": wall, "wall_observation": arm["wall_cost"]}
        timings = arm.get("timings")
        if isinstance(timings, list) and timings:
            info["passes_recorded"] = len(timings)
            for key in ("epoch_pipeline_seconds", "train_seconds", "development_seconds"):
                vals = [float(t[key]) for t in timings if key in t and t[key] is not None]
                if vals:
                    info[f"median_{key}"] = float(np.median(vals))
                    info[f"mean_{key}"] = float(np.mean(vals))
        profile = arm.get("profile")
        if isinstance(profile, dict):
            info["profile_summary_present"] = True
            top_ops = profile.get("top_operators") or profile.get("operators")
            if isinstance(top_ops, list):
                info["top_operators"] = top_ops[:5]
        else:
            info["profile_summary_present"] = False
        return info

    c_cost = arm_cost(ctrl)
    x_cost = arm_cost(cand)
    c_wall = c_cost["process_wall_seconds"]
    x_wall = x_cost["process_wall_seconds"]
    both_measured = all(arm["wall_cost"] is not None and arm["wall_cost"]["status"] == "measured"
                        for arm in (ctrl, cand))
    return {
        "control": c_cost,
        "candidate": x_cost,
        "candidate_to_control_wall_ratio": (x_wall / c_wall if both_measured and c_wall > 0.0 else None),
        "candidate_minus_control_wall_seconds": x_wall - c_wall if both_measured else None,
    }


def _derive_signals(
    paired: dict[str, Any],
    bias: dict[str, Any],
    complementarity: dict[str, Any],
    strata: dict[str, Any],
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    delta = float(paired["mean_delta_eV"])
    if paired["upper_95_eV"] is None:
        direction = "INCONCLUSIVE_NO_CI"
    elif paired["passes_material_gate"]:
        direction = "POSITIVE_MATERIAL"
    elif delta < 0.0 and paired["upper_95_eV"] is not None and paired["upper_95_eV"] < 0.0:
        direction = "POSITIVE_SUB_THRESHOLD"
    elif paired["passes_noninferiority_gate"] and abs(delta) <= 0.0005:
        direction = "NEUTRAL_EQUIVALENT"
    elif delta > 0.0 and paired["lower_95_eV"] is not None and paired["lower_95_eV"] > 0.0:
        direction = "NEGATIVE_DEGRADATION"
    else:
        direction = "INCONCLUSIVE_OVERLAPPING_CI"

    shift_explained = bias.get("fraction_of_delta_removed_by_posthoc_shift")
    global_shift_dominant = bool(
        delta > 0.0005 and shift_explained is not None and shift_explained >= 0.50
    )

    windows = dynamics["paired_windows"]
    early = windows.get("early_passes_1_to_10_mean_delta_eV")
    late = windows.get("late_last_10_passes_mean_delta_eV")
    late_fade = bool(early is not None and late is not None and early < -0.0005 and late > early + 0.001)
    early_lag_recovery = bool(early is not None and late is not None and early > 0.001 and late < early - 0.001)

    phys = [row for row in strata["physical_gap_bins"] if row["delta_mae_eV"] is not None]
    worst_bin = max(phys, key=lambda r: r["delta_mae_eV"])["stratum"] if phys else None
    best_bin = min(phys, key=lambda r: r["delta_mae_eV"])["stratum"] if phys else None

    return {
        "direction": direction,
        "global_bias_shift_dominant": global_shift_dominant,
        "late_horizon_fade": late_fade,
        "early_lag_late_recovery": early_lag_recovery,
        "oracle_blend_gain_over_best_single_eV": complementarity["oracle_gain_over_best_single_eV"],
        "best_physical_gap_stratum": best_bin,
        "worst_physical_gap_stratum": worst_bin,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    p = report["primary_comparison"]
    r = report["row_win_loss"]
    b = report["signed_bias_and_shift"]
    q = report["error_quantiles"]
    c = report["complementarity"]
    d = report["trace_dynamics"]
    s = report["signals"]
    cost = report["cost"]
    ci_text = (f"[{p['lower_95_eV']:+.6f}, {p['upper_95_eV']:+.6f}] eV"
               if p["lower_95_eV"] is not None else "not measured (bootstrap disabled)")
    def seconds(value):
        return "unknown" if value is None else f"{value:.2f}s"

    lines = [
        f"# Post-Training Attribution Report (`{report['control_arm']}` vs `{report['candidate_arm']}`)",
        "",
        "## 1. Executive Summary & Signals",
        f"- **Direction**: `{s['direction']}`",
        f"- **Selection Metric**: `{report['primary_metric']}` (`{p['n_molecules']}` aligned development molecules)",
        f"- **Control MAE**: `{p['baseline_mae_eV']:.6f} eV`",
        f"- **Candidate MAE**: `{p['candidate_mae_eV']:.6f} eV`",
        f"- **Paired Delta (Candidate - Control)**: `{p['mean_delta_eV']:+.6f} eV` "
        f"(95% CI `{ci_text}`)",
        f"- **Material Gate (`<= -{p['threshold_eV']:.4f} eV`)**: `{p['passes_material_gate']}` | "
        f"**Noninferiority Gate (`<= +{p['noninferiority_margin_eV']:.4f} eV`)**: `{p['passes_noninferiority_gate']}`",
        f"- **Row Win / Loss / Tie**: `{r['candidate_better_rows']}` / `{r['control_better_rows']}` / `{r['tie_rows']}` "
        f"(Candidate win rate `{r['candidate_win_fraction'] * 100:.2f}%`, median row gain `{r['median_row_gain_eV']:+.6f} eV`)",
        f"- **Global Bias Shift Dominant**: `{s['global_bias_shift_dominant']}` | "
        f"**Late-Horizon Fade**: `{s['late_horizon_fade']}` | "
        f"**Early-Lag Recovery**: `{s['early_lag_late_recovery']}`",
        "",
        "## 2. Global Signed Bias & Post-Hoc Scalar Shift Diagnostic",
        "| Arm | Raw MAE (eV) | Mean Signed Error (eV) | Post-Hoc Optimal Shift (eV) | Shift-Corrected MAE (eV) |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `{report['control_arm']}` | `{p['baseline_mae_eV']:.6f}` | `{b['control']['mean_signed_error_eV']:+.6f}` | "
        f"`{b['control']['optimal_constant_shift_eV']:+.6f}` | `{b['control']['shift_corrected_mae_eV']:.6f}` |",
        f"| `{report['candidate_arm']}` | `{p['candidate_mae_eV']:.6f}` | `{b['candidate']['mean_signed_error_eV']:+.6f}` | "
        f"`{b['candidate']['optimal_constant_shift_eV']:+.6f}` | `{b['candidate']['shift_corrected_mae_eV']:.6f}` |",
        f"- **Shift-Corrected Delta (Candidate - Control)**: `{b['shift_corrected_delta_eV']:+.6f} eV` "
        f"(Raw Delta `{p['mean_delta_eV']:+.6f} eV`)",
        "",
        "## 3. Tail Absolute Error Quantiles (eV)",
        "| Quantile | Control (eV) | Candidate (eV) | Delta (Candidate - Control) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("p50_eV", "p75_eV", "p90_eV", "p95_eV", "p99_eV", "max_eV"):
        ctrl_v = q["control"][key]
        cand_v = q["candidate"][key]
        lines.append(f"| `{key}` | `{ctrl_v:.6f}` | `{cand_v:.6f}` | `{cand_v - ctrl_v:+.6f}` |")

    lines.extend([
        "",
        "## 4. Target HOMO-LUMO Gap Regime Stratification",
        "| Stratum | Rows | Target Mean (eV) | Control MAE (eV) | Candidate MAE (eV) | Delta (eV) | Candidate Win % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["strata"]["physical_gap_bins"]:
        if row["rows"] == 0:
            continue
        lines.append(
            f"| `{row['stratum']}` | `{row['rows']}` | `{row['target_mean_eV']:.4f}` | "
            f"`{row['control_mae_eV']:.6f}` | `{row['candidate_mae_eV']:.6f}` | "
            f"`{row['delta_mae_eV']:+.6f}` | `{row['candidate_win_fraction'] * 100:.2f}%` |"
        )

    w = d["paired_windows"]
    lines.extend([
        "",
        "## 5. Complementarity & Ensemble Check",
        f"- **Mean |Pred_Cand - Pred_Ctrl|**: `{c['mean_abs_prediction_diff_eV']:.6f} eV`",
        f"- **Signed Residual Pearson r**: `{c['signed_residual_pearson_r']:.6f}` | "
        f"**Absolute Error Pearson r**: `{c['abs_error_pearson_r']:.6f}`",
        f"- **50/50 Equal Blend MAE**: `{c['equal_blend_mae_eV']:.6f} eV`",
        f"- **Grid-Oracle Blend MAE**: `{c['oracle_blend_mae_eV']:.6f} eV` "
        f"(optimal candidate weight `w = {c['oracle_candidate_weight']:.2f}`, "
        f"gain over best single `{c['oracle_gain_over_best_single_eV']:+.6f} eV`)",
        "",
        "## 6. Optimization & Trace Dynamics",
        f"- **Control Selected Pass / Final Pass**: `{d['control'].get('selected_pass')}` (`{d['control'].get('selected_mae_eV', 0.0):.6f} eV`) "
        f"vs `{d['control'].get('final_pass')}` (`{d['control'].get('final_mae_eV', 0.0):.6f} eV`)",
        f"- **Candidate Selected Pass / Final Pass**: `{d['candidate'].get('selected_pass')}` (`{d['candidate'].get('selected_mae_eV', 0.0):.6f} eV`) "
        f"vs `{d['candidate'].get('final_pass')}` (`{d['candidate'].get('final_mae_eV', 0.0):.6f} eV`)",
        f"- **Early Window (Passes 1..10) Mean Delta**: `{w['early_passes_1_to_10_mean_delta_eV']:+.6f} eV`"
        if w["early_passes_1_to_10_mean_delta_eV"] is not None else "- **Early Window**: `N/A`",
        f"- **Late Window (Last 10 Passes) Mean Delta**: `{w['late_last_10_passes_mean_delta_eV']:+.6f} eV` "
        f"| **Last 5 Passes**: `{w['late_last_5_passes_mean_delta_eV']:+.6f} eV`"
        if w["late_last_10_passes_mean_delta_eV"] is not None else "- **Late Window**: `N/A`",
        f"- **Candidate Ahead in Evaluations**: `{w['candidate_ahead_evaluations']} / {w['total_paired_evaluations']}`",
        "",
        "## 7. Runtime Cost",
        f"- **Process Wall**: control `{seconds(cost['control']['process_wall_seconds'])}`, "
        f"candidate `{seconds(cost['candidate']['process_wall_seconds'])}`",
        f"- **Wall Ratio**: `{cost['candidate_to_control_wall_ratio']:.4f}x`"
        if cost["candidate_to_control_wall_ratio"] is not None else "- **Wall Ratio**: `N/A (requires measured costs)`",
        "- Post-hoc descriptive analysis only; row bootstrap does not measure training stochasticity, qualify a strict reference or authorize promotion.",
        "",
    ])
    return "\n".join(lines)


def _plot_curves(report: dict[str, Any], path: Path) -> bool:
    curve = report["trace_dynamics"].get("paired_curve", [])
    if not curve:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    passes = [row["pass_index"] for row in curve]
    ctrl_y = [row["control_mae_eV"] for row in curve]
    cand_y = [row["candidate_mae_eV"] for row in curve]
    delta_y = [row["delta_mae_eV"] * 1000.0 for row in curve]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    ax1.plot(passes, ctrl_y, label=f"Control ({report['control_arm']})", linewidth=1.8)
    ax1.plot(passes, cand_y, label=f"Candidate ({report['candidate_arm']})", linewidth=1.8)
    ax1.set_ylabel(f"Dev {report['primary_metric']} (eV)")
    ax1.set_title(f"Development Trajectory: {report['control_arm']} vs {report['candidate_arm']}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.plot(passes, delta_y, color="tab:red", linewidth=1.6, label="Candidate - Control (meV)")
    ax2.set_xlabel("Pass Index")
    ax2.set_ylabel("Delta MAE (meV)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def analyze_family_pair(
    control_dir: Path | str,
    candidate_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    primary_metric: str | None = None,
    threshold_eV: float = 0.003,
    noninferiority_margin_eV: float = 0.0005,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full diagnostic & failure-mode attribution between two family training arms."""
    import numpy as np

    ctrl = _load_arm_artifacts(Path(control_dir))
    cand = _load_arm_artifacts(Path(candidate_dir))

    # 1. Verify contract & row alignment
    c_rec = ctrl["recipe"]
    x_rec = cand["recipe"]
    for field in ("dataset", "seed", "target_mean_eV", "target_std_eV"):
        if c_rec.get(field) != x_rec.get(field):
            raise ValueError(f"Control and candidate recipe mismatch on frozen field '{field}'")
    if ctrl["contract"]["development_role_identity"] != cand["contract"]["development_role_identity"]:
        raise ValueError("Control and candidate development role identities differ")
    if ctrl["selection_weights"] != cand["selection_weights"]:
        raise ValueError("Control and candidate selection weights differ")
    if not np.array_equal(ctrl["source_idx"], cand["source_idx"]):
        raise ValueError("Control and candidate development_predictions.pt source_idx are not aligned")
    if not np.allclose(ctrl["target_eV"], cand["target_eV"], rtol=0.0, atol=1e-7):
        raise ValueError("Control and candidate development_predictions.pt target_eV differ")

    weights_mode = ctrl["selection_weights"]
    metric_key = primary_metric or PRIMARY_METRICS[weights_mode]
    if metric_key != PRIMARY_METRICS[weights_mode]:
        raise ValueError("Primary curve metric must match the selected prediction weights")

    target = ctrl["target_eV"]
    ctrl_pred = ctrl["prediction_eV"]
    cand_pred = cand["prediction_eV"]
    ctrl_signed = ctrl_pred - target
    cand_signed = cand_pred - target
    ctrl_abs = np.abs(ctrl_signed)
    cand_abs = np.abs(cand_signed)
    row_delta = cand_abs - ctrl_abs  # negative means candidate improved

    # 2. Primary Paired Bootstrap & Row Win/Loss
    paired = _compare_paired_errors(
        ctrl_abs,
        cand_abs,
        threshold_eV=threshold_eV,
        noninferiority_margin_eV=noninferiority_margin_eV,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    row_gain = ctrl_abs - cand_abs  # positive means candidate beat control
    q25_gain, q50_gain, q75_gain = [float(v) for v in np.quantile(row_gain, [0.25, 0.50, 0.75])]
    row_win_loss = {
        "candidate_better_rows": int(np.sum(cand_abs < ctrl_abs)),
        "control_better_rows": int(np.sum(cand_abs > ctrl_abs)),
        "tie_rows": int(np.sum(cand_abs == ctrl_abs)),
        "candidate_win_fraction": float(np.mean(cand_abs < ctrl_abs)),
        "median_row_gain_eV": q50_gain,
        "iqr_row_gain_eV": q75_gain - q25_gain,
    }

    # 3. Signed Error Bias & Post-hoc Constant Shift Diagnostic
    ctrl_shift, ctrl_shifted_mae = _optimal_constant_shift(ctrl_signed)
    cand_shift, cand_shifted_mae = _optimal_constant_shift(cand_signed)
    raw_delta = float(paired["mean_delta_eV"])
    shifted_delta = cand_shifted_mae - ctrl_shifted_mae
    fraction_removed = (
        float((raw_delta - shifted_delta) / raw_delta)
        if abs(raw_delta) > 1e-9 else None
    )
    signed_bias = {
        "control": {
            "mean_signed_error_eV": float(np.mean(ctrl_signed)),
            "median_signed_error_eV": float(np.median(ctrl_signed)),
            "optimal_constant_shift_eV": ctrl_shift,
            "shift_corrected_mae_eV": ctrl_shifted_mae,
        },
        "candidate": {
            "mean_signed_error_eV": float(np.mean(cand_signed)),
            "median_signed_error_eV": float(np.median(cand_signed)),
            "optimal_constant_shift_eV": cand_shift,
            "shift_corrected_mae_eV": cand_shifted_mae,
        },
        "raw_delta_eV": raw_delta,
        "shift_corrected_delta_eV": shifted_delta,
        "fraction_of_delta_removed_by_posthoc_shift": fraction_removed,
    }

    # 4. Quantiles & Top-20 Worst Rows
    ctrl_q = _quantiles(ctrl_abs)
    cand_q = _quantiles(cand_abs)
    worst_cand_idx = np.argsort(-cand_abs)[:20]
    worst_rows = [
        {
            "source_idx": int(ctrl["source_idx"][i]),
            "target_eV": float(target[i]),
            "control_abs_error_eV": float(ctrl_abs[i]),
            "candidate_abs_error_eV": float(cand_abs[i]),
            "delta_abs_error_eV": float(row_delta[i]),
        }
        for i in worst_cand_idx
    ]

    # 5. Target Gap Stratification
    strata = _strata_table(target, ctrl_abs, cand_abs)

    # 6. Complementarity & Oracle Blend
    equal_blend = 0.5 * (ctrl_pred + cand_pred)
    equal_blend_mae = float(np.mean(np.abs(equal_blend - target)))
    weights = np.linspace(0.0, 1.0, 101)
    blend_maes = [
        float(np.mean(np.abs((1.0 - w) * ctrl_pred + w * cand_pred - target)))
        for w in weights
    ]
    best_w_idx = int(np.argmin(blend_maes))
    oracle_w = float(weights[best_w_idx])
    oracle_mae = float(blend_maes[best_w_idx])
    best_single = min(float(paired["baseline_mae_eV"]), float(paired["candidate_mae_eV"]))
    complementarity = {
        "mean_abs_prediction_diff_eV": float(np.mean(np.abs(cand_pred - ctrl_pred))),
        "signed_residual_pearson_r": _pearson(ctrl_signed, cand_signed),
        "abs_error_pearson_r": _pearson(ctrl_abs, cand_abs),
        "equal_blend_mae_eV": equal_blend_mae,
        "oracle_candidate_weight": oracle_w,
        "oracle_blend_mae_eV": oracle_mae,
        "oracle_gain_over_best_single_eV": best_single - oracle_mae,
    }

    # 7. Trace Dynamics & Cost
    dynamics = _trace_dynamics(ctrl, cand, metric_key)
    cost = _cost_and_profile(ctrl, cand)
    signals = _derive_signals(paired, signed_bias, complementarity, strata, dynamics)

    ctrl_name = ctrl["arm_id"]
    cand_name = cand["arm_id"]

    report: dict[str, Any] = {
        "format": ATTRIBUTION_FORMAT,
        "status": "DIAGNOSED",
        "mechanical_acceptance": "MECHANICALLY_VERIFIED",
        "scientific_acceptance": "NOT_EVALUATED",
        "replay_readiness": "NOT_EVALUATED",
        "limitations": ["Post-hoc same-role shift/oracle blend diagnostics are not held-out gains.",
                        "Row bootstrap does not measure training stochasticity or establish strict V5 comparison."],
        "control_arm": ctrl_name,
        "candidate_arm": cand_name,
        "primary_metric": metric_key,
        "alignment": {
            "rows": int(target.size),
            "source_idx_identical": True,
            "target_eV_identical": True,
            "selection_weights": weights_mode,
            "seed": int(c_rec["seed"]),
        },
        "artifact_hashes": {
            "control": ctrl["hashes"],
            "candidate": cand["hashes"],
        },
        "primary_comparison": paired,
        "row_win_loss": row_win_loss,
        "signed_bias_and_shift": signed_bias,
        "error_quantiles": {
            "control": ctrl_q,
            "candidate": cand_q,
            "delta_candidate_minus_control": {k: cand_q[k] - ctrl_q[k] for k in ctrl_q},
            "candidate_top20_worst_rows": worst_rows,
        },
        "strata": strata,
        "complementarity": complementarity,
        "trace_dynamics": dynamics,
        "cost": cost,
        "signals": signals,
    }

    if output_dir is not None:
        out = Path(output_dir).absolute()
        _safe_local(out)
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "analysis_summary.json"
        md_path = out / "attribution_report.md"
        curve_path = out / "development_curves.png"
        atomic_json(summary_path, report)
        md_path.write_text(_render_markdown(report), encoding="utf-8")
        plotted = _plot_curves(report, curve_path)
        report["written_files"] = {
            "analysis_summary": str(summary_path),
            "attribution_report": str(md_path),
            **({"development_curves": str(curve_path)} if plotted else {}),
        }
    return report


__all__ = ["ATTRIBUTION_FORMAT", "analyze_family_pair"]

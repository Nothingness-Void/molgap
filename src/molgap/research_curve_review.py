"""Read-only learning-curve review over the existing validated RML replay pool.

No model, dataset or checkpoint is loaded. These retrospective summaries do not
change terminal labels, promotion gates or the scientific comparison class.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from statistics import mean

from .research_memory.trace import atomic_write, file_digest, json_bytes

METRICS = ("live_train_metric", "live_dev_metric", "ema_dev_metric")


def _change(rows, metric, window=10):
    values = [r[metric] for r in rows[-window:] if r.get(metric) is not None]
    return values[-1] - values[0] if len(values) >= 2 else None


def summarize_curve(rows, semantics):
    """Describe observed values, without converting units or extrapolating."""
    result = {}
    for metric in METRICS:
        observed = [r for r in rows if r.get(metric) is not None]
        if not observed:
            result[metric] = None
            continue
        if not all(math.isfinite(r[metric]) for r in observed):
            raise ValueError("Nonfinite curve metric")
        definition = semantics.get(metric)
        if not definition or definition.get("direction") != "minimize":
            raise ValueError("Review requires explicit minimize metric semantics")
        best = min(observed, key=lambda r: r[metric])
        result[metric] = {
            "semantics": definition, "best": best[metric],
            "best_optimizer_step": best["optimizer_step"],
            "best_observed_epoch_or_pass": best.get("epoch_or_pass"),
            "final": observed[-1][metric],
            "final_minus_best": observed[-1][metric] - best[metric],
            "last10_change": _change(rows, metric),
        }
    return result


def compare_curves(candidate, reference):
    """Align exact observed steps/presentations; positive gain favors candidate."""
    if candidate["comparability_key"] != reference["comparability_key"]:
        raise ValueError("Comparison worlds differ")
    if candidate["comparison_role"] != "candidate" or reference["comparison_role"] != "reference":
        raise ValueError("Expected candidate/reference views")
    ref_rows = {r["optimizer_step"]: r for r in reference["prefix_observations"]}
    if len(ref_rows) != len(reference["prefix_observations"]):
        raise ValueError("Duplicate reference steps")
    aligned = []
    for row in candidate["prefix_observations"]:
        other = ref_rows.get(row["optimizer_step"])
        if other is None:
            continue
        if row["sample_presentations"] != other["sample_presentations"]:
            raise ValueError("Presentation mismatch at shared step")
        point = {k: row[k] for k in ("optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate")}
        for metric in METRICS:
            a, b = row.get(metric), other.get(metric)
            # Normalized train loss is never subtracted from development eV.
            sa, sb = candidate["metric_semantics"].get(metric), reference["metric_semantics"].get(metric)
            compatible = sa is not None and sb is not None and all(sa.get(k) == sb.get(k)
                for k in ("unit", "target", "weights", "direction", "role_identity"))
            point[metric] = {"candidate": a, "reference": b,
                "gain": b - a if compatible and a is not None and b is not None else None}
        aligned.append(point)
    if not aligned:
        raise ValueError("No exact common exposure points")
    a = summarize_curve(candidate["prefix_observations"], candidate["metric_semantics"])
    b = summarize_curve(reference["prefix_observations"], reference["metric_semantics"])
    stats = {}
    for metric in METRICS:
        gains = [r[metric]["gain"] for r in aligned if r[metric]["gain"] is not None]
        stats[metric] = None if not gains else {
            "positive_points": sum(x > 0 for x in gains), "observed_points": len(gains),
            "final_gain": gains[-1], "last10_mean_gain": mean(gains[-10:]),
            "gain_at_observed_rungs": {str(i): aligned[i-1][metric]["gain"]
                for i in (10, 20, 30, 40, 50, 60) if i <= len(aligned)},
            "selected_curve_gain": b[metric]["best"] - a[metric]["best"],
        }
    return {"candidate_id": candidate["trajectory_id"], "reference_id": reference["trajectory_id"],
        "run_id": candidate["run_id"], "comparison_world": candidate["comparability_key"],
        "identity": candidate["comparability_identity"], "capability": candidate["capability"],
        "terminal_outcome": candidate["terminal_outcome"], "terminal_label": candidate["terminal_label"],
        "native_cost": candidate["native_cost"], "candidate_curve": a, "reference_curve": b,
        "candidate_trace_sha256": candidate["trace_sha256"],
        "reference_trace_sha256": reference["trace_sha256"],
        "aligned_points": len(aligned), "paired_metric_summaries": stats, "aligned_curve": aligned}


def review_repository(root):
    """Rebuild in memory from source owners; never rewrite derived indexes."""
    from .research_memory.validate import validate_repository_records
    from .research_memory.replay import build_replay_pool
    validated = validate_repository_records(root)
    root, records = validated["root"], validated["records"]
    pool = build_replay_pool(root, records)
    pairs = []
    for candidate in pool["entries"]:
        if candidate["comparison_role"] != "candidate":
            continue
        refs = [e for e in pool["entries"] if e["comparison_role"] == "reference"
                and e["comparability_key"] == candidate["comparability_key"]]
        if len(refs) != 1:
            raise ValueError("Ambiguous replay reference")
        pairs.append(compare_curves(candidate, refs[0]))
    sources = [{"path": p.relative_to(root).as_posix(), "sha256": file_digest(p)}
               for kind in ("trajectories", "traces", "comparison_readiness", "evidence")
               for p, _ in records[kind]]
    delivery_reviews = []
    # Delivery curves remain outside scientific replay comparisons.
    import json
    for experiment in ("pcqm_gptrans_execution40", "pcqm_gptrans_execution_tuning40"):
        path = root / "experiments" / experiment / "gpu/results/evidence_snapshot.json"
        if not path.is_file():
            continue
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        items = ([(f["path"], f["content"]) for f in snapshot["files"]]
                 if "files" in snapshot else list(snapshot["source_json"].items()))
        delivery_reviews.append({"experiment": experiment,
            "snapshot_path": path.relative_to(root).as_posix(), "snapshot_sha256": file_digest(path),
            "comparison_scope": "delivery-only; excluded from architecture replay pairs",
            "curves": [{"snapshot_member": member, "observed_points": len(trace["observations"]),
                "summary": summarize_curve(trace["observations"], trace["metric_semantics"])}
                for member, trace in items if member.endswith("canonical_trace.json")]})
    return {"analysis_kind": "retrospective-learning-curve-review",
        "training_executed": False, "model_inference_executed": False,
        "protected_role_access": False, "canonical_records_modified": False,
        "record_counts": {k: len(v) for k, v in records.items()},
        "replay": {"entries": len(pool["entries"]),
            "unique_trajectories": len({e["trajectory_id"] for e in pool["entries"]}),
            "comparable_groups": pool["comparable_group_count"], "candidate_pairs": len(pairs),
            "source_digest": pool["source_digest"], "exclusions": pool["exclusions"]},
        "trajectory_outcomes": dict(Counter(t["decision"]["outcome"] for _, t in records["trajectories"])),
        "sources": sources, "pairs": pairs, "delivery_reviews": delivery_reviews,
        "limits": ["Curve minima are producer metrics, not saved-prediction acceptance endpoints.",
            "No cross-world absolute MAE ranking; no extrapolation or interpolation.",
            "Online train/live dev/EMA dev have different evaluation semantics.",
            "Observation rung is not epoch: epoch_or_pass retains each producer's coordinate; 500K rungs are fractional passes.",
            "Reused development and one seed do not establish independent generalization.",
            "Historical partial pairs and null terminal labels retain their limitations.",
            "Timing across different jobs is not a causal architecture speed comparison."]}


def plot_review(report, output):
    """Plot four within-world questions, not a cross-world MAE leaderboard."""
    from PIL import Image, ImageDraw, ImageFont
    pairs = {p["candidate_id"]: p for p in report["pairs"]}
    prefix = "TC-gptrans-g1-"
    canvas = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()
    draw.text((45, 20), "Saved RML curves | seed 42 | no new model execution", fill="#182638", font=font(27))
    colors = ("#1266ad", "#ce572b", "#20846b", "#8956b1")

    def line(suffix, metric, side, label, gain=False):
        rows = pairs[prefix + suffix]["aligned_curve"]
        return label, [(r["optimizer_step"] / 1000,
            r[metric]["gain" if gain else side]) for r in rows if r["optimizer_step"] >= 10000]

    def panel(index, title, subtitle, limits, lines):
        ox, oy = 35 + (index % 2) * 795, 80 + (index // 2) * 505
        draw.text((ox, oy), title, fill="#182638", font=font(22))
        draw.text((ox, oy + 31), subtitle, fill="#526171", font=font(17))
        left, top, right, bottom = ox + 67, oy + 66, ox + 730, oy + 352
        low, high = limits
        observed = [y for _, values in lines for _, y in values]
        low, high = min(low, min(observed)), max(high, max(observed))
        pad = (high-low)*.03
        low, high = low-pad, high+pad
        def xy(x, y):
            return left + (x - 10) / 37 * (right-left), bottom - (y-low)/(high-low)*(bottom-top)
        for tick in range(5):
            y = low + (high-low)*tick/4
            py = xy(10, y)[1]
            draw.line((left, py, right, py), fill="#e0e5e9")
            draw.text((ox, py-9), f"{y:.3f}", fill="#526171", font=font(16))
        for x in (10, 20, 30, 40, 47):
            px = xy(x, low)[0]
            draw.text((px-10, bottom+7), str(x), fill="#526171", font=font(16))
        draw.line((left, top, left, bottom, right, bottom), fill="#526171", width=2)
        draw.text((left+180, bottom+31), "Optimizer steps (thousands)", fill="#526171", font=font(17))
        if low <= 0 <= high:
            py = xy(10, 0)[1]
            draw.line((left, py, right, py), fill="#78838b", width=2)
        for i, (label, values) in enumerate(lines):
            draw.line([xy(x, y) for x, y in values], fill=colors[i], width=3)
            ly = bottom + 64 + 23*i
            draw.line((left, ly+8, left+28, ly+8), fill=colors[i], width=3)
            draw.text((left+36, ly), label, fill="#28374a", font=font(17))

    panel(0, "A. EMA lag, not different live training", "Internal development MAE (eV); shared live trajectory", (.135, .25), [
        line("ema999-100k-s42", "ema_dev_metric", "reference", "EMA 0.9999"),
        line("ema999-100k-s42", "ema_dev_metric", "candidate", "EMA 0.999"),
        line("ema999-100k-s42", "live_dev_metric", "candidate", "Live weights")])
    panel(1, "B. Local-bond gain retained at this 500K budget", "Matched-control minus candidate MAE (eV); distinct worlds", (0, .014), [
        line("bond-local-100k-s42", "ema_dev_metric", "candidate", "100K vs its matched control", True),
        line("local-transfer-equal-updates-500k-s42", "ema_dev_metric", "candidate", "500K vs its matched control", True)])
    panel(2, "C. Fitting continues after dev gains flatten", "Triplet aggregate MAE (eV); online train is a different mode", (.065, .205), [
        line("local-triplet-aggregate-100k-s42", "live_train_metric", "candidate", "Online training"),
        line("local-triplet-aggregate-100k-s42", "live_dev_metric", "candidate", "Live development"),
        line("local-triplet-aggregate-100k-s42", "ema_dev_metric", "candidate", "EMA development")])
    panel(3, "D. Relation changes vs the same local parent", "Parent minus candidate MAE (eV)", (-.008, .014), [
        line("local-triplet-aggregate-100k-s42", "ema_dev_metric", "candidate", "Triplet aggregate", True),
        line("local-triplet-attention-100k-s42", "ema_dev_metric", "candidate", "Triplet attention", True),
        line("local-connected-pair-100k-s42", "ema_dev_metric", "candidate", "Connected pair FFN", True),
        line("local-bond-return-100k-s42", "ema_dev_metric", "candidate", "Same-block bond return", True)])
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    report = review_repository(args.repo_root)
    atomic_write(args.output, json_bytes(report))
    if args.figure:
        plot_review(report, args.figure)
    print({"records": report["record_counts"], "pairs": report["replay"]["candidate_pairs"],
           "output": str(args.output)})


if __name__ == "__main__":
    main()

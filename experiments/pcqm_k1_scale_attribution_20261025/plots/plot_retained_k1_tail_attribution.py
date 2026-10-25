"""Plot the retained K1 tail diagnostic means; performs no inference or statistics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RESULT = Path(r"D:\molgap-worktrees\k1-scale-attribution-20261025\experiments\pcqm_k1_scale_attribution_20261025\attempt_002\results\result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("k1_retained_tail_attribution.png"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("format") != "molgap-k1-retained-tail-diagnostic-v1":
        raise SystemExit("Expected the completed retained-tail diagnostic result")

    def mae(case, role):
        return float(result["cases"][case][role]["mae_eV"])

    states = (
        ("Initial parent pass 49", "initial"),
        ("Selected tail pass 15", "selected"),
        ("Final tail pass 20", "final"),
    )
    roles = (
        ("train", "TRAIN: 16,384 sampled rows\n(in-sample BN calibration cohort)"),
        ("development", "Development: 50,000 rows\n(previously selection-used)"),
    )
    arms = (
        ("wd1e3", "WD = 0.001", "#2864A0"),
        ("wd1e2", "WD = 0.01", "#D37225"),
    )
    bn_states = (("original", "original BN", "-"),
                 ("calibrated", "TRAIN-calibrated BN", "--"))
    xs = range(len(states))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.5))
    for ax, (role, title) in zip(axes, roles):
        for arm, arm_label, color in arms:
            for bn, bn_label, linestyle in bn_states:
                values = []
                for _, endpoint in states:
                    case = "initial" if endpoint == "initial" else f"tail_{arm}_{endpoint}"
                    values.append(mae(f"{case}_{bn}", role))
                ax.plot(xs, values, color=color, linestyle=linestyle,
                        marker="o", markersize=6, linewidth=2.2,
                        label=f"{arm_label}, {bn_label}")
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xticks(list(xs), [label for label, _ in states])
        ax.set_ylabel("MAE (eV; lower is better)")
        ax.grid(axis="y", color="#D8DEE8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9, pad=7)

    fig.suptitle("K1 retained tail attribution: fixed-cohort MAE",
                 fontsize=16, y=0.97, fontweight="semibold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.145), fontsize=9.5,
               columnspacing=2.2, handlelength=3.0)
    fig.text(0.5, 0.055,
             "X positions are named checkpoint states, not a shared epoch axis; both arms share the initial checkpoint. "
             "TRAIN is the in-sample calibration cohort; development was previously used for selection.",
             ha="center", va="center", fontsize=9, color="#475569", wrap=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.30, wspace=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"result={args.result}")
    print(f"output={args.output}")
    print(f"status={result['status']}; rows are fixed diagnostic cohorts; no statistics recomputed")


if __name__ == "__main__":
    main()

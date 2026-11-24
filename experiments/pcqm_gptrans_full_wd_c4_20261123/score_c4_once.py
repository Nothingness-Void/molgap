"""Score the C4 final-EMA weights on tail_holdout_v1 once.

Refuses to run if the result file or the prediction file already exists.
The scorer is platforms/local/pcqm_tail_holdout_eval.py, called as the lever-A
decision called it (--run-config, --initial-state, --weights-key model, --dev-range 0:50000).
Run with the repo venv python; nothing here uses worker processes.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RET = Path("D:/molgap-artifacts/retrieved/gptrans-c4-wd022-s43-20261123")
OUT_DIR = Path("D:/molgap-artifacts/tail_eval_c4_wd022_20261123")
OUT = OUT_DIR / "gptrans_c4_wd022_s43.json"
PREDS = OUT_DIR / "preds"
NAME = "gptrans_c4_wd022_s43"


def main():
    if OUT.exists() or PREDS.exists():
        raise SystemExit(f"STOP: {OUT} or {PREDS} exists. The protected tail is scored once.")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable, str(REPO / "platforms/local/pcqm_tail_holdout_eval.py"),
        "--raw-dir", "D:/molgap-data/ogb_pcqm4mv2_raw/extracted",
        "--data-root", "D:/molgap-data/pcqm_full_train_v1",
        "--run", f"{NAME}={RET / 'ema_model.pt'}:3250000",
        "--dev-range", "0:50000",
        "--run-config", str(RET / "run_config.json"),
        "--initial-state", str(RET / "gptrans_init_s43.pt"),
        "--weights-key", "model",
        "--save-predictions", str(PREDS),
        "--out", str(OUT),
    ]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()

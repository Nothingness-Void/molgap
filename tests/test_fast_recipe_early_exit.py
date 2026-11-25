import importlib.util
import sys
from pathlib import Path

IMS = Path(__file__).resolve().parents[1] / "platforms/ims"
sys.path.insert(0, str(IMS))
SPEC = importlib.util.spec_from_file_location("fast_recipe_run", IMS / "fast_recipe_run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def row(epoch, skips=0, pass_s=190.0, ema=0.1, loss=0.2):
    return {"epoch": epoch, "scaler_skips": skips, "pass_s": pass_s, "ema_dev_mae_eV": ema, "train_mae_eV": loss}


def test_early_exit_rules():
    ok = [row(1, 2, 440.0), row(2, 4), row(3, 6)]  # a slow compile pass 1 and a few skips are normal
    assert run.early_exit_reason(ok, 3906) is None
    assert "non-finite" in run.early_exit_reason(ok + [row(4, 6, ema=float("nan"))], 3906)
    assert "GradScaler" in run.early_exit_reason(ok + [row(4, 6 + 40)], 3906)  # 40 > 1% of 3906
    assert run.early_exit_reason(ok + [row(4, 6 + 39)], 3906) is None
    slow = [row(1, 0, 440.0), row(2, 0, 300.0), row(3, 0, 300.0)]
    assert run.early_exit_reason(slow, 3906) is None  # two slow passes are tolerated
    assert "in a row" in run.early_exit_reason(slow + [row(4, 0, 300.0)], 3906)
    assert run.early_exit_reason(slow + [row(4, 0, 200.0)], 3906) is None

"""Local checks for the P1 addon on the real driver and real rows (no tests directory; records land beside this file).

(a) identity of the 2D path: with P(2D) forced to 1 the P1 and zeros models train 20 steps and every control weight (live and
    EMA) is bit-equal to the control's; also control vs control (the determinism baseline). At step 0 the zero-initialised
    branch leaves 2D+3D outputs equal to the control, and evaluation hooks are inert.
(b) inference takes no coordinates: scoring with no coordinate field, and with the sidecar folders hidden, gives equal output.
(c) train-time draws repeat for equal inputs, honour the 0.40/0.20/0.40 split and the invalid-row rule; a run paused at step
    10 and resumed to 20 equals an uninterrupted 20-step run (the draws need no checkpoint state).

Usage: unit_checks.py --work DIR --fixed-root DIR --dev-sidecar DIR [--steps 20]. Single process, loader workers 0.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
C1 = HERE.parent / "pcqm_gptrans_dft_distance_aux_20261121"
sys.path.insert(0, str(REPO / "src"))
results = []


def record(name, passed, **facts):
    results.append({"check": name, "passed": bool(passed), **facts})
    print("PASS" if passed else "FAIL", name, json.dumps(facts, default=str)[:300], flush=True)


def train(work, arm, seed, steps, fixed_root, *, all_2d=False, tag=None):
    """Run the real IMS driver for `steps` optimizer steps (pauses at that step; a rerun resumes)."""
    out = work / (tag or f"{arm}_s{seed}_{steps}")
    code = ("import sys, runpy; sys.path.insert(0, 'src'); sys.path.insert(0, 'platforms/ims'); "
            + ("import molgap.gptrans_dft_input_p1 as m; m.PROBS = (1.0, 0.0); " if all_2d else "")
            + "sys.argv = sys.argv[1:]; runpy.run_path('platforms/ims/run_family_training.py', run_name='__main__')")
    init = C1 / "init" / ("s44.pt" if seed == 42 else "s43.pt")
    command = [sys.executable, "-u", "-c", code, "run_family_training.py", "--device", "0", "--recipe", str(HERE / f"{arm}_s{seed}" / "recipe.json"),
               "--initial-state", str(init), "--input-root", str(fixed_root), "--output", str(out), "--phase", "train",
               "--no-preflight-receipt", "--source-commit", "unit-check", "--cpus", "4", "--max-steps", str(steps), "--training-workers", "0"]
    done = subprocess.run(command, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(REPO / "src")},
                          capture_output=True, text=True)
    if done.returncode not in (0, 2):
        raise SystemExit(f"{arm} run failed: {done.stdout[-500:]} {done.stderr[-1500:]}")
    return out


def states(out):
    state = torch.load(out / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    return state["model"], state["ema"], state["optimizer_step"]


def shared_equal(control, other):
    """Every control key is present in `other` and bit-equal; returns the list of differing keys."""
    return [k for k, v in control.items() if k not in other or not torch.equal(v, other[k])]


def check_draws():
    from types import SimpleNamespace
    from molgap import gptrans_dft_input_p1 as p1
    size = 128
    def fake(rows, valid=True):
        atoms = size * 12
        return SimpleNamespace(source_idx=torch.tensor(rows), num_graphs=size, dft_pos=torch.zeros(atoms, 3),
                               dft_valid=torch.full((size,), valid, dtype=torch.bool))
    a, b = p1.draw_modes(42, fake(list(range(size)))), p1.draw_modes(42, fake(list(range(size))))
    c, d = p1.draw_modes(43, fake(list(range(size)))), p1.draw_modes(42, fake(list(range(1, size + 1))))
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    differ = not torch.equal(a[2], c[2]) and not torch.equal(a[2], d[2])
    use, only, share = 0, 0, 0
    batches = 200
    for k in range(batches):
        u, o, _ = p1.draw_modes(42, fake(list(range(k * size, (k + 1) * size))))
        use, only = use + int(u.sum()), only + int(o.sum())
    n = batches * size
    fractions = {"2d": 1 - use / n, "3d_only": only / n, "2d3d": (use - only) / n}
    invalid = p1.draw_modes(42, fake(list(range(size)), valid=False))
    record("(c) draws repeat for equal inputs, change with seed or rows, split ~0.40/0.20/0.40, invalid rows are 2D",
           same and differ and abs(fractions["2d"] - 0.4) < 0.02 and abs(fractions["3d_only"] - 0.2) < 0.02
           and abs(fractions["2d3d"] - 0.4) < 0.02 and not bool(invalid[0].any()), fractions=fractions)


def check_inputs_identity(fixed_root, work, dev_sidecar):
    """(a) model-level identity at init and (b) inference without coordinates / sidecar."""
    from copy import copy
    from molgap.family_training_kernel import load_family_model
    from molgap.family_training_runner import read_recipe
    from molgap.gptrans_dft_input_p1 import clear_inputs, set_inputs
    from molgap.gptrans_execution_backend import prepare_layout, prepared_forward
    from molgap.gptrans_execution_hotpath import MODEL_FIELDS
    from molgap.gptrans_dft_distance_aux import EXTRA_FIELDS
    sys.path.insert(0, str(C1))
    from score_dev_and_distance import shard_graphs
    sys.path.insert(0, str(HERE))
    import score_p1
    device = torch.device("cuda")
    graphs = shard_graphs(fixed_root, 10)[:256]
    graphs3d = score_p1.with_geometry(graphs, dev_sidecar)
    init = C1 / "init" / "s44.pt"
    built = {}
    for arm in ("ctl", "p1", "zer"):
        _, recipe = read_recipe(HERE / f"{arm}_s42" / "recipe.json")
        built[arm] = load_family_model(recipe, init)[0].to(device).eval()
    mean, std = recipe.target_mean_eV, recipe.target_std_eV
    base = score_p1.predict(built["ctl"], graphs, mean, std, device, "2d")
    out2d = {arm: score_p1.predict(built[arm], graphs, mean, std, device, "2d") for arm in ("p1", "zer")}
    record("(a) at init, 2D forward of P1 and zeros equals the control (torch.equal, 256 dev rows)",
           all(torch.equal(base, out2d[a]) for a in out2d) and shared_equal(built["ctl"].state_dict(), built["p1"].state_dict()) == [])
    both = {arm: score_p1.predict(built[arm], graphs3d, mean, std, device, "2d3d") for arm in ("p1", "zer")}
    only = score_p1.predict(built["p1"], graphs3d, mean, std, device, "3d")
    record("(a) at init the zero-initialised branch adds nothing: 2D+3D equals 2D; 3D-only differs (SPD and bond masked)",
           all(torch.equal(base, both[a]) for a in both) and not torch.equal(base, only))
    hidden = []
    try:  # (b) hide every sidecar folder, then score without coordinates
        for path in (Path(fixed_root) / "sidecars", Path(dev_sidecar)):
            target = path.with_name(path.name + "_hidden")
            path.rename(target)
            hidden.append((path, target))
        again = score_p1.predict(built["p1"], graphs, mean, std, device, "2d")
    finally:
        for path, target in hidden:
            target.rename(path)
    record("(b) 2D scoring needs no coordinates and no sidecar: identical output with every sidecar folder hidden",
           torch.equal(again, out2d["p1"]) and len(hidden) == 2 and all(getattr(built["p1"], "_molgap_addons")["dft_input"].state is None for _ in [0]))
    return built, graphs3d, device, (mean, std)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--dev-sidecar", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    check_draws()
    check_inputs_identity(args.fixed_root, args.work, args.dev_sidecar)
    if not args.skip_train:
        n = args.steps
        ctl_a, ctl_b = train(args.work, "ctl", 42, n, args.fixed_root, tag="ctl_a"), train(args.work, "ctl", 42, n, args.fixed_root, tag="ctl_b")
        (cm, ce, step), (cm2, ce2, _) = states(ctl_a), states(ctl_b)
        record(f"(a) determinism baseline: two control runs of {n} steps are bit-equal (live and EMA)",
               step == n and shared_equal(cm, cm2) == [] and shared_equal(ce, ce2) == [])
        for arm in ("p1", "zer"):
            out = train(args.work, arm, 42, n, args.fixed_root, all_2d=True, tag=f"{arm}_all2d")
            pm, pe, _ = states(out)
            record(f"(a) {arm} with P(2D)=1: all shared weights bit-equal to the control after {n} steps (live and EMA)",
                   shared_equal(cm, pm) == [] and shared_equal(ce, pe) == [], differing_live=shared_equal(cm, pm)[:5],
                   extra_keys=len([k for k in pm if k not in cm]))
        whole = train(args.work, "p1", 42, n, args.fixed_root, tag="p1_whole")
        half = train(args.work, "p1", 42, n // 2, args.fixed_root, tag="p1_resume")
        half = train(args.work, "p1", 42, n, args.fixed_root, tag="p1_resume")
        (wm, we, _), (hm, he, hstep) = states(whole), states(half)
        record(f"(c) P1 paused at step {n // 2} and resumed to {n} equals the uninterrupted run (all keys, live and EMA)",
               hstep == n and shared_equal(wm, hm) == [] and shared_equal(we, he) == [] and set(wm) == set(hm))
        differs = shared_equal(cm, wm)
        record("sanity: with the real mode schedule P1 differs from the control (the branch is live)", len(differs) > 0, differing_keys=len(differs))
    path = HERE / "unit_checks_record.json"
    path.write_text(json.dumps({"date": "2026-11-23", "results": results}, indent=1) + "\n", encoding="utf-8")
    failed = [r["check"] for r in results if not r["passed"]]
    print("FAILED:" if failed else "all passed", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

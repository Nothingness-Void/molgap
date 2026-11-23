"""Marginal local step time per arm: wall of a long and a short real driver run, (long - short) / steps (run with -I).

Startup, data load and checkpoint costs cancel in the difference. RTX 5060, FP32, loader workers 0, same sampler and seed for
every arm. The A100 trace (train_log.jsonl, see protocol.md gate (iv)) is the number that counts; this ratio is advisory.
Usage: step_time_probe.py --work DIR --fixed-root DIR [--short 50] [--long 350] [--rounds 2]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unit_checks  # noqa: E402 (reuses the driver launcher)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--short", type=int, default=50)
    parser.add_argument("--long", type=int, default=350)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args(argv)
    per_step = {arm: [] for arm in ("ctl", "p1", "zer")}
    for rnd in range(args.rounds):
        for arm in per_step:
            wall = {}
            for steps in (args.short, args.long):
                began = time.perf_counter()
                unit_checks.train(args.work, arm, 42, steps, args.fixed_root, tag=f"{arm}_t{steps}_r{rnd}")
                wall[steps] = time.perf_counter() - began
            per_step[arm].append((wall[args.long] - wall[args.short]) / (args.long - args.short))
            print(arm, rnd, wall, per_step[arm][-1], flush=True)
    median = {arm: statistics.median(v) for arm, v in per_step.items()}
    record = {"date": "2026-11-23", "device": "RTX 5060 FP32, loader workers 0", "short_steps": args.short, "long_steps": args.long,
              "s_per_step_by_round": per_step, "median_s_per_step": median,
              "ratio_p1_over_control": median["p1"] / median["ctl"], "ratio_zeros_over_control": median["zer"] / median["ctl"],
              "note": "advisory; gate (iv) is read from the A100 train_log.jsonl"}
    (HERE / "step_time_record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()

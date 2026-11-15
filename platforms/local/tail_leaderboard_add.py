"""Merge tail_holdout_v1 results from pcqm_tail_holdout_eval.py JSON files into the master leaderboard.

Entries are [name, {mae_ev, mae_mev, train_rows, checkpoint_sha256}] in "all" and in the "top_<rows/1000>k" bucket of their
train-row count, each list sorted by mae_ev ascending. An entry already present with the same name and checkpoint is kept; the
same name with another checkpoint stops, since every endpoint is scored once.
Usage: python tail_leaderboard_add.py LEADERBOARD.json EVAL.json [EVAL.json ...]
"""
import json, sys
from pathlib import Path


def main(argv=None):
    board_path, *evals = argv if argv is not None else sys.argv[1:]
    board = json.loads(Path(board_path).read_text(encoding="utf-8"))
    known = {name: entry["checkpoint_sha256"] for name, entry in board["all"]}
    for path in evals:
        evidence = json.loads(Path(path).read_text(encoding="utf-8"))
        assert evidence["role"] == "tail_holdout_v1", f"{path} is not a tail_holdout_v1 evaluation"
        for name, result in evidence["results"].items():
            sha = result["checkpoint_sha256"]
            if name in known:
                assert known[name] == sha, f"STOP: {name} is on the board with checkpoint {known[name]}, not {sha}"
                continue
            mae = result["tail_holdout_v1"]["mae_ev"]
            entry = [name, {"mae_ev": mae, "mae_mev": round(mae * 1000, 3), "train_rows": result["train_rows"], "checkpoint_sha256": sha}]
            board["all"].append(entry)
            board.setdefault(f"top_{result['train_rows'] // 1000}k", []).append(entry)
            known[name] = sha
            print(f"added {name}: {mae * 1000:.3f} meV")
    for key, value in board.items():
        if isinstance(value, list):
            value.sort(key=lambda e: e[1]["mae_ev"])
    board["total_count"] = len(board["all"])
    Path(board_path).write_text(json.dumps(board, indent=2), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

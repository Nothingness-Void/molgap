"""Read-only live Kaggle SSE log & training-progress inspector for configured accounts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import requests

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "src")]
from check_quota import DEFAULT_ACCOUNTS, DEFAULT_KEY_DIR  # noqa: E402
from molgap.kaggle_submission import connect_kaggle  # noqa: E402

ARTIFACTS_RUNS_DIR = Path(os.environ.get("MOLGAP_ARTIFACT_ROOT", "D:/molgap-artifacts")) / "runs"
STREAM_URL = "https://www.kaggle.com/api/v1/kernels/logs/stream/"


def _load_accounts(key_dir: Path, cred_overrides: list[str] | None) -> dict[str, tuple[Path, dict]]:
    accounts: dict[str, tuple[Path, dict]] = {}
    candidates = [Path(p.strip()) for p in (cred_overrides or ())]
    if not candidates:
        candidates = [key_dir / rel for _, rel in DEFAULT_ACCOUNTS]
    for path in candidates:
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        user = raw.get("username")
        if isinstance(user, str) and user and isinstance(raw.get("key"), str):
            accounts[user] = (path, raw)
    return accounts


def _resolve_target(target: str, attempt: str, runs_dir: Path) -> tuple[str, str]:
    if "/" in target and not target.endswith("submission_state.json"):
        return target, target
    journal = Path(target)
    if not journal.is_file():
        journal = runs_dir / target / attempt / "operations" / "submission_state.json"
    if not journal.is_file():
        raise FileNotFoundError(f"Cannot resolve kernel ref or journal for {target!r} (attempt={attempt})")
    state = json.loads(journal.read_text(encoding="utf-8"))
    ref = state.get("binding", {}).get("kernel") or state.get("kernel_response", {}).get("kernel")
    if not ref or "/" not in ref:
        raise ValueError(f"No bound kernel ref in {journal}")
    return ref, journal.parents[2].name


def _discover_recent_bound(runs_dir: Path, limit: int) -> list[tuple[str, str]]:
    if not runs_dir.is_dir():
        return []
    journals = sorted(runs_dir.glob("*/*/operations/submission_state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen: set[str] = set()
    targets: list[tuple[str, str]] = []
    for journal in journals:
        try:
            state = json.loads(journal.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("phase") != "BOUND":
            continue
        ref = state.get("binding", {}).get("kernel") or state.get("kernel_response", {}).get("kernel")
        if not ref or "/" not in ref or ref in seen:
            continue
        seen.add(ref)
        targets.append((ref, journal.parents[2].name))
        if len(targets) >= limit:
            break
    return targets


def _stream_sse_lines(username: str, key: str, ref: str, *, read_timeout: float) -> tuple[list[str], str]:
    lines: list[str] = []
    note = "eof"
    session = requests.Session()
    session.auth = (username, key)
    session.headers["User-Agent"] = "kaggle-api/molgap"
    try:
        with session.get(f"{STREAM_URL}{ref}", headers={"Accept": "text/event-stream, */*"},
                         stream=True, timeout=(6, read_timeout)) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(chunk_size=1024, decode_unicode=True):
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "END_OF_LOG":
                    note = "end_of_log"
                    break
                text = json.loads(payload).get("data", "") if payload.startswith("{") else payload
                for ln in text.splitlines():
                    if ln.strip():
                        lines.append(ln)
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
        note = "live_tail_timeout"
    except Exception as exc:
        note = f"stream_error:{type(exc).__name__}"
    return lines, note


def _output_log_lines(cred_path: Path, ref: str) -> list[str]:
    api = connect_kaggle(cred_path)
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        api.kernels_output(ref, path=td, quiet=True)
        for log_file in sorted(Path(td).glob("*.log")):
            raw_text = log_file.read_text(encoding="utf-8", errors="replace")
            for ln in raw_text.splitlines():
                if not ln.strip():
                    continue
                if ln.lstrip().startswith((",{", "[{", "{")):
                    try:
                        obj = json.loads(ln.lstrip(", "))
                        if isinstance(obj, dict) and "data" in obj:
                            lines.extend(x for x in str(obj["data"]).splitlines() if x.strip())
                            continue
                    except Exception:
                        pass
                lines.append(ln)
    return lines


def _parse_training_lines(lines: list[str]) -> dict:
    phase = "init"
    arms: dict[str, dict] = {}
    errors: list[str] = []
    for ln in lines:
        if ln.startswith("PAIR_PHASE_"):
            phase = ln.strip()
        elif "TRAIN_LOG " in ln:
            try:
                rec = json.loads(ln.split("TRAIN_LOG ", 1)[1])
            except Exception:
                continue
            arm_id = str(rec.get("arm_id") or "arm")
            p_num = int(rec.get("completed_passes") or 0)
            entry = arms.setdefault(arm_id, {"arm_id": arm_id, "history": {}, "latest": rec})
            entry["latest"] = rec
            if "development_mae_eV" in rec:
                entry["history"][p_num] = float(rec["development_mae_eV"])
        elif any(tok in ln for tok in ("Error:", "Exception:", "Traceback (most recent call last)")):
            errors.append(ln.strip())
    return {"phase": phase, "arms": list(arms.values()), "error_lines": errors[-8:]}


def _fmt_duration(sec: float | None) -> str:
    if sec is None or sec < 0:
        return "-"
    m, _ = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _query_scheduler(ref: str, accounts: dict[str, tuple[Path, dict]]) -> tuple[str, str]:
    owner = ref.split("/", 1)[0]
    if owner not in accounts:
        return "NO_CREDENTIAL", ""
    api = connect_kaggle(accounts[owner][0])
    st = api.kernels_status(ref)
    return str(getattr(st, "status", "UNKNOWN")).split(".")[-1], str(getattr(st, "failure_message", "") or "")


def inspect_kernel(ref: str, label: str, accounts: dict[str, tuple[Path, dict]],
                   *, read_timeout: float, sched: str | None = None, fail_msg: str = "") -> dict:
    owner = ref.split("/", 1)[0]
    if owner not in accounts:
        return {"run": label, "kernel": ref, "account": owner, "scheduler": "NO_CREDENTIAL", "arms": [], "lines": []}
    cred_path, cred_raw = accounts[owner]
    if sched is None:
        sched, fail_msg = _query_scheduler(ref, accounts)
    lines, stream_note = _stream_sse_lines(cred_raw["username"], cred_raw["key"], ref, read_timeout=read_timeout)
    if not lines and sched == "ERROR":
        try:
            lines = _output_log_lines(cred_path, ref)
            if lines:
                stream_note = "kernels_output"
        except Exception:
            pass
    parsed = _parse_training_lines(lines)
    return {
        "run": label, "kernel": ref, "account": owner, "scheduler": sched, "failure_message": fail_msg,
        "stream": stream_note, "line_count": len(lines), **parsed, "lines": lines,
    }


def _format_summary(records: list[dict]) -> str:
    headers = ("Account/Run", "Sched", "Phase/Arm", "Pass", "Best meV", "Last meV", "Train meV", "s/pass", "Elapsed/ETA", "Same-Pass Δ")
    rows: list[tuple[str, ...]] = []
    for rec in records:
        arms = rec.get("arms") or []
        ref_hist = arms[0]["history"] if len(arms) >= 2 else {}
        if not arms:
            rows.append((f"{rec['account']} | {rec['run']}", rec["scheduler"], rec.get("phase", "-"),
                         "-", "-", "-", "-", "-", "-", rec.get("failure_message") or rec.get("stream", "-")))
            continue
        for idx, arm in enumerate(arms):
            lt = arm["latest"]
            p = int(lt.get("completed_passes") or 0)
            p_tot = int(round(float(lt.get("planned_passes") or 0)))
            best = float(lt["best_development_mae_eV"]) * 1000.0 if "best_development_mae_eV" in lt else None
            last = float(lt["development_mae_eV"]) * 1000.0 if "development_mae_eV" in lt else None
            trn = float(lt["train_live_mae_eV"]) * 1000.0 if "train_live_mae_eV" in lt else None
            pw = float(lt["pass_wall_seconds"]) if "pass_wall_seconds" in lt else None
            el = _fmt_duration(lt.get("elapsed_seconds"))
            eta = _fmt_duration(lt.get("eta_seconds"))
            delta = "-"
            if idx > 0 and p in ref_hist and last is not None:
                delta = f"{last - ref_hist[p] * 1000.0:+.2f} (@P{p})"
            elif idx > 0 and arms[0]["latest"].get("best_development_mae_eV") and best is not None:
                delta = f"{best - float(arms[0]['latest']['best_development_mae_eV']) * 1000.0:+.2f} (best)"
            rows.append((
                f"{rec['account']} | {rec['run']}" if idx == 0 else "  '",
                f"{rec['scheduler']} ({lt.get('status', '')})" if idx == 0 else str(lt.get("status", "")),
                arm["arm_id"],
                f"{p}/{p_tot or '?'}",
                f"{best:.3f}" if best is not None else "-",
                f"{last:.3f}" if last is not None else "-",
                f"{trn:.3f}" if trn is not None else "-",
                f"{pw:.0f}" if pw is not None else "-",
                f"{el}/{eta}",
                delta,
            ))
    widths = [max(len(str(r[i])) for r in (headers, *rows)) for i in range(len(headers))]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = lambda r: "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([sep, fmt(headers), sep, *(fmt(r) for r in rows), sep])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="Kernel ref (<owner>/<slug>), logical_run_id, or submission_state.json")
    parser.add_argument("--run-id", action="append", default=[], help="Logical run ID under D:/molgap-artifacts/runs")
    parser.add_argument("--attempt-id", default="v1", help="Attempt subdirectory (default: v1)")
    parser.add_argument("--scan-limit", type=int, default=14, help="Recent bound runs to check when no targets given")
    parser.add_argument("--all", action="store_true", help="Include COMPLETE/ERROR runs when auto-discovering")
    parser.add_argument("--tail", type=int, default=0, help="Print last N raw log lines per kernel")
    parser.add_argument("--read-timeout", type=float, default=2.5, help="SSE tail timeout seconds (default: 2.5)")
    parser.add_argument("--key-dir", type=Path, default=DEFAULT_KEY_DIR, help="Directory holding Kaggle keys")
    parser.add_argument("--credentials", action="append", help="Optional path/to/kaggle.json override(s)")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    args = parser.parse_args(argv)

    accounts = _load_accounts(args.key_dir, args.credentials)
    explicit = [*args.targets, *args.run_id]
    targets = ([_resolve_target(t, args.attempt_id, ARTIFACTS_RUNS_DIR) for t in explicit]
               if explicit else _discover_recent_bound(ARTIFACTS_RUNS_DIR, args.scan_limit))

    statuses = [(ref, label, *_query_scheduler(ref, accounts)) for ref, label in targets]
    selected = statuses if (explicit or args.all) else [item for item in statuses if item[2] in {"RUNNING", "QUEUED"}]
    if not selected:
        selected = statuses[:4]

    with ThreadPoolExecutor(max_workers=max(1, min(6, len(selected)))) as pool:
        records = list(pool.map(
            lambda item: inspect_kernel(item[0], item[1], accounts, read_timeout=args.read_timeout, sched=item[2], fail_msg=item[3]),
            selected,
        ))

    if args.json:
        out = [{k: v for k, v in r.items() if k != "lines"} | ({"tail_lines": r["lines"][-args.tail:]} if args.tail else {}) for r in records]
        print(json.dumps({"observed_at": datetime.now().astimezone().isoformat(), "kernels": out}, ensure_ascii=False, indent=2))
    else:
        print(_format_summary(records))
        for rec in records:
            tail_n = args.tail or (15 if rec["scheduler"] == "ERROR" else 0)
            if tail_n and rec.get("lines"):
                print(f"\n=== {rec['kernel']} (last {min(tail_n, len(rec['lines']))} lines) ===")
                for ln in rec["lines"][-tail_n:]:
                    print("  ", ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

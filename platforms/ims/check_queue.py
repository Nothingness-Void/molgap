"""Read-only IMS/RCCS queue, quota, and chou V4 training dashboard/CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

JST = timezone(timedelta(hours=9))
REMOTE_BOUNDARY = "/lustre/home/users/sm2/chou"
DEFAULT_HOST = "ccfep.ims.ac.jp"
DEFAULT_SSH_PORT = 22
DEFAULT_USER = "sm2"
DEFAULT_WEB_PORT = 8765
DEFAULT_MIN_INTERVAL = 15.0
DEFAULT_KEY_DIR = Path(os.environ.get("MOLGAP_KEY_DIR", "D:/\u4e0b\u8f7d/Key"))

_SECTIONS = (
    "JOBINFO_C",
    "JOBINFO_M",
    "JOBINFO_W",
    "JOBINFO_S",
    "SHOWLIM_C",
    "CHOU_RUNS",
)

_REMOTE_SCRIPT = r"""set -u
BOUNDARY="$(realpath -e -- "__BOUNDARY__")" || exit 2
case "$BOUNDARY" in /lustre/home/users/sm2/chou|/lustre/home/users/sm2/chou/*) ;; *) exit 2;; esac
cd "$BOUNDARY" 2>/dev/null || exit 2
echo "===__SEC_JOBINFO_C__==="
jobinfo -c 2>&1
printf "\n===__RC_JOBINFO_C__%s===\n" "$?"
echo "===__SEC_JOBINFO_M__==="
jobinfo -m 2>&1
printf "\n===__RC_JOBINFO_M__%s===\n" "$?"
echo "===__SEC_JOBINFO_W__==="
jobinfo -w 2>&1
printf "\n===__RC_JOBINFO_W__%s===\n" "$?"
echo "===__SEC_JOBINFO_S__==="
jobinfo -s 2>&1
printf "\n===__RC_JOBINFO_S__%s===\n" "$?"
echo "===__SEC_SHOWLIM_C__==="
showlim -c 2>&1
printf "\n===__RC_SHOWLIM_C__%s===\n" "$?"
echo "===__SEC_CHOU_RUNS__==="
PY="$(command -v python3)" || exit 2
"$PY" - <<'__CHOU_PY__' 2>&1
import json
import os
import shlex

BOUNDARY = os.getcwd()
RUNS_ROOT = os.path.join(BOUNDARY, "molgap-v4-submit", "runs")

def safe_real(path):
    real = os.path.realpath(path)
    if real != BOUNDARY and not real.startswith(BOUNDARY + os.sep):
        raise ValueError("Outside boundary: " + real)
    return real

def read_json(path, default=None):
    try:
        real = safe_real(path)
        if not os.path.isfile(real):
            return default
        with open(real, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default

def file_mtime_utc(path):
    try:
        real = safe_real(path)
        if not os.path.exists(real):
            return None
        return int(os.path.getmtime(real))
    except Exception:
        return None

def tail_jsonl(path, limit=200):
    try:
        real = safe_real(path)
        if not os.path.isfile(real):
            return []
        with open(real, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 262144))
            if size > 262144:
                fh.readline()
            lines = fh.read().splitlines()[-limit:]
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out
    except FileNotFoundError:
        return []

def build_item_payload(log_rows, cm, status, total_passes, extra=None, cm_path=None, log_path=None):
    cm_summary = None
    if isinstance(cm, dict):
        cm_summary = {
            k: cm.get(k)
            for k in ("status", "arm", "completed_passes", "optimizer_steps", "sample_presentations", "selection", "wall_s", "peak_rss_bytes")
            if cm.get(k) is not None
        }
    payload = dict(extra or {})
    payload.update({
        "status": status,
        "total_passes": total_passes,
        "first_log": log_rows[0] if log_rows else None,
        "last_log": log_rows[-1] if log_rows else None,
        "log_tail": log_rows[-15:],
        "cm_mtime_utc": file_mtime_utc(cm_path) if cm_path else None,
        "log_mtime_utc": file_mtime_utc(log_path) if log_path else None,
        "completion_manifest": cm_summary,
        "curve": [
            {k: r.get(k) for k in ("epoch", "elapsed_s", "train_mae_eV", "dev_mae_eV", "ema_dev_mae_eV") if r.get(k) is not None}
            for r in log_rows
        ],
    })
    return payload

def recipe_total_passes(run_dir, item_name, serial_spec):
    candidates = []
    if isinstance(serial_spec, dict):
        for entry in serial_spec.get("items", []):
            if entry.get("name") == item_name or entry.get("id") == item_name:
                ref = entry.get("recipe")
                if ref:
                    candidates.append(os.path.join(run_dir, "code", ref))
    candidates.append(os.path.join(run_dir, "output", "training", item_name, "recipe.json"))
    candidates.append(os.path.join(run_dir, "output", "training", "recipe.json"))
    for p in candidates:
        data = read_json(p)
        if isinstance(data, dict):
            acc = data.get("acceptance_requirements") or {}
            sched = data.get("schedule") or {}
            passes = acc.get("completed_passes") or sched.get("total_units")
            if isinstance(passes, int) and passes > 0:
                return passes
    return None

runs = []
if os.path.isdir(safe_real(RUNS_ROOT)):
    dirs = []
    for name in os.listdir(safe_real(RUNS_ROOT)):
        full = os.path.join(RUNS_ROOT, name)
        try:
            real = safe_real(full)
        except Exception:
            continue
        if os.path.isdir(real):
            dirs.append((os.path.getmtime(real), name, real))
    dirs.sort(reverse=True)
    for mtime, name, run_dir in dirs[:12]:
        sub = read_json(os.path.join(run_dir, "submission.json")) or {}
        staging = read_json(os.path.join(run_dir, "staging_manifest.json")) or {}
        run_spec = read_json(os.path.join(run_dir, "run_spec.json")) or {}
        stage_train = read_json(os.path.join(run_dir, "output", "train", "stage_status.json")) or {}
        stage_pre = read_json(os.path.join(run_dir, "output", "preflight", "stage_status.json")) or {}
        serial_summary = read_json(os.path.join(run_dir, "output", "training", "serial_summary.json"))

        serial_spec = None
        cmd = stage_train.get("command") or []
        if isinstance(cmd, list) and "--spec" in cmd:
            idx = cmd.index("--spec")
            if idx + 1 < len(cmd):
                serial_spec = read_json(cmd[idx + 1])

        items = {}
        training_dir = os.path.join(run_dir, "output", "training")
        if serial_summary and isinstance(serial_summary.get("items"), dict):
            for item_id, info in serial_summary["items"].items():
                log_path = os.path.join(training_dir, item_id, "train_log.jsonl")
                cm_path = os.path.join(training_dir, item_id, "completion_manifest.json")
                log_rows = tail_jsonl(log_path, limit=200)
                cm = read_json(cm_path)
                total_passes = (cm or {}).get("completed_passes") or recipe_total_passes(run_dir, item_id, serial_spec)
                items[item_id] = build_item_payload(
                    log_rows, cm, info.get("status"), total_passes,
                    extra=info, cm_path=cm_path, log_path=log_path,
                )
        else:
            log_path = os.path.join(training_dir, "train_log.jsonl")
            cm_path = os.path.join(training_dir, "completion_manifest.json")
            log_rows = tail_jsonl(log_path, limit=200)
            cm = read_json(cm_path)
            if log_rows or cm:
                model_id = run_spec.get("model_id") or name
                status = "COMPLETE" if cm and cm.get("status") == "COMPLETE" else (stage_train.get("status") or "UNKNOWN").upper()
                total_passes = (cm or {}).get("completed_passes") or recipe_total_passes(run_dir, model_id, None)
                items[model_id] = build_item_payload(
                    log_rows, cm, status, total_passes,
                    cm_path=cm_path, log_path=log_path,
                )

        runs.append({
            "run_id": sub.get("run_id") or name,
            "mtime_utc": int(mtime),
            "mode": sub.get("mode") or run_spec.get("mode") or "single",
            "submission_status": sub.get("status"),
            "training_job": sub.get("training_job"),
            "resume_jobs": sub.get("resume_jobs"),
            "preflight_job": sub.get("preflight_job"),
            "resources": sub.get("resources") or run_spec.get("resources") or (serial_spec or {}).get("resources"),
            "submission_mtime_utc": file_mtime_utc(os.path.join(run_dir, "submission.json")),
            "staging_manifest_mtime_utc": file_mtime_utc(os.path.join(run_dir, "staging_manifest.json")),
            "submitted_at": sub.get("submitted_at") or staging.get("created_at") or run_spec.get("created_at"),
            "stage_train_status": stage_train.get("status"),
            "stage_train_started_at": stage_train.get("started_at"),
            "stage_train_finished_at": stage_train.get("finished_at"),
            "stage_train_mtime_utc": file_mtime_utc(os.path.join(run_dir, "output", "train", "stage_status.json")),
            "stage_preflight_status": stage_pre.get("status"),
            "model_id": run_spec.get("model_id"),
            "dataset_id": run_spec.get("dataset_id") or run_spec.get("dataset_ids"),
            "serial_order": (serial_summary or {}).get("order"),
            "serial_wall_s": (serial_summary or {}).get("wall_s"),
            "serial_reserve_s": (serial_summary or {}).get("reserve_s"),
            "serial_driver_starts": (serial_summary or {}).get("driver_starts"),
            "items": items,
        })

print(json.dumps({"runs": runs}, ensure_ascii=False))
__CHOU_PY__
printf "\n===__RC_CHOU_RUNS__%s===\n" "$?"
"""


def resolve_ssh_key(key_arg: Path | None) -> Path:
    candidates: list[Path] = []
    if key_arg is not None:
        candidates.append(key_arg)
    env_key = os.environ.get("MOLGAP_IMS_SSH_KEY")
    if env_key:
        candidates.append(Path(env_key))
    candidates.append(DEFAULT_KEY_DIR / "ccfep2023")
    for path in candidates:
        expanded = path.expanduser()
        if expanded.is_file():
            return expanded.resolve()
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"IMS SSH key not found. Checked: {checked}. Pass --key or set MOLGAP_IMS_SSH_KEY.")


def _to_int(text: str | None) -> int | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if not cleaned or cleaned in {"-", "-----", "n/a"}:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _split_sections(stdout: str) -> dict[str, str]:
    markers = {f"===__SEC_{name}__===": name for name in _SECTIONS}
    sections = {name: [] for name in _SECTIONS}
    current: str | None = None
    for line in stdout.splitlines():
        key = markers.get(line.strip())
        if key is not None:
            current = key
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _strip_job_id(job_id: str | None) -> str | None:
    if not job_id:
        return None
    return job_id.split(".", 1)[0].strip()


def _is_chou_workdir(workdir: str | None, remote_root: str) -> bool:
    if not workdir:
        return False
    norm = workdir.strip().strip('"').strip("'").rstrip("/")
    bases = {remote_root.rstrip("/")}
    if remote_root.startswith("/lustre/home/"):
        bases.add(remote_root[len("/lustre"):].rstrip("/"))
    return any(norm == b or norm.startswith(b + "/") for b in bases)


def _extract_owner(workdir: str | None, *, user: str | None = None, remote_root: str = REMOTE_BOUNDARY) -> str:
    """Derive personal directory owner from jobinfo -w Workdir without probing other paths."""
    if workdir:
        norm = workdir.strip().strip('"').strip("'").rstrip("/")
        parts = [p for p in norm.split("/") if p]
        user_tok = (user or "").strip()
        if user_tok and user_tok in parts:
            idx = parts.index(user_tok)
            if idx + 1 < len(parts):
                return parts[idx + 1]
        if "users" in parts:
            idx = parts.index("users")
            if idx + 2 < len(parts):
                return parts[idx + 2]
        if _is_chou_workdir(norm, remote_root):
            return "chou"
    return (user or "sm2").strip() or "sm2"


def parse_jobinfo_cmw(c_text: str, m_text: str, w_text: str, *, remote_root: str = REMOTE_BOUNDARY) -> tuple[list[dict], str | None]:
    jobs: dict[str, dict] = {}
    scheduler_ts: str | None = None

    def ensure(jid: str) -> dict:
        return jobs.setdefault(jid, {
            "job_id": jid,
            "queue": None,
            "queue_type": None,
            "name": None,
            "status": None,
            "cpus": None,
            "gpus": 0,
            "user": None,
            "group": None,
            "elapsed": None,
            "node_or_reason": None,
            "used_mem_mb": None,
            "lim_mem_mb": None,
            "workdir": None,
            "owner": None,
            "is_chou": False,
        })

    def is_noise(line: str) -> bool:
        s = line.strip()
        if not s or set(s) <= {"-"}:
            return True
        if s.startswith("Queue ") or s.startswith("Job Status at") or s.startswith("For the latest status"):
            return True
        return False

    for line in c_text.splitlines():
        m_ts = re.search(r"Job Status at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m_ts:
            scheduler_ts = m_ts.group(1)
        if is_noise(line):
            continue
        m = re.match(r"^\s*(\S+)\s+(\d+(?:\.\S+)?)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s*(.*)$", line)
        if not m:
            continue
        queue, raw_id, name, status, cpus, user_grp, elapsed, node_reason = m.groups()
        jid = _strip_job_id(raw_id) or raw_id
        row = ensure(jid)
        user, _, grp = user_grp.partition("/")
        row.update(
            queue=queue,
            name=name.lstrip("*") if row["name"] is None else row["name"],
            status=status,
            cpus=int(cpus),
            user=user,
            group=None if grp in {"---", "-", ""} else grp,
            elapsed=None if elapsed in {"-", "---"} else elapsed,
            node_or_reason=node_reason.strip() or None,
        )

    for line in m_text.splitlines():
        m_ts = re.search(r"Job Status at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m_ts:
            scheduler_ts = m_ts.group(1)
        if is_noise(line):
            continue
        m = re.match(r"^\s*(\S+)\s+(\d+(?:\.\S+)?)\s+(\S+)\s+(\S+)\s+(\d+(?:\+\d+)?)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
        if not m:
            continue
        qtype, raw_id, name, status, res_tok, user_grp, used_tok, lim_tok = m.groups()
        jid = _strip_job_id(raw_id) or raw_id
        row = ensure(jid)
        cpu_s, _, gpu_s = res_tok.partition("+")
        user, _, grp = user_grp.partition("/")
        row.update(
            queue_type=qtype,
            status=status,
            cpus=int(cpu_s),
            gpus=int(gpu_s) if gpu_s else 0,
            user=user,
            group=None if grp in {"---", "-", ""} else grp,
            used_mem_mb=_to_int(used_tok),
            lim_mem_mb=_to_int(lim_tok),
        )
        if row["name"] is None:
            row["name"] = name.lstrip("*")

    for line in w_text.splitlines():
        m_ts = re.search(r"Job Status at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m_ts:
            scheduler_ts = m_ts.group(1)
        if is_noise(line):
            continue
        m = re.match(r'^\s*(\S+)\s+(\d+(?:\.\S+)?)\s+(\S+)\s+(\S+)\s+(".*?"|\S+)\s*$', line)
        if not m:
            continue
        queue, raw_id, name, status, wd = m.groups()
        jid = _strip_job_id(raw_id) or raw_id
        row = ensure(jid)
        clean_wd = wd.strip().strip('"')
        row.update(
            queue=row["queue"] or queue,
            name=name,
            status=status,
            workdir=clean_wd,
            owner=_extract_owner(clean_wd, user=row["user"], remote_root=remote_root),
            is_chou=_is_chou_workdir(clean_wd, remote_root),
        )

    for r in jobs.values():
        if not r.get("owner"):
            r["owner"] = r.get("user") or "sm2"

    ordered = sorted(jobs.values(), key=lambda r: (not r["is_chou"], 0 if (r["gpus"] or 0) > 0 else 1, -int(r["job_id"]) if r["job_id"].isdigit() else 0))
    return ordered, scheduler_ts


def _parse_slash_quad(token: str) -> dict:
    parts = [p.strip() for p in token.split("/")]
    while len(parts) < 4:
        parts.append("-")
    return {
        "run": _to_int(parts[0]) or 0,
        "queue": _to_int(parts[1]) or 0,
        "hold": _to_int(parts[2]) or 0,
        "limit": _to_int(parts[3]),
    }


def _parse_cores_gpus(token: str) -> tuple[int, int | None]:
    m = re.search(r"(\d[\d,]*)(?:\s*\((\d[\d,]*)\))?", token.strip())
    if not m:
        return 0, None
    return _to_int(m.group(1)) or 0, _to_int(m.group(2)) if m.group(2) is not None else None


def parse_jobinfo_s(s_text: str) -> tuple[dict, dict, str | None]:
    user_group: dict = {"user": None, "group": None, "queue": "H", "rows": []}
    queue_status: dict = {"special": [], "within_week": [], "over_week": []}
    scheduler_ts: str | None = None
    mode: str | None = None
    horizon: str | None = None

    for raw in s_text.splitlines():
        line = raw.strip()
        m_ts = re.search(r"Job Status at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m_ts:
            scheduler_ts = m_ts.group(1)
            continue
        if not line or set(line) <= {"-"} or line.startswith("note:"):
            continue
        if line.startswith("User/Group Stat:"):
            mode = "ug"
            continue
        if line.startswith("Queue Status"):
            mode = "qs"
            m_q = re.search(r"Queue Status \(([^)]+)\)", line)
            if m_q:
                user_group["queue"] = m_q.group(1)
            continue

        if mode == "ug":
            if "user(" in line and "group(" in line:
                m_u = re.search(r"user\(([^)]+)\)", line)
                m_g = re.search(r"group\(([^)]+)\)", line)
                if m_u:
                    user_group["user"] = m_u.group(1)
                if m_g:
                    user_group["group"] = m_g.group(1)
                continue
            if "|" in line and line.count("|") >= 2:
                left, mid, right = [c.strip() for c in line.split("|", 2)]
                metric = left.split("(", 1)[0].strip()
                user_group["rows"].append({
                    "metric": metric,
                    "user": _parse_slash_quad(mid),
                    "group": _parse_slash_quad(right),
                })
        elif mode == "qs":
            low = line.lower()
            if low == "special types":
                horizon = "special"
                continue
            if low == "jobs within a week":
                horizon = "within_week"
                continue
            if low == "jobs more than a week":
                horizon = "over_week"
                continue
            if low.startswith("job ") or low.startswith("type "):
                continue
            if horizon and "|" in line and line.count("|") >= 4:
                parts = [c.strip() for c in line.split("|", 4)]
                free_cores, free_gpus = _parse_cores_gpus(parts[2])
                req_cores, req_gpus = _parse_cores_gpus(parts[4])
                queue_status[horizon].append({
                    "job_type": parts[0],
                    "free_nodes": _to_int(parts[1]) or 0,
                    "free_cores": free_cores,
                    "free_gpus": free_gpus,
                    "jobs_waiting": _to_int(parts[3]) or 0,
                    "req_cores": req_cores,
                    "req_gpus": req_gpus,
                })

    return user_group, queue_status, scheduler_ts


def parse_showlim_c(lim_text: str) -> dict:
    result: dict = {"group": None, "user": None}
    current: dict | None = None
    for raw in lim_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("CPU resource information:"):
            if "User=" in line:
                m_u = re.search(r"User=([^\s/]+)", line)
                m_g = re.search(r"Group=([^\s(]+)", line)
                current = {"scope": "user", "name": m_u.group(1) if m_u else "sm2", "group": m_g.group(1) if m_g else None}
                result["user"] = current
            else:
                m_g = re.search(r"Group=([^\s(]+)", line)
                current = {"scope": "group", "name": m_g.group(1) if m_g else "ew6"}
                result["group"] = current
            continue
        if current is not None and line.startswith("TOTAL"):
            tokens = line.split()
            if len(tokens) >= 5:
                limit = _to_int(tokens[1])
                total_use = _to_int(tokens[2])
                ended = _to_int(tokens[3])
                running = _to_int(tokens[4])
                remaining = max(0, limit - total_use) if (limit is not None and total_use is not None) else None
                current.update(limit=limit, total_use=total_use, ended_job=ended, running_job=running, remaining=remaining)
    return result


def _parse_utc_epoch(val: object) -> float | None:
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _parse_elapsed_s(text: str | None) -> float | None:
    if not text:
        return None
    s = text.strip()
    if not s or s in {"-", "---"}:
        return None
    days = 0
    if "-" in s:
        d_part, _, s = s.partition("-")
        if not d_part.isdigit():
            return None
        days = int(d_part)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, sec = 0, int(parts[0]), int(parts[1])
        else:
            return None
    except ValueError:
        return None
    return float(days * 86400 + h * 3600 + m * 60 + sec)


def _fmt_jst_ts(epoch_s: float | None) -> str | None:
    if epoch_s is None:
        return None
    return datetime.fromtimestamp(epoch_s, tz=JST).strftime("%Y-%m-%d %H:%M JST")


def _fmt_hms(seconds: float | None) -> str | None:
    if seconds is None or seconds < 0:
        return None
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _compute_s_per_epoch(curve: list[dict], last_log: dict | None) -> float | None:
    if len(curve) >= 2:
        start_idx = len(curve) - 1
        while start_idx > 0:
            e_prev, e_cur = curve[start_idx - 1].get("epoch"), curve[start_idx].get("epoch")
            t_prev, t_cur = curve[start_idx - 1].get("elapsed_s"), curve[start_idx].get("elapsed_s")
            if all(isinstance(v, (int, float)) for v in (e_prev, e_cur, t_prev, t_cur)) and e_cur > e_prev and t_cur > t_prev:
                start_idx -= 1
            else:
                break
        if start_idx < len(curve) - 1:
            e0, e1 = curve[start_idx]["epoch"], curve[-1]["epoch"]
            t0, t1 = curve[start_idx]["elapsed_s"], curve[-1]["elapsed_s"]
            return round((t1 - t0) / (e1 - e0), 1)
    if isinstance(last_log, dict) and last_log.get("epoch") and last_log.get("elapsed_s"):
        return round(float(last_log["elapsed_s"]) / float(last_log["epoch"]), 1)
    return None


def _compute_best_metrics(curve: list[dict]) -> dict:
    best_train_mae = None
    best_train_ep = None
    best_dev_mae = None
    best_dev_ep = None
    for pt in curve:
        ep = pt.get("epoch")
        tm = pt.get("train_mae_eV")
        dm = pt.get("dev_mae_eV")
        if isinstance(tm, (int, float)) and (best_train_mae is None or tm < best_train_mae):
            best_train_mae = float(tm)
            best_train_ep = ep
        if isinstance(dm, (int, float)) and (best_dev_mae is None or dm < best_dev_mae):
            best_dev_mae = float(dm)
            best_dev_ep = ep
    return {
        "best_train_mae_eV": best_train_mae,
        "best_train_epoch": best_train_ep,
        "best_dev_mae_eV": best_dev_mae,
        "best_dev_epoch": best_dev_ep,
    }


def _enrich_chou_runs(runs: list[dict], jobs: list[dict]) -> list[dict]:
    jobs_by_id = {j["job_id"]: j for j in jobs}
    enriched = []
    for run in runs:
        candidates: list[str] = []
        resume_jobs = run.get("resume_jobs")
        if isinstance(resume_jobs, list):
            candidates.extend(reversed(resume_jobs))
        elif isinstance(resume_jobs, str):
            candidates.append(resume_jobs)
        if run.get("training_job"):
            candidates.append(run["training_job"])
        if run.get("preflight_job"):
            candidates.append(run["preflight_job"])

        live_job = None
        jid = None
        for raw_cid in candidates:
            cid = _strip_job_id(raw_cid)
            if not cid:
                continue
            if jid is None:
                jid = cid
            if cid in jobs_by_id:
                live_job = jobs_by_id[cid]
                jid = cid
                break

        items_out = []
        order = run.get("serial_order") or list((run.get("items") or {}).keys())
        for item_id in order:
            info = dict((run.get("items") or {}).get(item_id) or {"status": "NOT_STARTED"})
            last_log = info.get("last_log")
            total_passes = info.get("total_passes")
            curve = info.get("curve") or []
            s_per_epoch = _compute_s_per_epoch(curve, last_log)
            eta_s = None
            if info.get("status") == "RUNNING" and s_per_epoch and total_passes and isinstance(last_log, dict) and last_log.get("epoch"):
                rem = max(0, int(total_passes) - int(last_log["epoch"]))
                eta_s = round(rem * s_per_epoch, 1)
            cm = info.get("completion_manifest")
            attempts = info.get("attempts") or []
            wall_s = (
                (attempts[-1].get("wall_s") if attempts else None)
                or (cm.get("wall_s") if isinstance(cm, dict) else None)
                or (last_log.get("elapsed_s") if info.get("status") == "COMPLETE" and isinstance(last_log, dict) else None)
            )
            peak_rss = info.get("peak_rss_bytes") or (cm.get("peak_rss_bytes") if isinstance(cm, dict) else None)
            finished_utc = info.get("cm_mtime_utc") or (info.get("log_mtime_utc") if info.get("status") == "COMPLETE" else None)
            best = _compute_best_metrics(curve)
            items_out.append({
                "item_id": item_id,
                "status": info.get("status"),
                "total_passes": total_passes,
                "first_log": info.get("first_log"),
                "last_log": last_log,
                "s_per_epoch": s_per_epoch,
                "eta_s": eta_s,
                "wall_s": wall_s,
                "peak_rss_bytes": peak_rss,
                "dev_mae_eV": info.get("dev_mae_eV"),
                **best,
                "finished_utc": finished_utc,
                "finished_jst": _fmt_jst_ts(finished_utc),
                "completion_manifest": cm,
                "attempts": attempts,
                "reason": info.get("reason"),
                "log_tail": info.get("log_tail") or [],
                "curve": curve,
            })
        all_job_ids = []
        for raw_cid in [run.get("training_job"), *(resume_jobs if isinstance(resume_jobs, list) else [])]:
            cid = _strip_job_id(raw_cid)
            if cid and cid not in all_job_ids:
                all_job_ids.append(cid)
        enriched.append({
            **{k: v for k, v in run.items() if k != "items"},
            "job_id": jid,
            "initial_job_id": _strip_job_id(run.get("training_job")),
            "all_job_ids": all_job_ids,
            "live_job": live_job,
            "items": items_out,
        })
    return enriched


def _compute_queue_wait_history(chou_runs: list[dict], now_epoch: float) -> list[dict]:
    records: list[dict] = []
    for run in chou_runs:
        res = run.get("resources") or {}
        live = run.get("live_job") or {}
        walltime = res.get("walltime") or (_fmt_hms(run.get("serial_wall_s")) if run.get("serial_wall_s") else None)
        gpus = res.get("gpus") if res.get("gpus") is not None else live.get("gpus")
        cpus = res.get("cpus") if res.get("cpus") is not None else live.get("cpus")
        driver_starts = [t for t in (_parse_utc_epoch(ts) for ts in (run.get("serial_driver_starts") or [])) if t is not None]

        init_submit = _parse_utc_epoch(run.get("staging_manifest_mtime_utc")) or _parse_utc_epoch(run.get("submitted_at"))
        if init_submit is None and not run.get("resume_jobs"):
            init_submit = _parse_utc_epoch(run.get("submission_mtime_utc"))
        first_start = driver_starts[0] if driver_starts else _parse_utc_epoch(run.get("stage_train_started_at"))
        if first_start is None and not run.get("resume_jobs") and live.get("status") == "Run":
            el = _parse_elapsed_s(live.get("elapsed"))
            if el is not None:
                first_start = now_epoch - el
        if init_submit is not None and first_start is not None:
            wait_s = first_start - init_submit
            if 0 <= wait_s <= 14 * 86400:
                records.append({
                    "run_id": run.get("run_id"),
                    "kind": "initial",
                    "job_id": run.get("initial_job_id") or run.get("job_id"),
                    "submitted_epoch": init_submit,
                    "started_epoch": first_start,
                    "submitted_jst": _fmt_jst_ts(init_submit),
                    "started_jst": _fmt_jst_ts(first_start),
                    "wait_s": round(wait_s, 1),
                    "walltime": walltime,
                    "gpus": gpus,
                    "cpus": cpus,
                })

        resume_jobs = run.get("resume_jobs") or []
        if isinstance(resume_jobs, list) and resume_jobs:
            res_jid = _strip_job_id(resume_jobs[-1])
            res_submit = _parse_utc_epoch(run.get("submission_mtime_utc"))
            res_start = driver_starts[-1] if len(driver_starts) >= 2 else None
            if res_start is None and live.get("job_id") == res_jid and live.get("status") == "Run":
                el = _parse_elapsed_s(live.get("elapsed"))
                if el is not None:
                    res_start = now_epoch - el
            if res_submit is not None and res_start is not None and (first_start is None or res_submit >= first_start):
                wait_s = res_start - res_submit
                if 0 <= wait_s <= 14 * 86400:
                    records.append({
                        "run_id": run.get("run_id"),
                        "kind": "resume",
                        "job_id": res_jid,
                        "submitted_epoch": res_submit,
                        "started_epoch": res_start,
                        "submitted_jst": _fmt_jst_ts(res_submit),
                        "started_jst": _fmt_jst_ts(res_start),
                        "wait_s": round(wait_s, 1),
                        "walltime": walltime,
                        "gpus": gpus,
                        "cpus": cpus,
                    })
    records.sort(key=lambda r: r["submitted_epoch"], reverse=True)
    return records[:20]


def fetch_snapshot(*, host: str, port: int, user: str, key_path: Path, remote_root: str) -> dict:
    if not re.fullmatch(r"/lustre/home/users/sm2/chou(?:/[A-Za-z0-9_-]+)*", remote_root):
        raise ValueError("Refusing remote root outside /lustre/home/users/sm2/chou")
    script = _REMOTE_SCRIPT.replace("__BOUNDARY__", remote_root)
    t0 = time.monotonic()
    cmd = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-i", str(key_path), "-p", str(port), f"{user}@{host}", "bash -s",
    ]
    proc = subprocess.run(cmd, input=script.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=45)
    latency_ms = int(round((time.monotonic() - t0) * 1000))
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0 and "===__SEC_JOBINFO_C__===" not in stdout:
        raise RuntimeError(f"SSH query failed (rc={proc.returncode}): {stderr or stdout.strip()}")

    codes = dict(re.findall(r"===__RC_(\w+)__(\d+)===", stdout))
    failed = [name for name in _SECTIONS if codes.get(name) != "0"]
    if failed:
        raise RuntimeError("partial:" + ",".join(failed) if len(failed) < len(_SECTIONS) else "scheduler:all")
    sec = _split_sections(re.sub(r"===__RC_\w+__\d+===", "", stdout))
    if any(sec[name] and "Job Status at" not in sec[name] for name in ("JOBINFO_C", "JOBINFO_M", "JOBINFO_W")):
        raise RuntimeError("parse:jobinfo")
    jobs, ts1 = parse_jobinfo_cmw(sec["JOBINFO_C"], sec["JOBINFO_M"], sec["JOBINFO_W"], remote_root=remote_root)
    user_group_stat, queue_status, ts2 = parse_jobinfo_s(sec["JOBINFO_S"])
    points = parse_showlim_c(sec["SHOWLIM_C"])
    chou_raw = []
    if not sec["CHOU_RUNS"]:
        raise RuntimeError("parse:CHOU_RUNS")
    if sec["CHOU_RUNS"]:
        try:
            chou_raw = json.loads(sec["CHOU_RUNS"])["runs"]
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            raise RuntimeError("parse:CHOU_RUNS")
    if not ts2 or not user_group_stat["rows"] or not points.get("group"):
        raise RuntimeError("parse:scheduler")
    if not isinstance(chou_raw, list):
        raise RuntimeError("parse:CHOU_RUNS")
    chou_runs = _enrich_chou_runs(chou_raw, jobs)
    now_epoch = time.time()
    queue_wait_history = _compute_queue_wait_history(chou_runs, now_epoch)

    gpu_1w = next((r for r in queue_status["within_week"] if r["job_type"] == "gpu"), None)
    gpu_gt1w = next((r for r in queue_status["over_week"] if r["job_type"] == "gpu"), None)

    return {
        "fetched_at_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
        "fetched_at_epoch": now_epoch,
        "scheduler_timestamp": ts2 or ts1,
        "ssh_latency_ms": latency_ms,
        "host": host,
        "user": user,
        "remote_root": remote_root,
        "summary": {
            "gpu_within_week": gpu_1w,
            "gpu_over_week": gpu_gt1w,
            "chou_running_jobs": sum(1 for j in jobs if j["is_chou"] and j["status"] == "Run"),
            "chou_queued_jobs": sum(1 for j in jobs if j["is_chou"] and j["status"] != "Run"),
            "sm2_total_jobs": len(jobs),
        },
        "user_group_stat": user_group_stat,
        "queue_status": queue_status,
        "points": points,
        "jobs": jobs,
        "chou_runs": chou_runs,
        "queue_wait_history": queue_wait_history,
        "raw": {
            "jobinfo_c": sec["JOBINFO_C"],
            "jobinfo_m": sec["JOBINFO_M"],
            "jobinfo_w": sec["JOBINFO_W"],
            "jobinfo_s": sec["JOBINFO_S"],
            "showlim_c": sec["SHOWLIM_C"],
        },
    }


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def render_terminal(snap: dict) -> str:
    lines: list[str] = []
    lines.append(f"IMS/RCCS Queue H | Fetched: {snap['fetched_at_jst']} | Scheduler: {snap.get('scheduler_timestamp') or '-'} | SSH {snap['ssh_latency_ms']}ms")
    lines.append("")

    s = snap["summary"]
    g1 = s.get("gpu_within_week") or {}
    g2 = s.get("gpu_over_week") or {}
    lines.append(
        f"[A100 Queue H] 1w: free={g1.get('free_gpus', '-')} GPU ({g1.get('free_cores', '-')} CPU), "
        f"waiting={g1.get('jobs_waiting', '-')} jobs ({g1.get('req_gpus', '-')} GPU) | "
        f">1w: free={g2.get('free_gpus', '-')} GPU, waiting={g2.get('jobs_waiting', '-')} jobs"
    )

    ug = snap["user_group_stat"]
    for row in ug.get("rows", []):
        if row["metric"] in {"NJob", "GPUs", "CPUs"}:
            u, g = row["user"], row["group"]
            ulim = str(u["limit"]) if u["limit"] is not None else "-"
            glim = str(g["limit"]) if g["limit"] is not None else "-"
            lines.append(
                f"  {row['metric']:4s}: user({ug['user']}) Run/Q/H/Lim={u['run']}/{u['queue']}/{u['hold']}/{ulim}"
                f" | group({ug['group']})={g['run']}/{g['queue']}/{g['hold']}/{glim}"
            )

    pts = snap.get("points") or {}
    gp = pts.get("group") or {}
    up = pts.get("user") or {}
    if gp:
        lines.append(
            f"[Points] group({gp.get('name')}): used={gp.get('total_use', 0):,} / {gp.get('limit', 0):,} "
            f"(remain={gp.get('remaining', 0):,}, running={gp.get('running_job', 0):,}) | "
            f"user({up.get('name')}): used={up.get('total_use', 0):,} (running={up.get('running_job', 0):,})"
        )

    lines.append("")
    lines.append("[sm2 Active Jobs]")
    lines.append(f"  {'JobID':9s} {'St':4s} {'Res':6s} {'Elapsed':8s} {'Mem(MB)':13s} {'Node':8s} {'Owner':10s} {'Name / Workdir'}")
    for j in snap.get("jobs", []):
        res = f"{j['cpus']}c+{j['gpus']}g" if j.get("gpus") else f"{j['cpus']}c"
        mem = f"{j.get('used_mem_mb') or '-'}/{j.get('lim_mem_mb') or '-'}"
        owner = j.get("owner") or ("chou" if j["is_chou"] else "other")
        lines.append(
            f"  {j['job_id']:9s} {(j.get('status') or '-'):4s} {res:6s} {(j.get('elapsed') or '-'):8s} "
            f"{mem:13s} {(j.get('node_or_reason') or '-'):8s} {owner:10s} {j.get('name') or '-'}  ({j.get('workdir') or '-'})"
        )

    active_runs = [r for r in snap.get("chou_runs", []) if r.get("live_job") or r.get("stage_train_status") == "running"]
    if active_runs:
        lines.append("")
        lines.append("[Active chou V4 Runs]")
        for r in active_runs:
            live = r.get("live_job") or {}
            lines.append(f"  * {r['run_id']} (Job {r.get('job_id')} on {live.get('node_or_reason')}, {live.get('elapsed')})")
            for item in r.get("items", []):
                ll = item.get("last_log") or {}
                if item["status"] == "COMPLETE":
                    lines.append(f"      - {item['item_id']:14s}: COMPLETE ({_fmt_dur(item.get('wall_s'))})")
                elif item["status"] == "RUNNING":
                    ep = ll.get("epoch", "-")
                    tp = f"/{item['total_passes']}" if item.get("total_passes") else ""
                    tmae = f"{ll['train_mae_eV']*1000:.2f} meV" if isinstance(ll.get("train_mae_eV"), (int, float)) else "-"
                    dmae = f" | dev={ll['dev_mae_eV']*1000:.2f} meV" if isinstance(ll.get("dev_mae_eV"), (int, float)) else ""
                    spd = f"{item['s_per_epoch']}s/ep" if item.get("s_per_epoch") else "-"
                    eta = _fmt_dur(item.get("eta_s"))
                    lines.append(f"      - {item['item_id']:14s}: RUNNING Ep {ep}{tp} | train={tmae}{dmae} | {spd}, ETA {eta}")
                else:
                    lines.append(f"      - {item['item_id']:14s}: {item['status']}")
    return "\n".join(lines)


def public_snapshot(snap: dict) -> dict:
    """Allowlist values, not only keys: no remote free text crosses this boundary."""
    def numbers(obj, keys):
        return {k: obj[k] for k in keys if type(obj.get(k)) in (int, float) and math.isfinite(obj[k])}

    def job(obj):
        jid = str(obj.get("job_id", ""))
        return {"job_id": jid if re.fullmatch(r"[0-9]+", jid) else None,
                "status": obj.get("status") if obj.get("status") in {"Run", "Que", "Hold", "Wait", "End"} else "UNKNOWN",
                **numbers(obj, ("cpus", "gpus"))}

    jobs = [job(j) for j in snap.get("jobs", []) if j.get("is_chou") is True]
    runs = []
    for r in snap.get("chou_runs", []):
        items = []
        for i in r.get("items", []):
            item = numbers(i, ("total_passes", "eta_s", "s_per_epoch"))
            item["status"] = i.get("status") if i.get("status") in {"RUNNING", "COMPLETE", "PENDING", "FAILED", "SKIPPED"} else "UNKNOWN"
            item["epoch"] = numbers(i.get("last_log") or i, ("epoch",)).get("epoch")
            items.append(item)
        runs.append({"job_id": job(r)["job_id"], "items": items})
    return {"public": True, "fetched_at_epoch": snap["fetched_at_epoch"], "jobs": jobs, "chou_runs": runs,
            "points": {scope: numbers(snap.get("points", {}).get(scope) or {},
                       ("limit", "remaining", "total_use", "running_job")) for scope in ("user", "group")}}


def atomic_snapshot(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class SnapshotState:
    """HTTP reader has no connection settings or collection capability."""
    min_interval = 90

    def __init__(self, path: Path, stale_after: float = 240):
        self.path, self.stale_after = path, stale_after

    def get(self, *, force=False) -> dict:
        try:
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
            snap = envelope["snapshot"]
            failure = envelope.get("failure")
            if failure not in {None, "ssh_timeout", "ssh", "partial", "scheduler", "parse", "collector"}:
                raise ValueError("invalid failure")
            if snap is None:
                return {"public": True, "available": False, "stale": True, "failure": failure or "never_succeeded"}
            if snap.get("public") is not True or not isinstance(snap["chou_runs"], list):
                raise ValueError("not a public snapshot")
            stamp = snap["fetched_at_epoch"]
            if type(stamp) not in (float, int) or not math.isfinite(stamp) or stamp > time.time() + 60:
                raise ValueError("invalid timestamp")
            # Reapply the allowlist even if an operator points at a wrong file.
            clean = public_snapshot({**snap, "jobs": [{**j, "is_chou": True} for j in snap["jobs"]]})
            return {**clean, "available": True, "stale": bool(failure) or time.time() - stamp > self.stale_after,
                    "failure": failure or ("collector_overdue" if time.time() - stamp > self.stale_after else None), "last_success_epoch": stamp}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return {"public": True, "available": False, "stale": True, "failure": "snapshot_unreadable"}


def collect_once(path: Path, **connection) -> bool:
    prior = SnapshotState(path).get()
    last = prior if prior.get("available") else None
    try:
        snap = public_snapshot(fetch_snapshot(**connection))
        failure = None
    except subprocess.TimeoutExpired:
        snap, failure = last, "ssh_timeout"
    except Exception as exc:
        category = str(exc).split(":", 1)[0]
        snap, failure = last, category if category in {"partial", "scheduler", "parse"} else "ssh"
    atomic_snapshot(path, {"snapshot": snap, "failure": failure})
    return failure is None


class MonitorState:
    def __init__(self, *, host: str, port: int, user: str, key_path: Path, remote_root: str, min_interval: float) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.key_path = key_path
        self.remote_root = remote_root
        self.min_interval = max(5.0, float(min_interval))
        self._cond = threading.Condition()
        self._snapshot: dict | None = None
        self._last_fetch: float = 0.0
        self._in_flight: bool = False

    def get(self, *, force: bool = False) -> dict:
        with self._cond:
            while self._in_flight:
                self._cond.wait()
            now = time.monotonic()
            age = now - self._last_fetch if self._snapshot is not None else 1e9
            if self._snapshot is not None and (not force or age < self.min_interval):
                cooldown = max(0.0, round(self.min_interval - age, 1))
                return {**self._snapshot, "cached": True, "cooldown_remaining_s": cooldown, "min_interval_s": self.min_interval}
            self._in_flight = True

        err: str | None = None
        fresh: dict | None = None
        try:
            fresh = fetch_snapshot(
                host=self.host, port=self.port, user=self.user,
                key_path=self.key_path, remote_root=self.remote_root,
            )
        except Exception as exc:
            err = str(exc)

        with self._cond:
            self._in_flight = False
            if fresh is not None:
                self._snapshot = fresh
                self._last_fetch = time.monotonic()
                payload = {**fresh, "cached": False, "cooldown_remaining_s": self.min_interval, "min_interval_s": self.min_interval, "error": None}
            elif self._snapshot is not None:
                payload = {**self._snapshot, "cached": True, "cooldown_remaining_s": self.min_interval, "min_interval_s": self.min_interval, "error": err}
            else:
                payload = {"error": err or "Unknown error", "cached": False, "min_interval_s": self.min_interval}
            self._cond.notify_all()
            return payload


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>分子研 IMS/RCCS 队列与实验看板</title>
<style>
  :root {
    --bg:#090e17; --bg-grad:radial-gradient(1100px 420px at 15% -10%, rgba(56,189,248,.11), transparent 60%), radial-gradient(900px 380px at 85% -12%, rgba(34,197,94,.08), transparent 60%);
    --panel:#111827; --panel2:#182235; --panel3:#0c1320; --border:#24324a;
    --text:#f1f5f9; --muted:#94a3b8; --accent:#38bdf8; --ok:#22c55e;
    --warn:#f59e0b; --bad:#ef4444; --chip:#1e293b; --shadow:0 8px 24px rgba(0,0,0,.28);
  }
  html[data-theme="light"] {
    --bg:#f3f6fb; --bg-grad:radial-gradient(1000px 400px at 15% -10%, rgba(2,132,199,.08), transparent 60%), radial-gradient(800px 360px at 85% -12%, rgba(22,163,74,.06), transparent 60%);
    --panel:#ffffff; --panel2:#f8fafc; --panel3:#f1f5f9; --border:#e2e8f0;
    --text:#0f172a; --muted:#64748b; --accent:#0284c7; --ok:#16a34a;
    --warn:#d97706; --bad:#dc2626; --chip:#f1f5f9; --shadow:0 6px 18px rgba(15,23,42,.06);
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); background-image:var(--bg-grad); color:var(--text); line-height:1.5; min-height:100vh; }
  .wrap { max-width:1360px; margin:0 auto; padding:20px 22px 48px; }
  header { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:14px; padding-bottom:16px; border-bottom:1px solid var(--border); margin-bottom:16px; }
  h1 { margin:0; font-size:21px; font-weight:700; letter-spacing:.2px; display:flex; align-items:center; gap:10px; }
  .logo-dot { width:10px; height:10px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); display:inline-block; }
  .pill { font-size:12px; padding:2px 10px; border-radius:999px; font-weight:600; background:var(--chip); border:1px solid var(--border); color:var(--muted); display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }
  .pill.ok { color:#4ade80; border-color:rgba(34,197,94,.38); background:rgba(34,197,94,.12); }
  .pill.info { color:#38bdf8; border-color:rgba(56,189,248,.4); background:rgba(56,189,248,.12); }
  .pill.warn { color:#fbbf24; border-color:rgba(245,158,11,.42); background:rgba(245,158,11,.12); }
  .pill.bad { color:#f87171; border-color:rgba(239,68,68,.42); background:rgba(239,68,68,.12); }
  html[data-theme="light"] .pill.ok { color:#15803d; }
  html[data-theme="light"] .pill.info { color:#0369a1; }
  html[data-theme="light"] .pill.warn { color:#b45309; }
  html[data-theme="light"] .pill.bad { color:#b91c1c; }
  .pulse { width:7px; height:7px; border-radius:50%; background:currentColor; animation:pulse 1.8s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.4; transform:scale(.8); } }
  .controls { display:flex; flex-wrap:wrap; align-items:center; gap:9px; font-size:13px; color:var(--muted); }
  select, button { background:var(--panel2); color:var(--text); border:1px solid var(--border); border-radius:9px; padding:7px 12px; font-size:13px; cursor:pointer; transition:border-color .15s, background .15s; }
  select:hover, button:hover { border-color:var(--accent); }
  button.primary { background:linear-gradient(135deg,#0284c7,#0ea5e9); color:#fff; border-color:#0ea5e9; font-weight:600; }
  button.primary:disabled { opacity:.55; cursor:not-allowed; }
  .banner-err { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.42); color:var(--bad); padding:10px 14px; border-radius:10px; margin-bottom:14px; display:none; font-size:13px; }
  .hero { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; background:var(--panel); border:1px solid var(--border); border-left:4px solid var(--ok); border-radius:12px; padding:12px 16px; margin-bottom:16px; box-shadow:var(--shadow); font-size:13.5px; }
  .hero.tight { border-left-color:var(--warn); }
  .grid4 { display:grid; grid-template-columns:repeat(auto-fit,minmax(255px,1fr)); gap:14px; margin-bottom:18px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px 18px; box-shadow:var(--shadow); }
  .kpi-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .kpi-title { font-size:12px; color:var(--muted); font-weight:600; letter-spacing:.35px; }
  .kpi-val { font-size:28px; font-weight:700; margin:6px 0 4px; font-variant-numeric:tabular-nums; letter-spacing:-.3px; }
  .kpi-sub { font-size:12px; color:var(--muted); }
  .slots { display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }
  .slot-dot { width:10px; height:10px; border-radius:3px; background:var(--ok); box-shadow:0 0 6px rgba(34,197,94,.45); }
  .slot-dot.dim { background:var(--chip); border:1px solid var(--border); box-shadow:none; }
  .bar { height:8px; background:var(--panel3); border-radius:999px; overflow:hidden; margin-top:9px; border:1px solid var(--border); display:flex; }
  .bar-seg-a { height:100%; background:linear-gradient(90deg,#0284c7,#38bdf8); transition:width .35s; }
  .bar-seg-b { height:100%; background:#f59e0b; transition:width .35s; }
  .bar-ok { height:100%; background:linear-gradient(90deg,#38bdf8,#22c55e); transition:width .35s; }
  .donut-wrap { display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .donut svg { transform:rotate(-90deg); display:block; }
  .sec-head { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; margin:20px 0 10px; }
  .sec-head h2 { margin:0; font-size:16px; font-weight:700; }
  .run-card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:18px; margin-bottom:14px; box-shadow:var(--shadow); }
  .run-head { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; padding-bottom:12px; border-bottom:1px solid var(--border); margin-bottom:14px; }
  .run-meta { display:flex; flex-wrap:wrap; align-items:center; gap:10px; font-size:12.5px; color:var(--muted); margin-top:4px; }
  .chain { display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:9px; margin-bottom:14px; }
  .step { padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:var(--panel2); font-size:12px; cursor:pointer; user-select:none; transition:transform .12s, border-color .15s, box-shadow .15s, background .15s; }
  .step:hover { transform:translateY(-1px); border-color:var(--accent); }
  .step.COMPLETE { border-color:rgba(34,197,94,.42); background:rgba(34,197,94,.08); }
  .step.RUNNING { border-color:rgba(56,189,248,.65); background:rgba(56,189,248,.11); }
  .step.NOT_STARTED { opacity:.76; }
  .step.selected { border-color:var(--accent); box-shadow:0 0 0 2px rgba(56,189,248,.48), 0 4px 12px rgba(56,189,248,.14); }
  .step.COMPLETE.selected { border-color:var(--ok); box-shadow:0 0 0 2px rgba(34,197,94,.52), 0 4px 12px rgba(34,197,94,.16); }
  .step.NOT_STARTED.selected { opacity:1; }
  .spot { display:grid; grid-template-columns:1.08fr 1.12fr; gap:16px; background:var(--panel2); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
  @media (max-width:940px) { .spot { grid-template-columns:1fr; } }
  .spot-metrics { display:grid; grid-template-columns:repeat(3,minmax(108px,1fr)); gap:9px; margin-top:12px; }
  @media (max-width:640px) { .spot-metrics { grid-template-columns:repeat(2,minmax(110px,1fr)); } }
  .mini-stat { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:8px 10px; }
  .mini-stat .k { font-size:11px; color:var(--muted); }
  .mini-stat .v { font-size:16px; font-weight:700; margin-top:2px; font-variant-numeric:tabular-nums; }
  .mini-stat .s { font-size:11px; color:var(--muted); margin-top:2px; font-variant-numeric:tabular-nums; }
  .chart-box { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 12px; position:relative; display:flex; flex-direction:column; justify-content:space-between; }
  .chart-head { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; font-size:12px; color:var(--muted); margin-bottom:4px; }
  .legend { display:inline-flex; gap:10px; align-items:center; font-size:11px; }
  .lg-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }
  .pred-card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px 18px; box-shadow:var(--shadow); margin-top:18px; }
  .pred-top { display:grid; grid-template-columns:1.15fr 1.25fr; gap:16px; align-items:start; margin-top:12px; }
  @media (max-width:960px) { .pred-top { grid-template-columns:1fr; } }
  .pred-form { display:grid; grid-template-columns:repeat(3,minmax(110px,1fr)); gap:10px; background:var(--panel2); border:1px solid var(--border); border-radius:12px; padding:12px; }
  @media (max-width:620px) { .pred-form { grid-template-columns:1fr; } }
  .pred-field label { display:block; font-size:11px; color:var(--muted); margin-bottom:5px; font-weight:600; }
  .pred-field select { width:100%; }
  .pred-res { background:var(--panel2); border:1px solid var(--border); border-left:4px solid var(--ok); border-radius:12px; padding:12px 14px; }
  .pred-res.warn { border-left-color:var(--warn); }
  .pred-res.bad { border-left-color:var(--bad); }
  .pred-factors { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:8px; margin-top:10px; font-size:12px; }
  .pred-f { background:var(--panel); border:1px solid var(--border); border-radius:9px; padding:8px 10px; }
  .owner-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
  .owner-chip { background:var(--panel2); border:1px solid var(--border); color:var(--muted); padding:4px 10px; border-radius:999px; font-size:12px; cursor:pointer; }
  .owner-chip.active { border-color:var(--accent); color:var(--accent); background:rgba(56,189,248,.12); font-weight:600; }
  .tooltip { position:absolute; pointer-events:none; background:var(--panel3); color:var(--text); border:1px solid var(--accent); border-radius:8px; padding:6px 9px; font-size:11.5px; box-shadow:var(--shadow); display:none; z-index:10; white-space:nowrap; }
  .tabs { display:flex; flex-wrap:wrap; gap:6px; border-bottom:1px solid var(--border); margin:20px 0 12px; padding-bottom:8px; }
  .tab-btn { background:transparent; border:1px solid transparent; color:var(--muted); padding:7px 14px; border-radius:9px; font-weight:600; font-size:13px; }
  .tab-btn.active { background:var(--panel); border-color:var(--border); color:var(--accent); box-shadow:var(--shadow); }
  .tab-pane { display:none; }
  .tab-pane.active { display:block; }
  table { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
  th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; font-size:12px; background:var(--panel2); }
  tr.chou-row { background:rgba(56,189,248,.07); }
  tr.other-row td { color:var(--muted); }
  .mono { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }
  .split2 { display:grid; grid-template-columns:1.35fr 1fr; gap:14px; }
  @media (max-width:980px) { .split2 { grid-template-columns:1fr; } }
  pre { background:var(--panel3); border:1px solid var(--border); border-radius:10px; padding:12px 14px; overflow:auto; font-size:12px; color:var(--text); max-height:360px; margin:0; }
  details { margin-top:10px; }
  summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
  .toast { position:fixed; right:20px; bottom:20px; background:var(--panel2); color:var(--text); border:1px solid var(--accent); padding:9px 14px; border-radius:10px; font-size:13px; box-shadow:var(--shadow); opacity:0; transform:translateY(8px); transition:opacity .2s, transform .2s; pointer-events:none; z-index:50; }
  .toast.show { opacity:1; transform:translateY(0); }
  .skel { background:linear-gradient(90deg,var(--panel) 25%,var(--panel2) 50%,var(--panel) 75%); background-size:200% 100%; animation:shim 1.4s infinite; border-radius:12px; height:112px; border:1px solid var(--border); }
  @keyframes shim { 0% { background-position:200% 0; } 100% { background-position:-200% 0; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1><span class="logo-dot"></span>分子研 IMS/RCCS 调度与实验看板 <span id="connPill" class="pill">连接中</span></h1>
      <div style="font-size:12px;color:var(--muted);margin-top:4px" id="metaLine">正在通过只读 SSH 拉取实时队列与实验状态...</div>
    </div>
    <div class="controls">
      <button id="themeBtn" title="切换深浅色主题">☀️ 浅色</button>
      <label style="display:inline-flex;align-items:center;gap:6px">自动刷新
        <select id="autoSec">
          <option value="0" selected>手动 (关闭)</option>
          <option value="60">每 60 秒</option>
          <option value="180">每 3 分钟</option>
          <option value="300">每 5 分钟</option>
        </select>
      </label>
      <span id="countdown" class="mono" style="font-size:12px"></span>
      <button id="refreshBtn" class="primary">立即刷新</button>
    </div>
  </header>

  <div id="errBox" class="banner-err"></div>
  <div id="heroBar" class="hero" style="display:none"></div>

  <div class="grid4" id="kpiGrid">
    <div class="skel"></div><div class="skel"></div><div class="skel"></div><div class="skel"></div>
  </div>

  <div class="sec-head">
    <h2>chou 个人 V4 实验实时进度</h2>
    <span style="font-size:12px;color:var(--muted)" id="chouSub"></span>
  </div>
  <div id="chouRuns"><div class="skel" style="height:190px"></div></div>

  <div class="pred-card" id="waitPredictor">
    <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px">
      <div>
        <h2 style="margin:0;font-size:16px">A100 排队时间预测（交互试算）</h2>
        <div style="font-size:12px;color:var(--muted);margin-top:2px">综合 Queue H 空闲/排队比、节点碎片与回填概率、ew6 并发配额、每月首周一 09:00 JST 维护窗口与 chou 历史实测</div>
      </div>
      <span class="pill info">启发式预估 · 非官方承诺</span>
    </div>
    <div class="pred-top">
      <div class="pred-form">
        <div class="pred-field">
          <label for="predGpu">计划申请 A100</label>
          <select id="predGpu">
            <option value="1" selected>1 张 A100</option>
            <option value="2">2 张 A100</option>
            <option value="4">4 张 A100</option>
            <option value="0">0 张 (纯 CPU)</option>
          </select>
        </div>
        <div class="pred-field">
          <label for="predCpu">计划申请 CPU</label>
          <select id="predCpu">
            <option value="8">8 核</option>
            <option value="16" selected>16 核</option>
            <option value="24">24 核</option>
            <option value="32">32 核</option>
            <option value="48">48 核</option>
            <option value="64">64 核</option>
            <option value="96">96 核</option>
          </select>
        </div>
        <div class="pred-field">
          <label for="predWall">计划墙钟时间 (Walltime)</label>
          <select id="predWall">
            <option value="1">1 小时 (短测)</option>
            <option value="6">6 小时 (常规短训)</option>
            <option value="12">12 小时</option>
            <option value="16" selected>16 小时 (V4 标准链)</option>
            <option value="24">24 小时 (1 天)</option>
            <option value="48">48 小时 (2 天)</option>
            <option value="72">72 小时 (3 天)</option>
            <option value="120">120 小时 (5 天)</option>
            <option value="192">&gt; 1 周 (8 天)</option>
          </select>
        </div>
      </div>
      <div id="predResult" class="pred-res"><div style="color:var(--muted)">等待数据加载...</div></div>
    </div>
    <div id="predFactors" class="pred-factors"></div>
    <details id="waitHistDetails">
      <summary id="waitHistSummary">查看 chou 历史排队实测记录</summary>
      <div style="overflow:auto;margin-top:10px"><table id="waitHistTbl"></table></div>
    </details>
  </div>

  <div class="tabs" role="tablist">
    <button class="tab-btn active" data-tab="jobs">sm2 作业列表 <span id="tabJobsCnt"></span></button>
    <button class="tab-btn" data-tab="queue">Queue H 全队列与配额</button>
    <button class="tab-btn" data-tab="history">历史 V4 记录 <span id="tabHistCnt"></span></button>
    <button class="tab-btn" data-tab="raw">原始调度输出</button>
  </div>

  <div id="tab-jobs" class="tab-pane active">
    <div id="ownerBar" class="owner-bar"></div>
    <div class="card" style="padding:0;overflow:auto"><table id="jobsTbl"></table></div>
  </div>

  <div id="tab-queue" class="tab-pane">
    <div class="split2">
      <div class="card" style="padding:14px 16px;overflow:auto">
        <div style="font-weight:700;margin-bottom:10px">Queue H 全队列供需</div>
        <table id="queueTbl"></table>
      </div>
      <div class="card" style="padding:14px 16px;overflow:auto">
        <div style="font-weight:700;margin-bottom:10px">User / Group 配额明细</div>
        <table id="ugTbl"></table>
      </div>
    </div>
  </div>

  <div id="tab-history" class="tab-pane">
    <div id="pastRuns"></div>
  </div>

  <div id="tab-raw" class="tab-pane">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font-size:13px;color:var(--muted)">原始 `jobinfo -c/-m/-w/-s` 与 `showlim -c` 快照</span>
        <button id="copyRawBtn">复制全文</button>
      </div>
      <pre id="rawBox"></pre>
    </div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
const $ = id => document.getElementById(id);
let timerId = null, tickId = null, nextAt = 0, cooldownUntil = 0;

function showToast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2600);
}
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('ims_theme', theme);
  $('themeBtn').textContent = theme === 'light' ? '🌙 深色' : '☀️ 浅色';
}
applyTheme(localStorage.getItem('ims_theme') || 'dark');
$('themeBtn').onclick = () => applyTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light');

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  };
});

function esc(s) {
  return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function fmtNum(v) { return v == null ? '-' : Number(v).toLocaleString(); }
function fmtDur(sec) {
  if (sec == null || sec < 0) return '-';
  const s = Math.round(sec), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h > 0 ? `${h}h ${String(m).padStart(2,'0')}m` : `${m}m ${String(r).padStart(2,'0')}s`;
}
function fmtEtaClock(baseEpochSec, etaSec) {
  if (!baseEpochSec || etaSec == null || etaSec < 0) return '-';
  const d = new Date((baseEpochSec + etaSec) * 1000);
  const nowDay = new Date(baseEpochSec * 1000).toLocaleDateString('zh-CN', {timeZone:'Asia/Tokyo'});
  const etaDay = d.toLocaleDateString('zh-CN', {timeZone:'Asia/Tokyo'});
  const hm = d.toLocaleTimeString('zh-CN', {timeZone:'Asia/Tokyo', hour12:false, hour:'2-digit', minute:'2-digit'});
  return (etaDay !== nowDay ? '次日 ' : '今日 ') + hm + ' JST';
}
function parseElapsedSec(txt) {
  if (!txt) return null;
  let s = String(txt).trim(), days = 0;
  if (s.includes('-')) {
    const [d, rest] = s.split('-', 2);
    if (!/^\d+$/.test(d)) return null;
    days = Number(d); s = rest;
  }
  const parts = s.split(':').map(Number);
  if (parts.some(Number.isNaN)) return null;
  if (parts.length === 2) return days * 86400 + parts[0] * 60 + parts[1];
  if (parts.length === 3) return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2];
  return null;
}
function fmtMae(v) { return typeof v === 'number' ? (v * 1000).toFixed(2) + ' meV' : '-'; }
function fmtSci(v) { return typeof v === 'number' ? (v === 0 ? '0' : v.toExponential(1)) : '-'; }
function fmtGb(bytes) { return typeof bytes === 'number' ? (bytes / (1024 ** 3)).toFixed(2) + ' GB' : '-'; }
function fmtJstClockRange(baseEpochSec, minH, maxH) {
  if (!baseEpochSec) return '-';
  const f = h => {
    const d = new Date((baseEpochSec + h * 3600) * 1000);
    return d.toLocaleString('zh-CN', {timeZone:'Asia/Tokyo', month:'2-digit', day:'2-digit', hour12:false, hour:'2-digit', minute:'2-digit'});
  };
  return `${f(minH)} ~ ${f(maxH)} JST`;
}
function nextFirstMondayMaintEpoch(nowEpochSec) {
  const jstMs = nowEpochSec * 1000 + 9 * 3600 * 1000;
  const nowJst = new Date(jstMs);
  for (let addM = 0; addM < 2; addM++) {
    const y = nowJst.getUTCFullYear() + Math.floor((nowJst.getUTCMonth() + addM) / 12);
    const m = (nowJst.getUTCMonth() + addM) % 12;
    const first = new Date(Date.UTC(y, m, 1, 9, 0, 0));
    const offsetDays = (8 - first.getUTCDay()) % 7;
    const maintUtcMs = Date.UTC(y, m, 1 + offsetDays, 9, 0, 0) - 9 * 3600 * 1000;
    if (maintUtcMs / 1000 > nowEpochSec) return maintUtcMs / 1000;
  }
  return null;
}

function buildChartHtml(item, idx) {
  const pts = (item.curve || []).filter(p => typeof p.train_mae_eV === 'number' || typeof p.dev_mae_eV === 'number');
  if (pts.length < 2) return '<div class="chart-box" style="color:var(--muted);display:flex;align-items:center;justify-content:center;min-height:150px">暂无足够 Epoch 点绘制收敛曲线</div>';
  const allVals = [];
  let hasDev = false;
  pts.forEach(p => {
    if (typeof p.train_mae_eV === 'number') allVals.push(p.train_mae_eV * 1000);
    if (typeof p.dev_mae_eV === 'number') { allVals.push(p.dev_mae_eV * 1000); hasDev = true; }
  });
  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
  const padY = Math.max(0.5, (maxV - minV) * 0.12);
  const lo = minV - padY, hi = maxV + padY, span = hi - lo;
  const W = 450, H = 152, L = 46, R = 14, T = 12, B = 24;
  const pw = W - L - R, ph = H - T - B;
  const yOf = v => T + (1 - ((v * 1000) - lo) / span) * ph;
  const mapped = pts.map((p, i) => ({
    x: L + (i / (pts.length - 1)) * pw,
    yt: typeof p.train_mae_eV === 'number' ? yOf(p.train_mae_eV) : null,
    yd: typeof p.dev_mae_eV === 'number' ? yOf(p.dev_mae_eV) : null,
    p,
  }));
  const trainPts = mapped.filter(k => k.yt != null);
  const devPts = mapped.filter(k => k.yd != null);
  const trainLine = trainPts.map(k => `${k.x.toFixed(1)},${k.yt.toFixed(1)}`).join(' ');
  const devLine = devPts.map(k => `${k.x.toFixed(1)},${k.yd.toFixed(1)}`).join(' ');
  const areaBase = devPts.length >= 2 ? devPts : trainPts;
  const areaLine = areaBase.map(k => `${k.x.toFixed(1)},${(devPts.length >= 2 ? k.yd : k.yt).toFixed(1)}`).join(' ');
  const areaPts = `${areaBase[0].x.toFixed(1)},${T+ph} ${areaLine} ${areaBase[areaBase.length-1].x.toFixed(1)},${T+ph}`;
  const grid = [0, 0.5, 1].map(f => {
    const y = T + f * ph, v = (hi - f * span).toFixed(1);
    return `<line x1="${L}" y1="${y.toFixed(1)}" x2="${W-R}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-dasharray="3,3"/>
            <text x="${L-6}" y="${(y+4).toFixed(1)}" text-anchor="end" fill="var(--muted)" font-size="10">${v}</text>`;
  }).join('');

  let bestDevK = null;
  devPts.forEach(k => { if (!bestDevK || k.p.dev_mae_eV < bestDevK.p.dev_mae_eV) bestDevK = k; });

  const dots = mapped.map((k, i) => {
    const isLast = i === mapped.length - 1;
    const isBestDev = bestDevK && k === bestDevK;
    const trStr = typeof k.p.train_mae_eV === 'number' ? (k.p.train_mae_eV * 1000).toFixed(2) : '';
    const dvStr = typeof k.p.dev_mae_eV === 'number' ? (k.p.dev_mae_eV * 1000).toFixed(2) : '';
    const mainY = k.yt ?? k.yd;
    return `${k.yd != null ? `<circle cx="${k.x.toFixed(1)}" cy="${k.yd.toFixed(1)}" r="${isBestDev ? 4.8 : (isLast ? 3.8 : 2.3)}" fill="${isBestDev ? '#22c55e' : 'var(--ok)'}" stroke="${isBestDev ? '#fff' : 'var(--panel)'}" stroke-width="${isBestDev ? 1.8 : 1.2}"/>` : ''}
      ${k.yt != null ? `<circle cx="${k.x.toFixed(1)}" cy="${k.yt.toFixed(1)}" r="${isLast ? 3.8 : 2.4}" fill="var(--accent)" stroke="var(--panel)" stroke-width="1.2"/>` : ''}
      <circle class="c-pt" cx="${k.x.toFixed(1)}" cy="${mainY.toFixed(1)}" r="7" fill="transparent"
        data-ep="${esc(k.p.epoch)}" data-mae="${trStr}" data-dev="${dvStr}" data-el="${esc(fmtDur(k.p.elapsed_s))}" style="cursor:pointer"/>`;
  }).join('');
  const gid = `grad_${idx}`;
  return `<div class="chart-box" data-chart="${idx}">
    <div class="chart-head">
      <div class="legend">
        <span><span class="lg-dot" style="background:var(--accent)"></span>Train</span>
        ${hasDev ? `<span><span class="lg-dot" style="background:var(--ok)"></span>Dev${bestDevK ? ` (最优 Ep ${bestDevK.p.epoch}: ${(bestDevK.p.dev_mae_eV*1000).toFixed(2)})` : ''}</span>` : ''}
      </div>
      <span class="mono">${pts[0].epoch}..${pts[pts.length-1].epoch} ep</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block">
      <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${hasDev ? '#22c55e' : '#38bdf8'}" stop-opacity="0.22"/><stop offset="100%" stop-color="${hasDev ? '#22c55e' : '#38bdf8'}" stop-opacity="0.01"/></linearGradient></defs>
      ${grid}
      <polygon points="${areaPts}" fill="url(#${gid})"/>
      ${trainPts.length >= 2 ? `<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" points="${trainLine}"/>` : ''}
      ${devPts.length >= 2 ? `<polyline fill="none" stroke="var(--ok)" stroke-width="2.2" stroke-linejoin="round" points="${devLine}"/>` : ''}
      ${dots}
      <text x="${L}" y="${H-5}" fill="var(--muted)" font-size="10">Ep ${pts[0].epoch}</text>
      <text x="${W-R}" y="${H-5}" text-anchor="end" fill="var(--muted)" font-size="10">Ep ${pts[pts.length-1].epoch}</text>
    </svg>
    <div class="tooltip"></div>
  </div>`;
}

function bindChartTooltips() {
  document.querySelectorAll('.chart-box').forEach(box => {
    const tip = box.querySelector('.tooltip');
    if (!tip) return;
    box.querySelectorAll('.c-pt').forEach(pt => {
      pt.addEventListener('mouseenter', e => {
        const d = pt.dataset;
        const parts = [`Ep ${d.ep}`];
        if (d.mae) parts.push(`Train ${d.mae} meV`);
        if (d.dev) parts.push(`Dev ${d.dev} meV`);
        if (d.el && d.el !== '-') parts.push(`用时 ${d.el}`);
        tip.textContent = parts.join(' · ');
        tip.style.display = 'block';
        const r = box.getBoundingClientRect();
        tip.style.left = Math.min(r.width - tip.offsetWidth - 8, Math.max(8, e.clientX - r.left - 35)) + 'px';
        tip.style.top = Math.max(6, e.clientY - r.top - 36) + 'px';
      });
      pt.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    });
  });
}

let latestSnap = null;
const selectedItemByRun = {};
let selectedOwnerFilter = 'ALL';

function renderRunsSection(d) {
  const runs = d.chou_runs || [];
  const activeRuns = runs.filter(r => r.live_job || r.stage_train_status === 'running');
  const pastRuns = runs.filter(r => !r.live_job && r.stage_train_status !== 'running');
  $('chouSub').textContent = `在跑 ${activeRuns.length} 项 · 历史 ${pastRuns.length} 项 · 点击任意子任务卡片切换查看全部数据`;
  $('tabHistCnt').textContent = `(${pastRuns.length})`;
  let chartIdx = 0;

  const renderRun = (r, isActive) => {
    const live = r.live_job;
    const elSec = live ? parseElapsedSec(live.elapsed) : null;
    const wallTxt = (elSec != null && r.serial_wall_s) ? ` · 墙钟 ${_fmtWall(elSec)} / ${_fmtWall(r.serial_wall_s)}` : '';
    const memTxt = (live && live.used_mem_mb != null) ? ` · 内存 ${fmtNum(live.used_mem_mb)}/${fmtNum(live.lim_mem_mb)} MB` : '';
    const badge = live
      ? `<span class="pill ${live.status === 'Run' ? 'ok' : 'warn'}">${live.status === 'Run' ? '<span class="pulse"></span>' : ''}${esc(live.status)} · Job ${esc(r.job_id)} · ${esc(live.node_or_reason)} · ${esc(live.elapsed)}</span>`
      : `<span class="pill">${r.job_id ? `Job ${esc(r.job_id)} · ` : ''}${esc(r.stage_train_status || r.submission_status || '已完成/历史')}</span>`;

    const items = r.items || [];
    const chosenId = selectedItemByRun[r.run_id];
    const cur = items.find(it => it.item_id === chosenId)
      || items.find(it => it.status === 'RUNNING')
      || items.find(it => it.status === 'COMPLETE' && (it.last_log || it.completion_manifest || (it.curve && it.curve.length)))
      || items.find(it => it.last_log || it.completion_manifest)
      || items[0];
    const curId = cur ? cur.item_id : null;

    const stepsHtml = items.map(it => {
      const icon = it.status === 'COMPLETE' ? '✓' : (it.status === 'RUNNING' ? '●' : '○');
      const ll = it.last_log || {};
      const ep = ll.epoch != null ? `Ep ${ll.epoch}${it.total_passes ? '/' + it.total_passes : ''}` : it.status;
      const dVal = it.best_dev_mae_eV ?? it.dev_mae_eV ?? ll.dev_mae_eV;
      const tVal = ll.train_mae_eV;
      const maeTxt = dVal != null
        ? `Dev ${(dVal * 1000).toFixed(1)}${tVal != null ? ` / Tr ${(tVal * 1000).toFixed(1)}` : ''}`
        : (tVal != null ? `Train ${(tVal * 1000).toFixed(1)} meV` : 'MAE -');
      const sub = it.status === 'COMPLETE' ? fmtDur(it.wall_s) : (it.status === 'RUNNING' ? `ETA ${fmtDur(it.eta_s)}` : '排队等待前置');
      const selCls = it.item_id === curId ? ' selected' : '';
      return `<div class="step ${esc(it.status)}${selCls}" role="button" tabindex="0" data-run="${esc(r.run_id)}" data-item="${esc(it.item_id)}" title="点击切换下方聚焦面板">
        <div style="font-weight:700;display:flex;justify-content:space-between;gap:6px"><span class="mono">${icon} ${esc(it.item_id)}</span><span class="mono" style="font-size:11px;white-space:nowrap">${esc(ep)}</span></div>
        <div style="color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${maeTxt} · ${sub}</div>
      </div>`;
    }).join('');

    let spotHtml = '';
    if (cur) {
      const ll = cur.last_log || {};
      const cm = cur.completion_manifest;
      const hasData = Boolean(cur.last_log || cm || (cur.curve && cur.curve.length));
      const epNum = ll.epoch ?? (cm && cm.completed_passes) ?? null;
      const pctEp = (cur.total_passes && epNum != null) ? Math.min(100, (epNum / cur.total_passes) * 100) : 0;
      const trLatest = fmtMae(ll.train_mae_eV);
      const trBest = cur.best_train_mae_eV != null ? `最优 ${fmtMae(cur.best_train_mae_eV)} @ Ep ${cur.best_train_epoch}` : '-';
      const dvLatest = fmtMae(ll.dev_mae_eV ?? cur.dev_mae_eV);
      const dvBest = cur.best_dev_mae_eV != null
        ? `最优 ${fmtMae(cur.best_dev_mae_eV)} @ Ep ${cur.best_dev_epoch}`
        : (ll.ema_dev_mae_eV != null ? `EMA ${fmtMae(ll.ema_dev_mae_eV)}` : '-');
      const spdTxt = cur.s_per_epoch != null ? `${cur.s_per_epoch} s/ep` : '-';
      const timeVal = cur.status === 'COMPLETE' ? fmtDur(cur.wall_s) : (cur.status === 'RUNNING' ? `${spdTxt} · 剩 ${fmtDur(cur.eta_s)}` : '未开始');
      const timeSub = cur.status === 'COMPLETE' ? `均速 ${spdTxt}` : (cur.status === 'RUNNING' ? `预计完成 ${fmtEtaClock(d.fetched_at_epoch, cur.eta_s)}` : (cur.reason || '等待前置完成'));
      const finishVal = cur.status === 'COMPLETE' ? (cur.finished_jst || '已完成') : (cur.status === 'RUNNING' ? fmtEtaClock(d.fetched_at_epoch, cur.eta_s) : '-');
      const attCnt = (cur.attempts || []).length;
      const finishSub = cur.status === 'COMPLETE'
        ? (attCnt > 1 ? `历经 ${attCnt} 次运行段完成` : '已落盘 completion_manifest')
        : (cur.status === 'RUNNING' ? '按当前速率估算' : '-');
      const stepVal = fmtNum((cm && cm.optimizer_steps) ?? ll.step);
      const sampVal = fmtNum((cm && cm.sample_presentations) ?? ll.sample_presentations);
      const rssVal = cur.peak_rss_bytes ? fmtGb(cur.peak_rss_bytes) : (live && live.used_mem_mb != null ? `${fmtNum(live.used_mem_mb)} MB (Live)` : '-');
      const selInfo = (cm && cm.selection) ? (cm.selection.selected_checkpoint || cm.selection.best_checkpoint || (cm.selection.best_epoch != null ? `Ep ${cm.selection.best_epoch}` : '已归档')) : (ll.lr != null ? `LR ${fmtSci(ll.lr)}` : '-');
      const cid = chartIdx++;

      if (hasData) {
        spotHtml = `
          <div class="spot">
            <div>
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;font-size:13px">
                <span>聚焦子项：<b class="mono">${esc(cur.item_id)}</b> <span class="pill ${cur.status==='COMPLETE'?'ok':(cur.status==='RUNNING'?'info':'')}" style="font-size:11px;padding:1px 8px">${esc(cur.status)}</span></span>
                <span class="mono"><b>Ep ${epNum ?? '-'}${cur.total_passes ? '/' + cur.total_passes : ''}</b> (${pctEp.toFixed(1)}%)</span>
              </div>
              ${cur.total_passes ? `<div class="bar"><div class="bar-ok" style="width:${pctEp.toFixed(1)}%"></div></div>` : ''}
              <div class="spot-metrics">
                <div class="mini-stat"><div class="k">Train MAE (末轮)</div><div class="v" style="color:var(--accent)">${trLatest}</div><div class="s">${trBest}</div></div>
                <div class="mini-stat"><div class="k">Dev MAE (末轮)</div><div class="v" style="color:var(--ok)">${dvLatest}</div><div class="s">${dvBest}</div></div>
                <div class="mini-stat"><div class="k">${cur.status === 'COMPLETE' ? '实际总耗时' : '速率 / 剩余'}</div><div class="v">${timeVal}</div><div class="s">${esc(timeSub)}</div></div>
                <div class="mini-stat"><div class="k">${cur.status === 'COMPLETE' ? '完成时刻' : '预计完成时刻'}</div><div class="v" style="font-size:14px">${esc(finishVal)}</div><div class="s">${esc(finishSub)}</div></div>
                <div class="mini-stat"><div class="k">Step / 样本数</div><div class="v" style="font-size:15px">${stepVal}</div><div class="s">样本 ${sampVal}</div></div>
                <div class="mini-stat"><div class="k">峰值内存 / 选模</div><div class="v" style="font-size:15px">${rssVal}</div><div class="s">${esc(selInfo)}</div></div>
              </div>
              ${(cur.log_tail && cur.log_tail.length) ? `<details><summary>查看最近 ${cur.log_tail.length} 条 train_log.jsonl${cm ? ' 与 completion_manifest' : ''}</summary>
                ${cm ? `<pre style="margin-top:8px;max-height:140px">${esc(JSON.stringify(cm, null, 2))}</pre>` : ''}
                <pre style="margin-top:8px">${esc(cur.log_tail.map(x => JSON.stringify(x)).join('\n'))}</pre>
              </details>` : (cm ? `<details><summary>查看 completion_manifest</summary><pre style="margin-top:8px">${esc(JSON.stringify(cm, null, 2))}</pre></details>` : '')}
            </div>
            ${buildChartHtml(cur, cid)}
          </div>`;
      } else {
        spotHtml = `<div class="spot" style="grid-template-columns:1fr"><div style="color:var(--muted);font-size:13px">聚焦子项：<b class="mono" style="color:var(--text)">${esc(cur.item_id)}</b> · 状态 <b>${esc(cur.status)}</b>${cur.total_passes ? ` · 计划 ${cur.total_passes} Epochs` : ''} · 尚未产生训练日志（排队或等待前置完成）。</div></div>`;
      }
    }

    const forceOpenHist = Boolean(!isActive && chosenId);
    return `<div class="run-card" data-run-card="${esc(r.run_id)}">
      <div class="run-head">
        <div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><b class="mono" style="font-size:16px">${esc(r.run_id)}</b> <span class="pill info">${esc(r.mode)}</span></div>
          <div class="run-meta"><span>数据集: ${esc(Array.isArray(r.dataset_id) ? r.dataset_id[0] + ' 等' : (r.dataset_id || '-'))}${memTxt}${wallTxt}</span></div>
        </div>
        <div>${badge}</div>
      </div>
      <div class="chain">${stepsHtml}</div>
      ${isActive ? spotHtml : `<details${forceOpenHist ? ' open' : ''}><summary>查看子项完整数据、日志与曲线 (${esc(curId || '-')})</summary><div style="margin-top:10px">${spotHtml}</div></details>`}
    </div>`;
  };

  $('chouRuns').innerHTML = activeRuns.length ? activeRuns.map(r => renderRun(r, true)).join('')
    : '<div class="card" style="color:var(--muted)">当前没有运行中的 chou V4 任务；请切换至「历史 V4 记录」查看已结束实验。</div>';
  $('pastRuns').innerHTML = pastRuns.length ? pastRuns.map(r => renderRun(r, false)).join('')
    : '<div class="card" style="color:var(--muted)">暂无历史记录。</div>';

  document.querySelectorAll('.step[data-run][data-item]').forEach(el => {
    const activate = () => {
      selectedItemByRun[el.dataset.run] = el.dataset.item;
      renderRunsSection(latestSnap);
    };
    el.onclick = activate;
    el.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); } };
  });
  bindChartTooltips();
}

function renderJobsSection(d) {
  const jobs = d.jobs || [];
  $('tabJobsCnt').textContent = `(${jobs.length})`;
  const byOwner = {};
  jobs.forEach(j => {
    const ow = j.owner || (j.is_chou ? 'chou' : 'sm2');
    if (!byOwner[ow]) byOwner[ow] = {owner: ow, isChou: Boolean(j.is_chou), jobs: 0, gpus: 0, cpus: 0};
    byOwner[ow].jobs += 1;
    byOwner[ow].gpus += Number(j.gpus || 0);
    byOwner[ow].cpus += Number(j.cpus || 0);
  });
  const owners = Object.values(byOwner).sort((a, b) => (b.isChou - a.isChou) || (b.gpus - a.gpus) || (b.jobs - a.jobs));
  if (selectedOwnerFilter !== 'ALL' && !byOwner[selectedOwnerFilter]) selectedOwnerFilter = 'ALL';

  $('ownerBar').innerHTML = owners.length ? (
    `<span style="font-size:12px;color:var(--muted)">按文件夹归属筛选：</span>` +
    `<button class="owner-chip${selectedOwnerFilter==='ALL'?' active':''}" data-owner="ALL">全部 (${jobs.length})</button>` +
    owners.map(o => `<button class="owner-chip${selectedOwnerFilter===o.owner?' active':''}" data-owner="${esc(o.owner)}">${o.isChou ? '本人 ' : ''}${esc(o.owner)} · ${o.jobs} 作业${o.gpus ? ` (${o.gpus} GPU)` : ` (${o.cpus} CPU)`}</button>`).join('')
  ) : '';

  document.querySelectorAll('.owner-chip[data-owner]').forEach(btn => {
    btn.onclick = () => { selectedOwnerFilter = btn.dataset.owner; renderJobsSection(latestSnap); };
  });

  const shown = selectedOwnerFilter === 'ALL' ? jobs : jobs.filter(j => (j.owner || (j.is_chou ? 'chou' : 'sm2')) === selectedOwnerFilter);
  $('jobsTbl').innerHTML = `<thead><tr><th>Job ID</th><th>名称</th><th>状态</th><th>资源</th><th>已运行</th><th>节点 / 原因</th><th>内存使用</th><th>归属 (Workdir)</th><th>Workdir</th></tr></thead><tbody>` +
    (shown.length ? shown.map(j => {
      const cls = j.is_chou ? 'chou-row' : 'other-row';
      const res = j.gpus ? `<span class="pill info">${j.cpus} CPU + ${j.gpus} GPU</span>` : `${j.cpus} CPU`;
      const memPct = (j.used_mem_mb != null && j.lim_mem_mb) ? Math.min(100, (j.used_mem_mb / j.lim_mem_mb) * 100) : 0;
      const mem = (j.used_mem_mb != null || j.lim_mem_mb != null)
        ? `<div>${fmtNum(j.used_mem_mb)} / ${fmtNum(j.lim_mem_mb)} MB</div><div class="bar" style="height:5px;margin-top:4px;max-width:130px"><div class="bar-seg-a" style="width:${memPct.toFixed(0)}%"></div></div>` : '-';
      const ow = j.owner || (j.is_chou ? 'chou' : 'sm2');
      const badge = j.is_chou ? `<span class="pill ok">本人 ${esc(ow)}</span>` : `<span class="pill info">${esc(ow)}</span>`;
      return `<tr class="${cls}"><td class="mono">${esc(j.job_id)}</td><td class="mono">${esc(j.name)}</td><td><span class="pill ${j.status==='Run'?'ok':'warn'}">${esc(j.status)}</span></td><td>${res}</td><td class="mono">${esc(j.elapsed)}</td><td class="mono">${esc(j.node_or_reason)}</td><td>${mem}</td><td>${badge}</td><td class="mono" style="font-size:12px">${esc(j.workdir)}</td></tr>`;
    }).join('') : `<tr><td colspan="9" style="color:var(--muted)">当前过滤条件下无在跑或排队作业</td></tr>`) + `</tbody>`;
}

function renderWaitPredictor(d) {
  if (!d) return;
  const reqGpu = Number($('predGpu').value);
  const reqCpu = Number($('predCpu').value);
  const wallH = Number($('predWall').value);
  const isOverWeek = wallH > 168;

  const s = d.summary || {};
  const qs = d.queue_status || {};
  const gPool = isOverWeek ? (s.gpu_over_week || {}) : (s.gpu_within_week || {});
  const cpuPool = isOverWeek
    ? ((qs.over_week || []).find(r => r.job_type === 'cpu') || {})
    : ((qs.within_week || []).find(r => r.job_type === 'cpu') || {});
  const pool = reqGpu > 0 ? gPool : cpuPool;

  const freeG = gPool.free_gpus ?? 0, waitJobs = pool.jobs_waiting ?? 0, reqPoolG = gPool.req_gpus ?? 0;
  const freeC = pool.free_cores ?? 0, reqPoolC = pool.req_cores ?? 0;

  const ugRows = (d.user_group_stat && d.user_group_stat.rows) || [];
  const ugMap = Object.fromEntries(ugRows.map(r => [r.metric, r]));
  const gGpu = (ugMap.GPUs && ugMap.GPUs.group) || {};
  const gRun = gGpu.run ?? 0, gLim = gGpu.limit ?? 6, gHeadroom = Math.max(0, gLim - gRun), gQueue = gGpu.queue ?? 0;

  const nowEpoch = d.fetched_at_epoch || (Date.now() / 1000);
  const maintEpoch = nextFirstMondayMaintEpoch(nowEpoch);
  const hoursToMaint = maintEpoch ? (maintEpoch - nowEpoch) / 3600 : 9999;
  const maintJst = maintEpoch ? new Date(maintEpoch * 1000).toLocaleString('zh-CN', {timeZone:'Asia/Tokyo', month:'2-digit', day:'2-digit', hour12:false, hour:'2-digit', minute:'2-digit'}) : '-';

  const hist = d.queue_wait_history || [];
  const histGpu = hist.filter(h => (h.gpus || 0) > 0);
  const medWait = arr => {
    if (!arr.length) return null;
    const sorted = [...arr].map(x => x.wait_s).sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  const initWaits = histGpu.filter(h => h.kind === 'initial');
  const resWaits = histGpu.filter(h => h.kind === 'resume');
  const medInit = medWait(initWaits), medRes = medWait(resWaits);

  let minH = 0.05, maxH = 0.5, cls = '', badgeTxt = '建议提交', tipTxt = '';
  const pressure = reqGpu > 0 ? (reqPoolG / Math.max(1, freeG)) : (reqPoolC / Math.max(1, freeC));

  if (reqGpu > gLim) {
    cls = 'bad'; minH = 999; maxH = 999;
    badgeTxt = `超出 ${d.user_group_stat?.group||'ew6'} 组并发上限 (${gLim} 卡)`;
    tipTxt = `ew6 组最多同时跑 ${gLim} 张 A100，无法单次申请超过 ${gLim} 卡。`;
  } else if (reqGpu > 0 && reqGpu > gHeadroom) {
    cls = 'warn';
    const needRelease = reqGpu - gHeadroom;
    const chouRunning = (d.chou_runs || []).filter(r => r.live_job && r.live_job.status === 'Run');
    const soonestChouEtaH = chouRunning.reduce((best, r) => {
      const it = (r.items || []).find(x => x.status === 'RUNNING');
      if (it && it.eta_s != null && r.items.filter(x => x.status !== 'COMPLETE').length <= 1) {
        return Math.min(best, it.eta_s / 3600);
      }
      return best;
    }, 99);
    minH = Math.max(1, soonestChouEtaH < 50 ? Math.max(0.5, soonestChouEtaH * 0.8) : 2 * needRelease);
    maxH = Math.max(minH + 4, 8 * needRelease + gQueue * 3);
    badgeTxt = `需等组配额释放 (+${needRelease} 卡)`;
    tipTxt = `ew6 组当前已用 ${gRun}/${gLim} 卡（仅剩 ${gHeadroom} 卡余量）。必须先等组内至少释放 ${needRelease} 卡后才会进入调度。`;
  } else if (wallH > hoursToMaint - 1) {
    cls = 'warn';
    minH = Math.max(2, hoursToMaint + 4);
    maxH = hoursToMaint + 16;
    badgeTxt = '受月度维护窗口拦截风险';
    tipTxt = `距下次首周一维护 (${maintJst} JST) 仅剩 ${hoursToMaint.toFixed(1)}h，${wallH}h 作业大概率跨维护，PBS 可能推迟到维护结束后开跑。建议将单次链缩至 <= ${Math.max(1, Math.floor(hoursToMaint - 2))}h。`;
  } else if (reqGpu === 0) {
    if (freeC >= reqCpu && pressure < 1.2) {
      cls = ''; minH = 0.02; maxH = 0.3; badgeTxt = '宽松（可快速调度）';
      tipTxt = `当前空闲 CPU ${fmtNum(freeC)} 核，申请池 ${fmtNum(reqPoolC)} 核，纯 CPU 任务较易直接上机。`;
    } else {
      cls = 'warn'; minH = 0.5; maxH = 6; badgeTxt = '需常规排队';
      tipTxt = `CPU 申请核数较多或队列偏满 (${waitJobs} 作业排队)，可能需等待部分作业释放节点核。`;
    }
  } else {
    const backfillScore =
      (freeG >= reqGpu ? 2 : -3) +
      (freeC >= reqCpu ? 1 : -2) +
      (wallH <= 6 ? 3 : wallH <= 16 ? 1.5 : wallH <= 24 ? 0 : -2.5) +
      (reqGpu === 1 ? 2 : reqGpu === 2 ? 0.5 : -2) +
      (pressure <= 1.0 ? 1 : pressure <= 2.0 ? 0 : -1.5) +
      (gQueue > 0 ? -1.5 : 0);

    if (backfillScore >= 5) {
      cls = ''; minH = 0.03; maxH = 0.75;
      badgeTxt = '极佳窗口 · 高概率回填秒排';
      tipTxt = `空闲 ${freeG} 卡 >= 申请 ${reqGpu} 卡，且 ${wallH}h 易于利用 PBS Backfill（回填）空隙上卡；组内尚余 ${gHeadroom} 卡。`;
    } else if (backfillScore >= 2) {
      cls = ''; minH = 0.5; maxH = 5;
      badgeTxt = '较好窗口 · 短中排 (有望回填)';
      tipTxt = `虽然队列里有 ${waitJobs} 个作业共申请 ${reqPoolG} 卡，但 ${reqGpu} 卡/${wallH}h 仍有机会趁大作业凑整卡间隙回填插空。`;
    } else if (backfillScore >= -1) {
      cls = 'warn'; minH = 3; maxH = 18;
      badgeTxt = '常规竞争 · 需等队列消耗';
      tipTxt = `${reqGpu} 卡/${wallH}h 较难回填到零散间隙中，通常需排队等待前方 ${reqPoolG} 张在等 A100 逐步消化。`;
    } else {
      cls = 'bad'; minH = 12; maxH = 48;
      badgeTxt = '高阻窗口 · 预计排队较久';
      tipTxt = isOverWeek
        ? `>1周池仅有 ${freeG} 空闲 A100 但已有 ${waitJobs} 作业 (${reqPoolG} 卡) 排队，强烈建议拆为 16h serial_chain 提交至 1周内队列。`
        : `${reqGpu} 卡或 ${wallH}h 申请较难吃到短空隙回填（当前队列 ${waitJobs} 作业共申请 ${reqPoolG} 卡）；建议拆为 <=16h serial_chain + 1~2 卡以显著缩短等卡时间。`;
    }
  }

  const fmtSpan = h => h >= 999 ? '不可行' : (h < 1 ? `${Math.max(1, Math.round(h * 60))} 分钟` : `${h.toFixed(h < 10 ? 1 : 0)} 小时`);
  const rangeStr = maxH >= 999 ? '超出组上限，无法调度' : `${fmtSpan(minH)} ~ ${fmtSpan(maxH)}`;
  const startClock = maxH >= 999 ? '-' : fmtJstClockRange(nowEpoch, minH, maxH);

  const resEl = $('predResult');
  resEl.className = 'pred-res' + (cls ? ' ' + cls : '');
  resEl.innerHTML = `
    <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px">
      <span style="font-size:18px;font-weight:700">预估等待：${esc(rangeStr)}</span>
      <span class="pill ${cls==='bad'?'bad':(cls==='warn'?'warn':'ok')}">${esc(badgeTxt)}</span>
    </div>
    <div style="font-size:12.5px;margin-top:6px;color:var(--text)">预计开工窗口：<b class="mono">${esc(startClock)}</b></div>
    <div style="font-size:12px;color:var(--muted);margin-top:6px">${esc(tipTxt)}</div>`;

  $('predFactors').innerHTML = `
    <div class="pred-f"><div style="color:var(--muted)">① ew6 组并发余量</div><div style="font-weight:700;margin-top:2px">已跑 ${gRun}/${gLim} 卡 · 组排队 ${gQueue} · 剩余可并发 ${gHeadroom} 卡</div></div>
    <div class="pred-f"><div style="color:var(--muted)">② ${isOverWeek ? '>1周' : '1周内'} 池供需与碎片</div><div style="font-weight:700;margin-top:2px">空闲 ${freeG} 卡 (${fmtNum(freeC)} 核) / 等待 ${waitJobs} 作业 (${reqPoolG} 卡)</div></div>
    <div class="pred-f"><div style="color:var(--muted)">③ 月度维护窗口校验</div><div style="font-weight:700;margin-top:2px">下次 ${esc(maintJst)} JST · 距今 ${hoursToMaint.toFixed(0)} 小时</div></div>
    <div class="pred-f"><div style="color:var(--muted)">④ chou 历史实测参考</div><div style="font-weight:700;margin-top:2px">首次中位 ${medInit!=null?fmtDur(medInit):'-'} (${initWaits.length}次) · 续跑中位 ${medRes!=null?fmtDur(medRes):'-'} (${resWaits.length}次)</div></div>`;

  $('waitHistSummary').textContent = `查看 chou 历史排队实测记录 (${hist.length} 条：由 staging_manifest/submission 与实际开跑时刻反推)`;
  $('waitHistTbl').innerHTML = `<thead><tr><th>Run ID</th><th>类型</th><th>Job ID</th><th>资源 / 墙钟</th><th>提交时刻 (JST)</th><th>上卡开跑 (JST)</th><th>实测排队</th></tr></thead><tbody>` +
    (hist.length ? hist.map(h => `<tr><td class="mono">${esc(h.run_id)}</td><td><span class="pill ${h.kind==='resume'?'info':''}">${h.kind==='resume'?'续跑':'首次'}</span></td><td class="mono">${esc(h.job_id||'-')}</td><td class="mono">${h.gpus ?? '-'}g ${h.cpus ?? '-'}c / ${esc(h.walltime||'-')}</td><td class="mono">${esc(h.submitted_jst||'-')}</td><td class="mono">${esc(h.started_jst||'-')}</td><td><b>${esc(fmtDur(h.wait_s))}</b></td></tr>`).join('')
      : `<tr><td colspan="7" style="color:var(--muted)">暂无足够历史排队时间记录</td></tr>`) + `</tbody>`;
}

function render(d) {
  if (d.public) {
    $('connPill').textContent = !d.available ? 'UNKNOWN / 从未成功或快照损坏' : d.stale ? 'STALE / 数据已过期' : '快照正常';
    $('connPill').className = 'pill ' + (d.stale ? 'bad' : 'ok');
    $('metaLine').textContent = '最后成功: ' + (d.last_success_epoch ? new Date(d.last_success_epoch * 1000).toISOString() : '无') + ' | 故障来源: ' + (d.failure || '无');
    $('errBox').style.display = d.stale ? 'block' : 'none';
    $('errBox').textContent = '旧数据不可视为当前队列；刷新只读取快照，不连接 IMS。';
    $('heroBar').style.display = 'block';
    $('heroBar').textContent = !d.available ? '队列未知' : d.jobs.length ? `自己的作业: ${d.jobs.length}` : '最后成功采集时队列为空';
    $('kpiGrid').innerHTML = !d.available ? '' : d.chou_runs.map(r => `<div class="card">Job ${esc(r.job_id || '-')}<br>${r.items.map(i => `${esc(i.status)} · Epoch ${i.epoch ?? '-'} / ${i.total_passes ?? '-'} · ETA ${fmtDur(i.eta_s)}`).join('<br>')}</div>`).join('');
    $('jobsTbl').innerHTML = !d.available ? '' : d.jobs.map(j => `<tr><td>${esc(j.job_id)}</td><td>${esc(j.status)}</td><td>${j.gpus ?? '-'} A100</td></tr>`).join('');
    $('queueTbl').textContent = '';
    $('ugTbl').textContent = !d.available ? '' : '配额汇总: ' + JSON.stringify(d.points);
    document.querySelectorAll('details').forEach(e => e.style.display = 'none');
    $('predResult').closest('.pred-card')?.style.setProperty('display', 'none');
    $('predResult').parentElement.style.display = 'none';
    $('chouRuns').textContent = '';
    return;
  }
  if (!d.fetched_at_epoch) {
    $('connPill').textContent = 'UNKNOWN';
    $('errBox').style.display = 'block';
    $('errBox').textContent = d.error || '尚无成功快照';
    return;
  }
  latestSnap = d;
  const errBox = $('errBox');
  if (d.error) { errBox.style.display = 'block'; errBox.textContent = '拉取提示: ' + d.error; }
  else { errBox.style.display = 'none'; }

  const pill = $('connPill');
  pill.className = 'pill ' + (d.error ? 'warn' : 'ok');
  pill.innerHTML = `<span class="pulse"></span>${d.cached ? '缓存命中' : '实时在线'}`;
  $('metaLine').textContent = `本地刷新 ${d.fetched_at_jst || '-'} · 调度器 ${d.scheduler_timestamp || '-'} · SSH 耗时 ${d.ssh_latency_ms ?? '-'} ms · 边界 ${d.remote_root || '-'}`;

  const s = d.summary || {};
  const g1 = s.gpu_within_week || {}, g2 = s.gpu_over_week || {};
  const ugRows = (d.user_group_stat && d.user_group_stat.rows) || [];
  const ugMap = Object.fromEntries(ugRows.map(r => [r.metric, r]));
  const gGpu = (ugMap.GPUs && ugMap.GPUs.group) || {}, uGpu = (ugMap.GPUs && ugMap.GPUs.user) || {};
  const pts = d.points || {}, gp = pts.group || {}, up = pts.user || {};

  const free1w = g1.free_gpus ?? 0, wait1w = g1.jobs_waiting ?? 0, req1w = g1.req_gpus ?? 0;
  const gRun = gGpu.run ?? 0, gLim = gGpu.limit ?? 6, gHeadroom = Math.max(0, gLim - gRun);
  const runs = d.chou_runs || [];
  const activeRuns = runs.filter(r => r.live_job || r.stage_train_status === 'running');

  let qVerdict = free1w > 0
    ? `1周内物理空闲 ${free1w} 卡，队列有 ${wait1w} 作业申请 ${req1w} 卡（因节点碎片/预留不等于全可秒排，短作业可吃回填）`
    : '当前 1 周内无空闲 A100，预计排队较久';
  if (gHeadroom === 0) qVerdict = `ew6 组已满并发 (${gRun}/${gLim} 卡)，新任务需先等组内释放卡位`;

  const hero = $('heroBar');
  hero.style.display = 'flex';
  hero.className = 'hero' + ((free1w === 0 || gHeadroom === 0) ? ' tight' : '');
  const curSummary = activeRuns.length ? activeRuns.map(r => {
    const it = (r.items || []).find(x => x.status === 'RUNNING');
    const ep = it && it.last_log ? ` (${it.item_id} Ep ${it.last_log.epoch}${it.total_passes ? '/' + it.total_passes : ''})` : '';
    return `${r.run_id}${ep}`;
  }).join('；') : '当前无在跑 chou 任务';
  hero.innerHTML = `<div><b>排队结论：</b>${esc(qVerdict)}；<b>${esc(d.user_group_stat?.group||'ew6')}</b> 组还可并跑 <b>${gHeadroom}</b> 卡</div>
                    <div class="pill info">chou 在跑：${esc(curSummary)}</div>`;

  const dotsCount = Math.min(16, Math.max(8, free1w + req1w));
  const dotsHtml = Array.from({length: dotsCount}, (_, i) => `<span class="slot-dot ${i < free1w ? '' : 'dim'}" title="${i < free1w ? '空闲 A100 卡槽' : '队列占用/申请中'}"></span>`).join('');

  const ringPct = gLim > 0 ? Math.min(1, gRun / gLim) : 0;
  const circ = 2 * Math.PI * 24;
  const dash = (circ * ringPct).toFixed(1);

  const settled = gp.ended_job ?? gp.total_use ?? 0;
  const runningPt = gp.running_job ?? 0;
  const limitPt = gp.limit || 1;
  const pctEnded = Math.min(100, (settled / limitPt) * 100);
  const pctRun = Math.min(100 - pctEnded, (runningPt / limitPt) * 100);

  $('kpiGrid').innerHTML = `
    <div class="card">
      <div class="kpi-top"><span class="kpi-title">A100 空闲卡数 (Queue H)</span><span class="pill ok">1周内</span></div>
      <div class="kpi-val" style="color:var(--ok)">${g1.free_gpus ?? '-'} <span style="font-size:14px;font-weight:500;color:var(--muted)">张空闲</span></div>
      <div class="kpi-sub">空闲 CPU ${fmtNum(g1.free_cores)} 核 · 超 1 周空闲 ${g2.free_gpus ?? '-'} 卡</div>
      <div class="slots">${dotsHtml}</div>
    </div>
    <div class="card">
      <div class="kpi-top"><span class="kpi-title">A100 排队队列 (Queue H)</span><span class="pill ${wait1w>10?'warn':'info'}">${wait1w} 作业等待</span></div>
      <div class="kpi-val">${g1.req_gpus ?? '-'} <span style="font-size:14px;font-weight:500;color:var(--muted)">张申请中</span></div>
      <div class="kpi-sub">1 周内申请 ${fmtNum(g1.req_cores)} 核 · 超 1 周排队 ${g2.jobs_waiting ?? 0} 作业</div>
    </div>
    <div class="card donut-wrap">
      <div>
        <div class="kpi-title">${esc(d.user_group_stat?.group||'ew6')} 组 GPU 并发</div>
        <div class="kpi-val">${gRun} <span style="font-size:16px;color:var(--muted)">/ ${gLim}</span></div>
        <div class="kpi-sub">组排队 ${gGpu.queue ?? 0} · sm2 占 ${uGpu.run ?? 0} 卡 (chou ${s.chou_running_jobs ?? 0})</div>
      </div>
      <div class="donut">
        <svg width="64" height="64" viewBox="0 0 64 64">
          <circle cx="32" cy="32" r="24" fill="none" stroke="var(--panel3)" stroke-width="7"/>
          <circle cx="32" cy="32" r="24" fill="none" stroke="var(--accent)" stroke-width="7" stroke-linecap="round" stroke-dasharray="${dash} ${circ.toFixed(1)}"/>
        </svg>
      </div>
    </div>
    <div class="card">
      <div class="kpi-top"><span class="kpi-title">${esc(gp.name||'ew6')} 点数余量 (showlim -c)</span><span class="pill">${(pctEnded+pctRun).toFixed(1)}% 已用</span></div>
      <div class="kpi-val">${fmtNum(gp.remaining)} <span style="font-size:13px;color:var(--muted);font-weight:500">剩余 pt</span></div>
      <div class="kpi-sub">组用 ${fmtNum(gp.total_use)} / ${fmtNum(gp.limit)} · sm2 ${fmtNum(up.total_use)} (运行 ${fmtNum(up.running_job)})</div>
      <div class="bar"><div class="bar-seg-a" style="width:${pctEnded.toFixed(2)}%"></div><div class="bar-seg-b" style="width:${pctRun.toFixed(2)}%"></div></div>
    </div>`;

  renderRunsSection(d);
  renderJobsSection(d);
  renderWaitPredictor(d);

  const qs = d.queue_status || {};
  const qRows = [
    ...(qs.within_week || []).map(r => ({...r, grp: '1 周内'})),
    ...(qs.over_week || []).map(r => ({...r, grp: '超 1 周'})),
    ...(qs.special || []).map(r => ({...r, grp: '特殊'})),
  ];
  $('queueTbl').innerHTML = `<thead><tr><th>时段</th><th>类型</th><th>空闲核 (GPU)</th><th>排队作业</th><th>申请核 (GPU)</th></tr></thead><tbody>` +
    qRows.map(r => `<tr${r.job_type==='gpu'?' class="chou-row"':''}><td>${esc(r.grp)}</td><td class="mono"><b>${esc(r.job_type)}</b></td><td>${fmtNum(r.free_cores)}${r.free_gpus!=null?` <b>(${r.free_gpus} GPU)</b>`:''}</td><td>${r.jobs_waiting}</td><td>${fmtNum(r.req_cores)}${r.req_gpus!=null?` <b>(${r.req_gpus} GPU)</b>`:''}</td></tr>`).join('') + `</tbody>`;

  $('ugTbl').innerHTML = `<thead><tr><th>资源项</th><th>${esc(d.user_group_stat?.user||'sm2')} (Run/Q/H/Lim)</th><th>${esc(d.user_group_stat?.group||'ew6')} (Run/Q/H/Lim)</th></tr></thead><tbody>` +
    ugRows.map(r => {
      const f = x => `${x.run} / ${x.queue} / ${x.hold} / ${x.limit ?? '-'}`;
      return `<tr><td class="mono"><b>${esc(r.metric)}</b></td><td class="mono">${f(r.user)}</td><td class="mono">${f(r.group)}</td></tr>`;
    }).join('') + `</tbody>`;

  const raw = d.raw || {};
  $('rawBox').textContent = ['--- jobinfo -c ---', raw.jobinfo_c, '--- jobinfo -m ---', raw.jobinfo_m, '--- jobinfo -w ---', raw.jobinfo_w, '--- jobinfo -s ---', raw.jobinfo_s, '--- showlim -c ---', raw.showlim_c].join('\n');
}

function _fmtWall(sec) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return `${h}h${String(m).padStart(2,'0')}m`;
}

async function pull(force) {
  const btn = $('refreshBtn');
  if (force && Date.now() < cooldownUntil) {
    const wait = Math.ceil((cooldownUntil - Date.now()) / 1000);
    showToast(`冷却保护中：请 ${wait} 秒后再刷新登录节点`);
    return;
  }
  btn.disabled = true; btn.textContent = '拉取中...';
  try {
    const res = await fetch('/api/status' + (force ? '?refresh=1' : ''));
    const data = await res.json();
    if (!data.cached && data.min_interval_s) cooldownUntil = Date.now() + data.min_interval_s * 1000;
    else if (data.cooldown_remaining_s) cooldownUntil = Date.now() + data.cooldown_remaining_s * 1000;
    render(data);
    if (force && !data.public && !data.cached) showToast('已从分子研刷新最新状态');
  } catch (e) {
    $('connPill').textContent = 'STALE / 网页读取失败';
    $('connPill').className = 'pill bad';
    $('errBox').style.display = 'block';
    $('errBox').textContent = '本机服务请求失败: ' + e;
  } finally {
    btn.disabled = false; btn.textContent = '立即刷新';
  }
}

$('copyRawBtn').onclick = async () => {
  try { await navigator.clipboard.writeText($('rawBox').textContent); showToast('已复制原始调度输出'); }
  catch { showToast('复制失败，请手动选中复制'); }
};

$('predGpu').onchange = () => {
  const g = Number($('predGpu').value);
  const rec = {0:'16', 1:'16', 2:'32', 4:'64'}[g];
  if (rec) $('predCpu').value = rec;
  renderWaitPredictor(latestSnap);
};
$('predCpu').onchange = () => renderWaitPredictor(latestSnap);
$('predWall').onchange = () => renderWaitPredictor(latestSnap);

function setupTimer() {
  clearInterval(timerId); clearInterval(tickId);
  const sec = Number($('autoSec').value);
  if (!sec) { $('countdown').textContent = ''; return; }
  nextAt = Date.now() + sec * 1000;
  timerId = setInterval(() => { nextAt = Date.now() + sec * 1000; pull(true); }, sec * 1000);
  tickId = setInterval(() => {
    const rem = Math.max(0, Math.ceil((nextAt - Date.now()) / 1000));
    $('countdown').textContent = `${rem}s`;
  }, 1000);
}

$('refreshBtn').onclick = () => pull(true);
$('autoSec').onchange = setupTimer;
pull(false);
</script>
</body>
</html>
"""


def make_handler(state: MonitorState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                page = getattr(state, "page", HTML_PAGE)
                if isinstance(state, SnapshotState):
                    # Keep the same page; replace account labels consistently in JS and DOM.
                    page = page.replace("chou", "own").replace("sm2", "account").replace("ew6", "group")
                    page = page.replace("d.own_runs", "d.chou_runs")
                    page = page.replace("pull(false);", "pull(false); setInterval(() => pull(false), 30000);")
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", page.encode("utf-8"))
                return
            if parsed.path == "/api/status":
                qs = parse_qs(parsed.query)
                force = qs.get("refresh", ["0"])[0] in {"1", "true", "yes"}
                payload = state.get(force=force)
                self._send(HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                return
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not Found")

    return Handler


def serve_web(state: MonitorState, *, port: int, open_browser: bool, explicit_port: bool) -> None:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    except OSError:
        if explicit_port:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"IMS dashboard listening on {url} (127.0.0.1 only, cooldown {state.min_interval:.0f}s; Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.", flush=True)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", type=Path, help="Path to IMS SSH private key")
    parser.add_argument("--remote-root", default=REMOTE_BOUNDARY)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL,
                        help="Minimum cooldown seconds between live SSH queries in web/watch mode")
    parser.add_argument("--json", action="store_true", help="Print structured snapshot JSON")
    parser.add_argument("--raw", action="store_true", help="Include raw scheduler output in terminal view")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="Poll periodically in terminal (clamped >= min-interval)")
    parser.add_argument("--web", action="store_true", help="Start local 127.0.0.1 web dashboard")
    parser.add_argument("--port", type=int, default=None, help=f"Web port (default {DEFAULT_WEB_PORT})")
    parser.add_argument("--open", action="store_true", help="Open browser automatically with --web")
    parser.add_argument("--snapshot", type=Path, help="Serve only sanitized snapshot; never SSH")
    parser.add_argument("--collect", type=Path, help="Collect sanitized snapshots independently of HTTP")
    parser.add_argument("--interval", type=float, default=90, help="Collector interval 60..120 seconds; failure backoff up to 900")
    args = parser.parse_args(argv)

    if args.snapshot:
        if not args.web or args.collect:
            parser.error("--snapshot requires --web and cannot collect")
        serve_web(SnapshotState(args.snapshot), port=args.port or DEFAULT_WEB_PORT,
                  open_browser=args.open, explicit_port=args.port is not None)
        return 0
    key_path = resolve_ssh_key(args.key)
    if args.collect:
        if args.web or args.watch or not 60 <= args.interval <= 120:
            parser.error("--collect needs interval 60..120 and cannot serve/watch")
        delay = args.interval
        while True:
            ok = collect_once(args.collect, host=args.host, port=args.ssh_port, user=args.user,
                              key_path=key_path, remote_root=args.remote_root)
            delay = args.interval if ok else min(900, delay * 2)
            time.sleep(delay)
    state = MonitorState(
        host=args.host, port=args.ssh_port, user=args.user,
        key_path=key_path, remote_root=args.remote_root, min_interval=args.min_interval,
    )

    if args.web:
        serve_web(state, port=args.port or DEFAULT_WEB_PORT, open_browser=args.open, explicit_port=args.port is not None)
        return 0

    if args.watch is not None:
        interval = max(state.min_interval, float(args.watch))
        try:
            while True:
                snap = fetch_snapshot(host=args.host, port=args.ssh_port, user=args.user,
                                      key_path=key_path, remote_root=args.remote_root)
                os.system("cls" if os.name == "nt" else "clear")
                print(render_terminal(snap))
                print(f"\n(Refreshing every {interval:.0f}s; Ctrl+C to stop)")
                time.sleep(interval)
        except KeyboardInterrupt:
            return 0

    snap = fetch_snapshot(host=args.host, port=args.ssh_port, user=args.user,
                          key_path=key_path, remote_root=args.remote_root)
    if args.json:
        print(json.dumps(snap, indent=2, ensure_ascii=False))
    else:
        print(render_terminal(snap))
        if args.raw:
            for k, v in snap["raw"].items():
                print(f"\n=== {k} ===\n{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

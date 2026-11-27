"""Read-only Kaggle quota and selected/live kernel dashboard."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import threading
import time
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "platforms/ims")]
from check_queue import SnapshotState, atomic_snapshot, make_handler, ThreadingHTTPServer  # noqa: E402

JST = timezone(timedelta(hours=9))

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kaggle 看板</title><style>body{font:16px system-ui;background:#101923;color:#edf5fa;max-width:55rem;margin:5vh auto;padding:1rem}
section{background:#1d3040;padding:1rem;margin:1rem 0;border-radius:.6rem}button{padding:.5rem}#dataAge{color:#ffcf79}</style>
<h1>Kaggle 看板</h1>
<button id="refreshSnapshot">刷新快照</button>
<button id="copySummary">复制摘要</button><span id="summaryFeedback" role="status"></span>
<!-- dashboard-language -->
<p id="dataAge">尚无快照</p>
<div id="accounts"></div>
<h2>任务</h2>
<label>搜索 <input id="kernelSearch" type="search" placeholder="任务编号或状态"></label>
<div id="kernels"></div>
<script>
let snapshot;
function addSnapshotRow(container, text) {
  const row = document.createElement('section');
  row.textContent = text;
  container.append(row);
}
function renderSnapshotKernels() {
  const container = document.getElementById('kernels');
  const query = document.getElementById('kernelSearch').value;
  const jobs = snapshot?.kernels || [];
  const shown = jobs.filter(k => dashboardMatches(query, k.label, k.status));
  container.replaceChildren();
  for (const k of shown) addSnapshotRow(container, k.label + ' · ' + k.status);
  if (!shown.length) addSnapshotRow(container, jobs.length ? '没有匹配任务' : '暂无任务状态');
}
async function refresh() {
  const button = document.getElementById('refreshSnapshot');
  if (button.disabled) return;
  button.disabled = true;
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    snapshot = await response.json();
    const container = document.getElementById('accounts');
    container.replaceChildren();
    for (const r of snapshot.accounts) {
      const quota = r.at ? '剩余 ' + r.remaining_hours + ' / ' + r.total_hours + ' GPU 小时，已用 ' + r.used_hours +
        '，预留 ' + r.reserved_hours + '，重置 ' + r.refresh_time_utc + ' · ' + new Date(r.at * 1000).toLocaleString()
        : '尚无配额快照';
      addSnapshotRow(container, r.account + ' · ' + quota + (r.error ? ' · 配额读取失败' : ''));
    }
    dashboardObserved(snapshot.at, 300, snapshot.partial ? '部分数据不可用' : '');
    renderSnapshotKernels();
  } catch {
    dashboardFetchFailed();
  } finally {
    button.disabled = false;
  }
}
document.getElementById('kernelSearch').oninput = renderSnapshotKernels;
document.getElementById('refreshSnapshot').onclick = refresh;
bindDashboardSummary(() => snapshot?.at ? kaggleSummary(snapshot.accounts, snapshot.kernels.map(k => k.status)) : []);
refresh();setInterval(refresh,30000);
</script></html>"""

KAGGLE_SUMMARY = r"""
function kaggleSummary(accounts, statuses) {
  const lines = ['Kaggle'];
  accounts.forEach((a, index) => {
    if (Number.isFinite(a.remaining_hours)) {
      lines.push(`账号 ${index + 1} · 剩余 ${a.remaining_hours} GPU 小时${a.error ? '（旧数据）' : ''}`);
    } else {
      lines.push(`账号 ${index + 1} · 配额不可用`);
    }
  });
  const counts = statuses.reduce((c, s) => {
    const key = ['RUNNING', 'QUEUED', 'COMPLETE', 'ERROR', 'CANCELLED', 'PENDING'].includes(s) ? s : 'UNKNOWN';
    c[key] = (c[key] || 0) + 1;
    return c;
  }, {});
  if (statuses.length) lines.push(Object.entries(counts).map(([s, n]) => `${s} ${n}`).join(' · '));
  return lines;
}
"""


class Reader:
    page = PAGE.replace("<script>", "<script>\n" + KAGGLE_SUMMARY, 1)

    def __init__(self, path: Path):
        self.path = path

    def get(self, *, force: bool = False) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data["accounts"], list) or not isinstance(data["kernels"], list):
                raise ValueError()
            return {**data, "stale": time.time() - data["at"] > 300 or data["partial"]}
        except (OSError, TypeError, KeyError, ValueError):
            return {"accounts": [], "kernels": [], "at": None, "stale": True}


def collect(path: Path, credentials: list[tuple[str, Path]], kernels: list[str]) -> None:
    from check_quota import fetch_account_quota
    from check_logs import _query_scheduler
    previous = {r["account"]: r for r in Reader(path).get()["accounts"]}
    now = time.time()
    accounts, rows, jobs = {}, [], []
    for label, credential in credentials:
        try:
            quota = fetch_account_quota(label, credential)
            accounts[quota["username"]] = (credential, None)
            row = {k: quota[k] for k in ("remaining_hours", "total_hours", "used_hours", "reserved_hours", "refresh_time_utc")}
            rows.append(dict(row, account=label, at=now))
        except Exception:
            rows.append({**previous.get(label, {}), "account": label, "error": True})
    for n, ref in enumerate(kernels, 1):
        try:
            status, _ = _query_scheduler(ref, accounts)
        except Exception:
            status = "UNKNOWN"
        jobs.append({"label": f"Kernel {n}", "status": status if status in {"RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCELLED", "PENDING"} else "UNKNOWN"})
    atomic_snapshot(path, {"at": now, "accounts": rows, "kernels": jobs,
                           "partial": any(r.get("error") for r in rows) or any(j["status"] == "UNKNOWN" for j in jobs)})


def _load_bound_job_meta(runs_dir: Path, run_label: str, kernel_ref: str, attempt_id: str = "v1") -> dict:
    empty = {"declared_arms": [], "device_count": None, "accelerator": None, "kernel_id": None}
    run_dir = runs_dir / run_label
    if not run_dir.is_dir():
        return empty

    candidates: list[Path] = []
    preferred = run_dir / attempt_id
    if (preferred / "operations" / "submission_state.json").is_file():
        candidates.append(preferred)
    others = sorted(
        (p.parents[1] for p in run_dir.glob("*/operations/submission_state.json") if p.parents[1] != preferred),
        key=lambda d: (d / "operations" / "submission_state.json").stat().st_mtime,
        reverse=True,
    )
    candidates.extend(others)

    matched: Path | None = None
    state_data: dict | None = None
    for cand in candidates:
        try:
            raw = json.loads((cand / "operations" / "submission_state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        bound_ref = (raw.get("binding") or {}).get("kernel") or (raw.get("kernel_response") or {}).get("kernel")
        if bound_ref == kernel_ref:
            matched = cand
            state_data = raw
            break
    if matched is None:
        return empty

    kernel_id = None
    accelerator = None
    device_count = None
    declared_arms: list[dict] = []

    if isinstance(state_data, dict):
        kresp = state_data.get("kernel_response") or {}
        if isinstance(kresp, dict):
            if kresp.get("kernel_id") is not None:
                kernel_id = kresp.get("kernel_id")
            if isinstance(kresp.get("accelerator"), str) and kresp.get("accelerator"):
                accelerator = kresp["accelerator"]

    launch_path = matched / "prepared" / "source_dataset" / "experiment_launch.json"
    try:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if isinstance(launch, dict):
            if isinstance(launch.get("accelerator"), str) and launch.get("accelerator"):
                accelerator = launch["accelerator"]
            if launch.get("device_count") is not None:
                device_count = int(launch["device_count"])
            jobs = launch.get("jobs")
            if isinstance(jobs, list):
                for j in jobs:
                    if isinstance(j, dict) and j.get("arm_id"):
                        dev = int(j["device"]) if j.get("device") is not None else None
                        declared_arms.append({"arm_id": str(j["arm_id"]), "device": dev})
    except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
        pass

    if not declared_arms:
        receipt_path = matched / "operations" / "launch_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if isinstance(receipt, dict):
                binding_arms = (receipt.get("binding") or {}).get("arms")
                if isinstance(binding_arms, list):
                    for item in binding_arms:
                        if isinstance(item, dict) and item.get("arm_id"):
                            declared_arms.append({"arm_id": str(item["arm_id"]), "device": None})
                if not declared_arms:
                    phys = (receipt.get("physical_runs") or {}).get("value")
                    if isinstance(phys, list) and phys and isinstance(phys[0], dict):
                        for aid in phys[0].get("arm_ids") or ():
                            if aid:
                                declared_arms.append({"arm_id": str(aid), "device": None})
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass

    return {
        "declared_arms": declared_arms,
        "device_count": device_count,
        "accelerator": accelerator,
        "kernel_id": kernel_id,
    }


def _parse_arm_curves(lines: list[str], declared_arms: list[dict] | None = None) -> tuple[str, list[dict], list[str]]:
    phase = "init"
    arms: dict[str, dict] = {}
    errors: list[str] = []
    resolved_arms: set[str] = set()
    for ln in lines:
        if ln.startswith("INPUT_RESOLVED "):
            parts = ln.split()
            if len(parts) >= 2:
                resolved_arms.add(parts[1])

    for item in declared_arms or ():
        if not isinstance(item, dict) or not item.get("arm_id"):
            continue
        aid = str(item["arm_id"])
        if aid not in arms:
            arms[aid] = {
                "arm_id": aid,
                "device": item.get("device"),
                "declared": True,
                "input_resolved": aid in resolved_arms,
                "log_seen": False,
                "points_by_pass": {},
                "latest": {},
            }

    for ln in lines:
        if ln.startswith("PAIR_PHASE_"):
            phase = ln.strip()
        elif "TRAIN_LOG " in ln:
            try:
                rec = json.loads(ln.split("TRAIN_LOG ", 1)[1])
            except Exception:
                continue
            arm_id = str(rec.get("arm_id") or "arm")
            entry = arms.setdefault(arm_id, {
                "arm_id": arm_id,
                "device": None,
                "declared": False,
                "input_resolved": arm_id in resolved_arms,
                "log_seen": False,
                "points_by_pass": {},
                "latest": rec,
            })
            entry["latest"] = rec
            entry["log_seen"] = True
            p_num = int(rec.get("completed_passes") or 0)
            entry["points_by_pass"][p_num] = {
                "pass": p_num,
                "planned_passes": rec.get("planned_passes"),
                "train_mae_eV": rec.get("train_live_mae_eV"),
                "dev_mae_eV": rec.get("development_mae_eV"),
                "best_dev_mae_eV": rec.get("best_development_mae_eV"),
                "elapsed_s": rec.get("elapsed_seconds"),
                "pass_wall_s": rec.get("pass_wall_seconds"),
                "eta_s": rec.get("eta_seconds"),
                "status": rec.get("status"),
            }
        elif any(tok in ln for tok in ("Error:", "Exception:", "Traceback (most recent call last)")):
            errors.append(ln.strip())

    for aid in sorted(resolved_arms):
        if aid not in arms:
            arms[aid] = {
                "arm_id": aid,
                "device": None,
                "declared": False,
                "input_resolved": True,
                "log_seen": False,
                "points_by_pass": {},
                "latest": {},
            }

    arm_list: list[dict] = []
    ref_map: dict[int, float] | None = None
    for entry in arms.values():
        if not entry["log_seen"]:
            arm_list.append({
                "arm_id": entry["arm_id"],
                "device": entry["device"],
                "declared": entry["declared"],
                "input_resolved": entry["input_resolved"],
                "log_seen": False,
                "latest": {},
                "completed_passes": 0,
                "planned_passes": None,
                "best_dev_mae_eV": None,
                "best_dev_pass": None,
                "best_train_mae_eV": None,
                "best_train_pass": None,
                "same_pass_delta_meV": None,
                "curve": [],
            })
            continue

        pts = [entry["points_by_pass"][k] for k in sorted(entry["points_by_pass"])]
        best_dev, best_dev_p = None, None
        best_trn, best_trn_p = None, None
        for pt in pts:
            dm = pt.get("dev_mae_eV")
            tm = pt.get("train_mae_eV")
            if isinstance(dm, (int, float)) and (best_dev is None or dm < best_dev):
                best_dev, best_dev_p = float(dm), pt["pass"]
            if isinstance(tm, (int, float)) and (best_trn is None or tm < best_trn):
                best_trn, best_trn_p = float(tm), pt["pass"]
        lt = entry["latest"]
        if lt.get("best_development_mae_eV") is not None:
            best_dev = float(lt["best_development_mae_eV"])
        is_ref = ref_map is None
        if is_ref:
            ref_map = {pt["pass"]: float(pt["dev_mae_eV"]) for pt in pts if isinstance(pt.get("dev_mae_eV"), (int, float))}
        cur_p = int(lt.get("completed_passes") or 0)
        cur_dev = lt.get("development_mae_eV")
        same_pass_delta = None
        if not is_ref and ref_map is not None and cur_p in ref_map and isinstance(cur_dev, (int, float)):
            same_pass_delta = round((float(cur_dev) - ref_map[cur_p]) * 1000.0, 2)
        arm_list.append({
            "arm_id": entry["arm_id"],
            "device": entry["device"],
            "declared": entry["declared"],
            "input_resolved": entry["input_resolved"],
            "log_seen": True,
            "latest": lt,
            "completed_passes": cur_p,
            "planned_passes": int(round(float(lt.get("planned_passes") or 0))) or None,
            "best_dev_mae_eV": best_dev,
            "best_dev_pass": best_dev_p,
            "best_train_mae_eV": best_trn,
            "best_train_pass": best_trn_p,
            "same_pass_delta_meV": same_pass_delta,
            "curve": pts,
        })
    return phase, arm_list, errors[-10:]


def fetch_live_snapshot(
    *,
    key_dir: Path,
    cred_overrides: list[str] | None = None,
    explicit_targets: list[str] | None = None,
    attempt_id: str = "v1",
    scan_limit: int = 12,
    read_timeout: float = 2.5,
) -> dict:
    from check_quota import DEFAULT_ACCOUNTS, fetch_account_quota, _resolve_targets
    import check_logs
    from check_logs import _discover_recent_bound, _query_scheduler, _resolve_target, inspect_kernel

    runs_dir = check_logs.ARTIFACTS_RUNS_DIR
    t0 = time.monotonic()
    cred_targets = _resolve_targets(cred_overrides, key_dir) if cred_overrides else [(lbl, key_dir / rel) for lbl, rel in DEFAULT_ACCOUNTS]

    def _fetch_one(item: tuple[str, Path]) -> tuple[dict, tuple[str, Path] | None]:
        lbl, p = item
        try:
            q = fetch_account_quota(lbl, p)
            return q, (q["username"], p)
        except Exception as exc:
            return {"account": lbl, "username": "-", "error": f"{type(exc).__name__}: {exc}"}, None

    with ThreadPoolExecutor(max_workers=max(1, len(cred_targets))) as pool:
        quota_res = list(pool.map(_fetch_one, cred_targets))

    accounts_rows = [r for r, _ in quota_res]
    accounts_map = {u: (p, json.loads(p.read_text(encoding="utf-8"))) for _, pair in quota_res if pair for u, p in [pair]}

    raw_targets = ([_resolve_target(t, attempt_id, runs_dir) for t in explicit_targets]
                   if explicit_targets else _discover_recent_bound(runs_dir, scan_limit))

    def _qsched(item: tuple[str, str]) -> tuple[str, str, str, str]:
        ref, label = item
        try:
            st, msg = _query_scheduler(ref, accounts_map)
        except Exception as exc:
            st, msg = "UNKNOWN", str(exc)
        return ref, label, st, msg

    with ThreadPoolExecutor(max_workers=max(1, min(6, len(raw_targets) or 1))) as pool:
        statuses = list(pool.map(_qsched, raw_targets))

    meta_by_ref = {
        (ref, label): _load_bound_job_meta(runs_dir, label, ref, attempt_id)
        for ref, label, _, _ in statuses
    }

    inspect_list = []
    seen_refs: set[str] = set()
    for item in statuses:
        if item[2] in {"RUNNING", "QUEUED"} and item[0] not in seen_refs:
            inspect_list.append(item)
            seen_refs.add(item[0])
    for item in statuses:
        if len(inspect_list) >= max(4, min(scan_limit, 10)):
            break
        if item[0] not in seen_refs:
            inspect_list.append(item)
            seen_refs.add(item[0])

    def _inspect(item: tuple[str, str, str, str]) -> dict:
        ref, label, st, msg = item
        meta = meta_by_ref.get((ref, label), {"declared_arms": [], "device_count": None, "accelerator": None, "kernel_id": None})
        rec = inspect_kernel(ref, label, accounts_map, read_timeout=read_timeout, sched=st, fail_msg=msg)
        phase, arms, errors = _parse_arm_curves(rec.get("lines") or [], declared_arms=meta["declared_arms"])
        return {
            "run": rec["run"],
            "kernel": rec["kernel"],
            "kaggle_url": f"https://www.kaggle.com/code/{rec['kernel']}",
            "account": rec["account"],
            "scheduler": rec["scheduler"],
            "failure_message": rec.get("failure_message") or "",
            "stream": rec.get("stream") or "none",
            "line_count": rec.get("line_count") or 0,
            "phase": phase,
            "declared_arms": meta["declared_arms"],
            "arm_count": len(arms) or len(meta["declared_arms"]),
            "device_count": meta["device_count"],
            "accelerator": meta["accelerator"],
            "kernel_id": meta["kernel_id"],
            "arms": arms,
            "error_lines": errors,
            "tail_lines": (rec.get("lines") or [])[-40:],
        }

    kernels_out: list[dict] = []
    if inspect_list:
        with ThreadPoolExecutor(max_workers=max(1, min(6, len(inspect_list)))) as pool:
            kernels_out = list(pool.map(_inspect, inspect_list))

    observed_arms_map = {k["kernel"]: k["arms"] for k in kernels_out}
    discovered_out = []
    for r, l, st, msg in statuses:
        meta = meta_by_ref.get((r, l), {"declared_arms": [], "device_count": None, "accelerator": None, "kernel_id": None})
        arm_cnt = len(observed_arms_map.get(r, [])) or len(meta["declared_arms"])
        discovered_out.append({
            "kernel": r,
            "run": l,
            "account": r.split("/", 1)[0],
            "scheduler": st,
            "failure_message": msg,
            "declared_arms": meta["declared_arms"],
            "arm_count": arm_cnt,
            "device_count": meta["device_count"],
            "accelerator": meta["accelerator"],
            "kernel_id": meta["kernel_id"],
        })

    ok_accounts = [r for r in accounts_rows if "error" not in r]
    latency_ms = int(round((time.monotonic() - t0) * 1000))
    now_epoch = time.time()
    return {
        "mode": "live",
        "fetched_at_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
        "fetched_at_epoch": now_epoch,
        "latency_ms": latency_ms,
        "summary": {
            "total_remaining_hours": round(sum(r["remaining_hours"] for r in ok_accounts), 2),
            "total_used_hours": round(sum(r["used_hours"] for r in ok_accounts), 2),
            "total_reserved_hours": round(sum(r["reserved_hours"] for r in ok_accounts), 2),
            "total_allowed_hours": round(sum(r["total_hours"] for r in ok_accounts), 2),
            "accounts_ok": len(ok_accounts),
            "accounts_total": len(accounts_rows),
            "discovered_kernels": len(statuses),
            "running_kernels": sum(1 for s in statuses if s[2] == "RUNNING"),
            "queued_kernels": sum(1 for s in statuses if s[2] == "QUEUED"),
            "running_arms": sum(d["arm_count"] for d in discovered_out if d["scheduler"] == "RUNNING" and d["arm_count"] > 0),
            "queued_arms": sum(d["arm_count"] for d in discovered_out if d["scheduler"] == "QUEUED" and d["arm_count"] > 0),
            "complete_kernels": sum(1 for s in statuses if s[2] == "COMPLETE"),
            "error_kernels": sum(1 for s in statuses if s[2] == "ERROR"),
        },
        "accounts": accounts_rows,
        "discovered": discovered_out,
        "kernels": kernels_out,
    }


def collect_full(path: Path, **settings) -> None:
    try:
        data = fetch_live_snapshot(**settings)
        if not data["summary"]["accounts_ok"]:
            raise RuntimeError("Kaggle quota unavailable")
        atomic_snapshot(path, {"full": True, "snapshot": data, "failure": None})
    except Exception:
        prior = SnapshotState(path, 300, full=True).get()
        atomic_snapshot(path, {"full": True, "snapshot": prior if prior.get("available") else None,
                               "failure": "collector"})


LIVE_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaggle 多账号配额与实验监控看板</title>
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
  h1 { margin:0; font-size:21px; font-weight:700; display:flex; align-items:center; gap:10px; }
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
  select, button, a.btn-link { background:var(--panel2); color:var(--text); border:1px solid var(--border); border-radius:9px; padding:7px 12px; font-size:13px; cursor:pointer; text-decoration:none; transition:border-color .15s; display:inline-flex; align-items:center; gap:5px; }
  select:hover, button:hover, a.btn-link:hover { border-color:var(--accent); }
  button.primary { background:linear-gradient(135deg,#0284c7,#0ea5e9); color:#fff; border-color:#0ea5e9; font-weight:600; }
  button.primary:disabled { opacity:.55; cursor:not-allowed; }
  .banner-err { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.42); color:var(--bad); padding:10px 14px; border-radius:10px; margin-bottom:14px; display:none; font-size:13px; }
  .hero { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; background:var(--panel); border:1px solid var(--border); border-left:4px solid var(--ok); border-radius:12px; padding:12px 16px; margin-bottom:16px; box-shadow:var(--shadow); font-size:13.5px; }
  .hero.tight { border-left-color:var(--warn); }
  .grid3 { display:grid; grid-template-columns:repeat(auto-fit,minmax(275px,1fr)); gap:14px; margin-bottom:18px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px 18px; box-shadow:var(--shadow); }
  .kpi-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .kpi-title { font-size:12px; color:var(--muted); font-weight:600; }
  .kpi-val { font-size:27px; font-weight:700; margin:6px 0 4px; font-variant-numeric:tabular-nums; }
  .kpi-sub { font-size:12px; color:var(--muted); }
  .bar { height:8px; background:var(--panel3); border-radius:999px; overflow:hidden; margin-top:9px; border:1px solid var(--border); display:flex; }
  .bar-seg-a { height:100%; background:linear-gradient(90deg,#0284c7,#38bdf8); transition:width .35s; }
  .bar-seg-b { height:100%; background:#f59e0b; transition:width .35s; }
  .bar-ok { height:100%; background:linear-gradient(90deg,#38bdf8,#22c55e); transition:width .35s; }
  .sec-head { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; margin:20px 0 10px; }
  .sec-head h2 { margin:0; font-size:16px; font-weight:700; }
  .run-card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:18px; margin-bottom:14px; box-shadow:var(--shadow); }
  .run-head { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; padding-bottom:12px; border-bottom:1px solid var(--border); margin-bottom:14px; }
  .run-meta { display:flex; flex-wrap:wrap; align-items:center; gap:10px; font-size:12.5px; color:var(--muted); margin-top:4px; }
  .chain { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:9px; margin-bottom:14px; }
  .step { padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:var(--panel2); font-size:12px; cursor:pointer; user-select:none; transition:transform .12s, border-color .15s, box-shadow .15s; }
  .step:hover { transform:translateY(-1px); border-color:var(--accent); }
  .step.COMPLETE { border-color:rgba(34,197,94,.42); background:rgba(34,197,94,.08); }
  .step.RUNNING { border-color:rgba(56,189,248,.65); background:rgba(56,189,248,.11); }
  .step.ERROR { border-color:rgba(239,68,68,.52); background:rgba(239,68,68,.09); }
  .step.selected { border-color:var(--accent); box-shadow:0 0 0 2px rgba(56,189,248,.48); }
  .spot { display:grid; grid-template-columns:1.05fr 1.15fr; gap:16px; background:var(--panel2); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
  .spot > * { min-width:0; }
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
  .tooltip { position:absolute; pointer-events:none; background:var(--panel3); color:var(--text); border:1px solid var(--accent); border-radius:8px; padding:6px 9px; font-size:11.5px; box-shadow:var(--shadow); display:none; z-index:10; white-space:nowrap; }
  .tabs { display:flex; flex-wrap:wrap; gap:6px; border-bottom:1px solid var(--border); margin:20px 0 12px; padding-bottom:8px; }
  .tab-btn { background:transparent; border:1px solid transparent; color:var(--muted); padding:7px 14px; border-radius:9px; font-weight:600; font-size:13px; }
  .tab-btn.active { background:var(--panel); border-color:var(--border); color:var(--accent); box-shadow:var(--shadow); }
  .tab-pane { display:none; }
  .tab-pane.active { display:block; }
  .filter-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
  .chip-btn { background:var(--panel2); border:1px solid var(--border); color:var(--muted); padding:4px 10px; border-radius:999px; font-size:12px; cursor:pointer; }
  .chip-btn.active { border-color:var(--accent); color:var(--accent); background:rgba(56,189,248,.12); font-weight:600; }
  table { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
  th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; font-size:12px; background:var(--panel2); }
  .mono { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }
  pre { background:var(--panel3); border:1px solid var(--border); border-radius:10px; padding:12px 14px; overflow:auto; max-width:100%; white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; color:var(--text); max-height:360px; margin:0; }
  details { margin-top:10px; min-width:0; max-width:100%; }
  summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
  .toast { position:fixed; right:20px; bottom:20px; background:var(--panel2); color:var(--text); border:1px solid var(--accent); padding:9px 14px; border-radius:10px; font-size:13px; box-shadow:var(--shadow); opacity:0; transform:translateY(8px); transition:opacity .2s, transform .2s; pointer-events:none; z-index:50; }
  .toast.show { opacity:1; transform:translateY(0); }
  .skel { background:linear-gradient(90deg,var(--panel) 25%,var(--panel2) 50%,var(--panel) 75%); background-size:200% 100%; animation:shim 1.4s infinite; border-radius:12px; height:112px; border:1px solid var(--border); }
  @keyframes shim { 0% { background-position:200% 0; } 100% { background-position:-200% 0; } }
  a.klink { color:var(--accent); text-decoration:none; }
  a.klink:hover { text-decoration:underline; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1><span class="logo-dot"></span>Kaggle 多账号配额与实验监控看板 <span id="connPill" class="pill">连接中</span></h1>
      <div style="font-size:12px;color:var(--muted);margin-top:4px" id="metaLine">正在通过 Kaggle API 与本地 run 绑定日志拉取实时状态...</div>
      <div id="dataAge" style="font-size:12px;color:var(--muted);margin-top:4px">尚无快照</div>
    </div>
    <div class="controls">
      <a class="btn-link" href="http://127.0.0.1:8765/" target="_blank" rel="noopener" data-local-only>前往 IMS 看板</a>
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
      <button id="copySummary">复制摘要</button>
      <span id="summaryFeedback" role="status"></span>
      <!-- dashboard-language -->
    </div>
  </header>

  <div id="errBox" class="banner-err"></div>
  <div id="heroBar" class="hero" style="display:none"></div>

  <div class="grid3" id="quotaGrid">
    <div class="skel"></div><div class="skel"></div><div class="skel"></div>
  </div>

  <div class="sec-head">
    <h2 id="kernelsTitle">实验实时与近期追踪</h2>
    <span style="font-size:12px;color:var(--muted)" id="kernelsSub"></span>
  </div>
  <div id="kernelCards"><div class="skel" style="height:200px"></div></div>

  <div class="tabs" role="tablist">
    <button class="tab-btn active" data-tab="all">全部最近绑定任务 <span id="tabAllCnt"></span></button>
    <button class="tab-btn" data-tab="raw">原始 JSON 快照</button>
  </div>

  <div id="tab-all" class="tab-pane active">
    <div id="statusFilterBar" class="filter-bar"></div>
    <div class="filter-bar"><label>搜索任务 <input id="kernelSearch" type="search" placeholder="Run ID、Kernel 或账号"></label>
      <span id="kernelMatches" role="status"></span></div>
    <div class="card" style="padding:0;overflow:auto"><table id="allTbl"></table></div>
  </div>

  <div id="tab-raw" class="tab-pane">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font-size:13px;color:var(--muted)">最新 Kaggle API 与 TRAIN_LOG 解析快照 JSON</span>
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
let latestSnap = null;
const selectedArmByKernel = {};
let selectedFilter = 'ALL';

function showToast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2600);
}
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('kaggle_theme', theme);
  $('themeBtn').textContent = theme === 'light' ? '🌙 深色' : '☀️ 浅色';
}
applyTheme(localStorage.getItem('kaggle_theme') || 'dark');
$('themeBtn').onclick = () => applyTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light');

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  };
});

function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function fmtNum(v) { return v == null ? '-' : Number(v).toLocaleString(); }
function fmtMae(v) { return typeof v === 'number' ? (v * 1000).toFixed(2) + ' meV' : '-'; }
function fmtDur(sec) {
  if (sec == null || sec < 0) return '-';
  const s = Math.round(sec), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h > 0 ? `${h}h ${String(m).padStart(2,'0')}m` : `${m}m ${String(r).padStart(2,'0')}s`;
}
function schedPill(st) {
  if (st === 'RUNNING') return 'info';
  if (st === 'COMPLETE') return 'ok';
  if (st === 'QUEUED') return 'warn';
  if (st === 'ERROR' || st === 'CANCELLED') return 'bad';
  return '';
}
function fmtAccel(accel, devCount) {
  const n = Number(devCount);
  const cnt = Number.isFinite(n) && n > 0 ? n : 0;
  const s = String(accel || '');
  const short = s.includes('T4') ? 'T4' : (s.includes('P100') ? 'P100' : s);
  if (short && cnt > 1) return `${short}×${cnt}`;
  if (short) return short;
  return cnt > 1 ? `${cnt} GPU` : '';
}
function fmtArmList(arms) {
  return (arms || []).map(a => (a && a.device != null ? `GPU ${a.device}: ${a.arm_id}` : String(a?.arm_id || ''))).filter(Boolean).join(' · ');
}

function buildChartHtml(arm, idx) {
  const pts = (arm.curve || []).filter(p => typeof p.train_mae_eV === 'number' || typeof p.dev_mae_eV === 'number');
  if (pts.length < 2) return '<div class="chart-box" style="color:var(--muted);display:flex;align-items:center;justify-content:center;min-height:150px">暂无足够 Pass 点绘制收敛曲线</div>';
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
        data-pass="${esc(k.p.pass)}" data-mae="${trStr}" data-dev="${dvStr}" data-el="${esc(fmtDur(k.p.elapsed_s))}" style="cursor:pointer"/>`;
  }).join('');
  const gid = `kgrad_${idx}`;
  return `<div class="chart-box">
    <div class="chart-head">
      <div class="legend">
        <span><span class="lg-dot" style="background:var(--accent)"></span>Train</span>
        ${hasDev ? `<span><span class="lg-dot" style="background:var(--ok)"></span>Dev${bestDevK ? ` (最优 P${bestDevK.p.pass}: ${(bestDevK.p.dev_mae_eV*1000).toFixed(2)})` : ''}</span>` : ''}
      </div>
      <span class="mono">P${pts[0].pass}..P${pts[pts.length-1].pass}</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block">
      <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${hasDev ? '#22c55e' : '#38bdf8'}" stop-opacity="0.22"/><stop offset="100%" stop-color="${hasDev ? '#22c55e' : '#38bdf8'}" stop-opacity="0.01"/></linearGradient></defs>
      ${grid}
      <polygon points="${areaPts}" fill="url(#${gid})"/>
      ${trainPts.length >= 2 ? `<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" points="${trainLine}"/>` : ''}
      ${devPts.length >= 2 ? `<polyline fill="none" stroke="var(--ok)" stroke-width="2.2" stroke-linejoin="round" points="${devLine}"/>` : ''}
      ${dots}
      <text x="${L}" y="${H-5}" fill="var(--muted)" font-size="10">P${pts[0].pass}</text>
      <text x="${W-R}" y="${H-5}" text-anchor="end" fill="var(--muted)" font-size="10">P${pts[pts.length-1].pass}</text>
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
        const parts = [`Pass ${d.pass}`];
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

function renderKernelCards(d) {
  const kernels = d.kernels || [];
  const active = kernels.filter(k => k.scheduler === 'RUNNING' || k.scheduler === 'QUEUED');
  const shown = active.length ? kernels : kernels.slice(0, 6);
  $('kernelsTitle').textContent = active.length ? 'Kaggle 实时运行与排队实验' : 'Kaggle 近期绑定实验汇总（当前无运行中 Kernel）';
  $('kernelsSub').textContent = `已扫描 ${d.summary?.discovered_kernels ?? kernels.length} 个绑定实验 · 详细检查 ${kernels.length} 个 · 点击 Arm 卡片切换图表`;

  if (!shown.length) {
    $('kernelCards').innerHTML = '<div class="card" style="color:var(--muted)">暂未在 D:/molgap-artifacts/runs 发现已绑定的 Kaggle 实验。</div>';
    return;
  }

  let chartIdx = 0;
  $('kernelCards').innerHTML = shown.map(k => {
    const arms = k.arms || [];
    const chosenId = selectedArmByKernel[k.kernel];
    const cur = arms.find(a => a.arm_id === chosenId) || arms[0];
    const curId = cur ? cur.arm_id : null;
    const armCount = k.arm_count || arms.length || (k.declared_arms || []).length;
    const accelTag = fmtAccel(k.accelerator, k.device_count);
    const armRoleTag = armCount === 2 ? '双 Arm' : (armCount === 1 ? '单 Arm' : (armCount > 0 ? `${armCount} Arms` : ''));
    const topoPill = armRoleTag ? `<span class="pill ${armCount >= 2 ? 'ok' : ''}">${esc(armRoleTag)}${accelTag ? ` · ${esc(accelTag)}` : ''}</span>` : '';
    const armMapStr = fmtArmList(arms.length ? arms : (k.declared_arms || []));

    const armChips = arms.map((a, idx) => {
      const lt = a.latest || {};
      const st = lt.status || k.scheduler;
      const isPendingLog = a.log_seen === false;
      const cls = isPendingLog ? (k.scheduler === 'ERROR' ? 'ERROR' : '') : (st === 'COMPLETE' ? 'COMPLETE' : (st === 'RUNNING' ? 'RUNNING' : (k.scheduler === 'ERROR' ? 'ERROR' : '')));
      const sel = a.arm_id === curId ? ' selected' : '';
      const gpuBadge = a.device != null ? `<span class="pill info" style="padding:1px 6px;font-size:10.5px">GPU ${esc(a.device)}</span> ` : '';
      const rightStr = isPendingLog ? (a.input_resolved ? '已加载输入' : '等待日志') : `P${a.completed_passes ?? '-'}/${a.planned_passes ?? '?'}`;
      const bestStr = a.best_dev_mae_eV != null ? `Best ${(a.best_dev_mae_eV*1000).toFixed(2)} meV` : 'MAE -';
      const deltaStr = a.same_pass_delta_meV != null ? ` · Δ ${a.same_pass_delta_meV >= 0 ? '+' : ''}${a.same_pass_delta_meV}` : (idx === 0 && arms.length > 1 ? ' · Baseline' : '');
      const subLine = isPendingLog ? '待首轮 Pass 日志' : `${bestStr}${deltaStr}`;
      return `<div class="step ${cls}${sel}" role="button" tabindex="0" data-kernel="${esc(k.kernel)}" data-arm="${esc(a.arm_id)}">
        <div style="font-weight:700;display:flex;justify-content:space-between;align-items:center;gap:6px"><span>${gpuBadge}<span class="mono">${esc(a.arm_id)}</span></span><span class="mono" style="font-size:11px">${esc(rightStr)}</span></div>
        <div style="color:var(--muted);margin-top:4px">${esc(subLine)}</div>
      </div>`;
    }).join('');

    let spotHtml = '';
    if (cur && cur.log_seen === false) {
      spotHtml = `<div class="spot" style="grid-template-columns:1fr"><div style="color:var(--muted);font-size:13px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;color:var(--text)">
          <span>聚焦 Arm：<b class="mono">${esc(cur.arm_id)}</b>${cur.device != null ? ` <span class="pill info">GPU ${esc(cur.device)}</span>` : ''}</span>
          ${arms.length > 1 ? `<span style="font-size:12px;color:var(--muted)">共 ${arms.length} 个 Arm · 点击上方卡片切换</span>` : ''}
        </div>
        暂未收到训练指标，正在等待首轮 Pass 结束（流模式：${esc(k.stream)}，共 ${k.line_count} 行）${k.failure_message ? ` · 失败原因：<span style="color:var(--bad)">${esc(k.failure_message)}</span>` : ''}。
        ${k.tail_lines && k.tail_lines.length ? `<details><summary>查看最近 ${k.tail_lines.length} 行日志输出${k.error_lines && k.error_lines.length ? ` (含 ${k.error_lines.length} 条异常)` : ''}</summary>
          ${k.error_lines && k.error_lines.length ? `<pre style="margin-top:8px;color:var(--bad);max-height:130px">${esc(k.error_lines.join('\n'))}</pre>` : ''}
          <pre style="margin-top:8px">${esc(k.tail_lines.join('\n'))}</pre>
        </details>` : ''}
      </div></div>`;
    } else if (cur) {
      const lt = cur.latest || {};
      const pct = cur.planned_passes ? Math.min(100, (cur.completed_passes / cur.planned_passes) * 100) : 0;
      const cid = chartIdx++;
      spotHtml = `
        <div class="spot">
          <div>
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;font-size:13px">
              <span>聚焦 Arm：<b class="mono">${esc(cur.arm_id)}</b>${cur.device != null ? ` <span class="pill info">GPU ${esc(cur.device)}</span>` : ''}${arms.length > 1 ? ` <span style="font-size:11.5px;color:var(--muted)">(共 ${arms.length} 个 Arm)</span>` : ''}</span>
              <span class="mono"><b>Pass ${cur.completed_passes}/${cur.planned_passes ?? '?'}</b> (${pct.toFixed(1)}%)</span>
            </div>
            ${cur.planned_passes ? `<div class="bar"><div class="bar-ok" style="width:${pct.toFixed(1)}%"></div></div>` : ''}
            <div class="spot-metrics">
              <div class="mini-stat"><div class="k">Best Dev MAE</div><div class="v" style="color:var(--ok)">${fmtMae(cur.best_dev_mae_eV)}</div><div class="s">${cur.best_dev_pass != null ? `@ Pass ${cur.best_dev_pass}` : '-'}</div></div>
              <div class="mini-stat"><div class="k">Latest Dev MAE</div><div class="v">${fmtMae(lt.development_mae_eV)}</div><div class="s">${cur.same_pass_delta_meV != null ? `Same-Pass Δ ${cur.same_pass_delta_meV>=0?'+':''}${cur.same_pass_delta_meV} meV` : '对照基线'}</div></div>
              <div class="mini-stat"><div class="k">Train Live MAE</div><div class="v" style="color:var(--accent)">${fmtMae(lt.train_live_mae_eV)}</div><div class="s">${cur.best_train_mae_eV != null ? `最优 ${fmtMae(cur.best_train_mae_eV)} @ P${cur.best_train_pass}` : '-'}</div></div>
              <div class="mini-stat"><div class="k">单 Pass 耗时</div><div class="v">${lt.pass_wall_seconds != null ? `${Number(lt.pass_wall_seconds).toFixed(0)} s` : '-'}</div><div class="s">阶段 ${esc(k.phase)}</div></div>
              <div class="mini-stat"><div class="k">已运行 / ETA</div><div class="v" style="font-size:15px">${fmtDur(lt.elapsed_seconds)}</div><div class="s">剩 ${fmtDur(lt.eta_seconds)}</div></div>
              <div class="mini-stat"><div class="k">日志流 / 行数</div><div class="v" style="font-size:15px">${esc(k.stream)}</div><div class="s">共 ${fmtNum(k.line_count)} 行</div></div>
            </div>
            ${k.tail_lines && k.tail_lines.length ? `<details><summary>查看最近 ${k.tail_lines.length} 行日志输出${k.error_lines && k.error_lines.length ? ` (含 ${k.error_lines.length} 条异常)` : ''}</summary>
              ${k.error_lines && k.error_lines.length ? `<pre style="margin-top:8px;color:var(--bad);max-height:130px">${esc(k.error_lines.join('\n'))}</pre>` : ''}
              <pre style="margin-top:8px">${esc(k.tail_lines.join('\n'))}</pre>
            </details>` : ''}
          </div>
          ${buildChartHtml(cur, cid)}
        </div>`;
    } else {
      spotHtml = `<div class="spot" style="grid-template-columns:1fr"><div style="color:var(--muted);font-size:13px">尚未解析到 TRAIN_LOG 指标（流模式：${esc(k.stream)}，共 ${k.line_count} 行）${k.failure_message ? ` · 失败原因：<span style="color:var(--bad)">${esc(k.failure_message)}</span>` : ''}。
        ${k.tail_lines && k.tail_lines.length ? `<details open><summary>查看最近日志</summary><pre style="margin-top:8px">${esc(k.tail_lines.join('\n'))}</pre></details>` : ''}
      </div></div>`;
    }

    return `<div class="run-card">
      <div class="run-head">
        <div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><b class="mono" style="font-size:16px">${esc(k.run)}</b> <span class="pill info">${esc(k.account)}</span> ${topoPill}</div>
          <div class="run-meta"><a class="klink mono" href="${esc(k.kaggle_url)}" target="_blank" rel="noopener">${esc(k.kernel)} ↗</a>${k.kernel_id != null ? ` · <span class="mono">#${esc(k.kernel_id)}</span>` : ''} · 阶段 <span class="mono">${esc(k.phase)}</span>${armMapStr ? ` · <span class="mono">${esc(armMapStr)}</span>` : ''}</div>
        </div>
        <div><span class="pill ${schedPill(k.scheduler)}">${k.scheduler==='RUNNING'?'<span class="pulse"></span>':''}${esc(k.scheduler)}</span></div>
      </div>
      ${k.failure_message ? `<div style="margin-bottom:10px;padding:8px 12px;border-radius:9px;background:rgba(239,68,68,.12);color:var(--bad);font-size:12.5px"><b>失败信息：</b>${esc(k.failure_message)}</div>` : ''}
      ${arms.length ? `<div class="chain">${armChips}</div>` : ''}
      ${spotHtml}
    </div>`;
  }).join('');

  document.querySelectorAll('.step[data-kernel][data-arm]').forEach(el => {
    const act = () => { selectedArmByKernel[el.dataset.kernel] = el.dataset.arm; renderKernelCards(latestSnap); };
    el.onclick = act;
    el.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } };
  });
  bindChartTooltips();
}

function renderAllTable(d) {
  const rows = d.discovered || [];
  $('tabAllCnt').textContent = `(${rows.length})`;
  const counts = {};
  rows.forEach(r => { counts[r.scheduler] = (counts[r.scheduler] || 0) + 1; });
  const statuses = Object.keys(counts);
  $('statusFilterBar').innerHTML = `<span style="font-size:12px;color:var(--muted)">按状态筛选：</span>` +
    `<button class="chip-btn${selectedFilter==='ALL'?' active':''}" data-st="ALL">全部 (${rows.length})</button>` +
    [...new Set([...statuses, ...(selectedFilter === 'ALL' ? [] : [selectedFilter])])].map(st => `<button class="chip-btn${selectedFilter===st?' active':''}" data-st="${esc(st)}">${esc(st)} (${counts[st] || 0})</button>`).join('');
  document.querySelectorAll('.chip-btn[data-st]').forEach(b => {
    b.onclick = () => { selectedFilter = b.dataset.st; renderAllTable(latestSnap); };
  });

  const query = $('kernelSearch').value;
  const shown = rows.filter(r => (selectedFilter === 'ALL' || r.scheduler === selectedFilter)
    && dashboardMatches(query, r.run, r.kernel, r.account, r.kernel_id, ...(r.declared_arms || []).map(a => a.arm_id)));
  $('kernelMatches').textContent = `${shown.length} / ${rows.length} 项`;
  $('allTbl').innerHTML = `<thead><tr><th>Logical Run ID</th><th>账号</th><th>状态</th><th>Arms</th><th>Kaggle Kernel</th><th>备注</th></tr></thead><tbody>` +
    (shown.length ? shown.map(r => {
      const cnt = r.arm_count || (r.declared_arms || []).length;
      const roleTag = cnt === 2 ? '双 Arm' : (cnt === 1 ? '单 Arm' : (cnt > 0 ? `${cnt} Arms` : ''));
      const armsText = fmtArmList(r.declared_arms || []);
      const armsCell = (roleTag ? `<span class="pill ${cnt>=2?'ok':''}">${esc(roleTag)}</span> ` : '') +
        `<span class="mono" style="font-size:12px">${esc(armsText || '-')}</span>`;
      const idTag = r.kernel_id != null ? ` <span class="mono" style="color:var(--muted);font-size:11.5px">#${esc(r.kernel_id)}</span>` : '';
      return `<tr><td class="mono"><b>${esc(r.run)}</b></td><td class="mono">${esc(r.account)}</td><td><span class="pill ${schedPill(r.scheduler)}">${esc(r.scheduler)}</span></td><td>${armsCell}</td><td><a class="klink mono" href="https://www.kaggle.com/code/${esc(r.kernel)}" target="_blank" rel="noopener">${esc(r.kernel)} ↗</a>${idTag}</td><td style="color:var(--muted);font-size:12px">${esc(r.failure_message || '-')}</td></tr>`;
    }).join('')
      : `<tr><td colspan="6" style="color:var(--muted)">暂无匹配的绑定任务</td></tr>`) + `</tbody>`;
}

function render(d) {
  latestSnap = d;
  dashboardObserved(d.fetched_at_epoch, 300, d.stale ? '刷新失败，显示上次数据'
    : d.error ? '读取失败，暂无快照' : (d.accounts || []).some(a => a.error) ? '部分配额不可用' : '');
  const errBox = $('errBox');
  if (d.error) { errBox.style.display = 'block'; errBox.textContent = '拉取提示: ' + d.error; }
  else { errBox.style.display = 'none'; }

  const pill = $('connPill');
  pill.className = 'pill ' + (d.error ? 'warn' : 'ok');
  pill.innerHTML = `<span class="pulse"></span>${d.stale ? '显示旧快照' : d.error ? '读取失败' : d.cached ? '缓存命中' : '实时在线'}`;
  $('metaLine').textContent = `本地刷新 ${d.fetched_at_jst || '-'} · 接口耗时 ${d.latency_ms ?? '-'} ms`;
  if (!d.fetched_at_epoch && d.error) {
    $('quotaGrid').textContent = '暂无配额数据';
    $('kernelCards').textContent = '暂无作业数据';
    renderAllTable(d);
    return;
  }

  const s = d.summary || {};
  const hasActive = (s.running_kernels || 0) + (s.queued_kernels || 0) > 0;
  const hero = $('heroBar');
  hero.style.display = 'flex';
  hero.className = 'hero' + (s.total_remaining_hours < 15 ? ' tight' : '');
  const runArmStr = s.running_arms ? ` (${s.running_arms} Arms)` : '';
  const qArmStr = s.queued_arms ? ` (${s.queued_arms} Arms)` : '';
  hero.innerHTML = `<div><b>多账号 GPU 配额总览：</b>可用 <b>${s.total_remaining_hours ?? '-'} h</b> / ${s.total_allowed_hours ?? '-'} h（已用 ${s.total_used_hours ?? '-'} h，预留 ${s.total_reserved_hours ?? '-'} h）</div>
                    <div class="pill ${hasActive ? 'info' : 'ok'}">运行 ${s.running_kernels ?? 0}${runArmStr} · 排队 ${s.queued_kernels ?? 0}${qArmStr} · 完成 ${s.complete_kernels ?? 0} · 异常 ${s.error_kernels ?? 0}</div>`;

  $('quotaGrid').innerHTML = (d.accounts || []).map(a => {
    if (a.error) {
      return `<div class="card"><div class="kpi-top"><span class="kpi-title">${esc(a.account)}</span><span class="pill bad">读取失败</span></div><div style="margin-top:8px;color:var(--bad);font-size:12.5px">${esc(a.error)}</div></div>`;
    }
    const tot = a.total_hours || 1;
    const pctUsed = Math.min(100, (a.used_hours / tot) * 100);
    const pctRsv = Math.min(100 - pctUsed, (a.reserved_hours / tot) * 100);
    return `<div class="card">
      <div class="kpi-top"><span class="kpi-title">${esc(a.account)} · <span class="mono">${esc(a.username)}</span></span><span class="pill ok">剩 ${esc(a.remaining_hm)}</span></div>
      <div class="kpi-val" style="color:var(--ok)">${a.remaining_hours.toFixed(2)} <span style="font-size:14px;color:var(--muted);font-weight:500">/ ${a.total_hours.toFixed(1)} h</span></div>
      <div class="kpi-sub">已用 ${a.used_hours.toFixed(2)} h · 预留 ${a.reserved_hours.toFixed(2)} h · 重置 ${esc(a.refresh_time_local)} (还剩 ${esc(a.reset_in)})</div>
      <div class="bar"><div class="bar-seg-a" style="width:${pctUsed.toFixed(2)}%"></div><div class="bar-seg-b" style="width:${pctRsv.toFixed(2)}%"></div></div>
    </div>`;
  }).join('');

  renderKernelCards(d);
  renderAllTable(d);
  $('rawBox').textContent = JSON.stringify(d, null, 2);
}

async function pull(force) {
  const btn = $('refreshBtn');
  if (force && Date.now() < cooldownUntil) {
    const wait = Math.ceil((cooldownUntil - Date.now()) / 1000);
    showToast(`冷却保护中：请 ${wait} 秒后再刷新 Kaggle API`);
    return;
  }
  btn.disabled = true; btn.textContent = '拉取中...';
  try {
    const res = await fetch('/api/status' + (force ? '?refresh=1' : ''));
    const data = await res.json();
    if (!data.cached && data.min_interval_s) cooldownUntil = Date.now() + data.min_interval_s * 1000;
    else if (data.cooldown_remaining_s) cooldownUntil = Date.now() + data.cooldown_remaining_s * 1000;
    render(data);
    if (force && !data.cached && !data.error) showToast('已从 Kaggle 刷新最新配额与日志');
  } catch (e) {
    dashboardFetchFailed();
    $('errBox').style.display = 'block';
    $('errBox').textContent = '本机服务请求失败: ' + e;
  } finally {
    btn.disabled = false; btn.textContent = '立即刷新';
  }
}

$('copyRawBtn').onclick = async () => {
  try { await navigator.clipboard.writeText($('rawBox').textContent); showToast('已复制 JSON 快照'); }
  catch { showToast('复制失败，请手动选中复制'); }
};

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
$('kernelSearch').oninput = () => { if (latestSnap) renderAllTable(latestSnap); };
bindDashboardSummary(() => latestSnap?.fetched_at_epoch
  ? kaggleSummary(latestSnap.accounts || [], (latestSnap.discovered || []).map(k => k.scheduler)) : []);
pull(false);
</script>
</body>
</html>
"""


class LiveState:
    page = LIVE_PAGE.replace("<script>", "<script>\n" + KAGGLE_SUMMARY, 1)

    def __init__(self, *, key_dir: Path, cred_overrides: list[str] | None, explicit_targets: list[str] | None,
                 attempt_id: str, scan_limit: int, read_timeout: float, min_interval: float) -> None:
        self.key_dir = key_dir
        self.cred_overrides = cred_overrides
        self.explicit_targets = explicit_targets
        self.attempt_id = attempt_id
        self.scan_limit = scan_limit
        self.read_timeout = read_timeout
        self.min_interval = max(10.0, float(min_interval))
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
            fresh = fetch_live_snapshot(
                key_dir=self.key_dir, cred_overrides=self.cred_overrides,
                explicit_targets=self.explicit_targets, attempt_id=self.attempt_id,
                scan_limit=self.scan_limit, read_timeout=self.read_timeout,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"

        with self._cond:
            self._in_flight = False
            if fresh is not None:
                self._snapshot = fresh
                self._last_fetch = time.monotonic()
                payload = {**fresh, "cached": False, "cooldown_remaining_s": self.min_interval, "min_interval_s": self.min_interval, "error": None}
            elif self._snapshot is not None:
                self._snapshot = {**self._snapshot, "error": err, "stale": True}
                payload = {**self._snapshot, "cached": True, "min_interval_s": self.min_interval}
            else:
                payload = {"accounts": [], "kernels": [], "discovered": [], "error": err or "Unknown error", "cached": False, "min_interval_s": self.min_interval}
            self._cond.notify_all()
            return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("web", "collect", "live"))
    p.add_argument("snapshot", nargs="?", type=Path, help="Snapshot JSON path (required for web/collect)")
    p.add_argument("--credentials", type=Path, action="append", default=[])
    p.add_argument("--kernel", action="append", default=[])
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--open", action="store_true", help="Open local dashboard in browser (live mode)")
    p.add_argument("--key-dir", type=Path)
    p.add_argument("--scan-limit", type=int, default=12)
    p.add_argument("--attempt-id", default="v1")
    p.add_argument("--read-timeout", type=float, default=2.5)
    p.add_argument("--min-interval", type=float, default=20.0)
    p.add_argument("--full-snapshot", action="store_true", help="Use live dashboard data in collect/web mode")
    args = p.parse_args(argv)

    if args.mode in {"web", "collect"} and args.snapshot is None:
        p.error(f"{args.mode} requires snapshot path")
    if args.mode == "web":
        if args.full_snapshot:
            reader = SnapshotState(args.snapshot, 300, full=True)
            reader.page = LiveState.page
        else:
            reader = Reader(args.snapshot)
        with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(reader)) as server:
            server.serve_forever()
    elif args.mode == "collect":
        if not args.credentials:
            p.error("collect requires at least one --credentials")
        if args.full_snapshot:
            collect_full(args.snapshot, key_dir=args.key_dir or Path("."),
                         cred_overrides=[str(p) for p in args.credentials], explicit_targets=args.kernel or None,
                         attempt_id=args.attempt_id, scan_limit=args.scan_limit, read_timeout=args.read_timeout)
        else:
            collect(args.snapshot, [(f"Account {i}", v) for i, v in enumerate(args.credentials, 1)], args.kernel)
    else:
        from check_quota import DEFAULT_KEY_DIR

        cred_args = [str(x) for x in args.credentials] or None
        state = LiveState(
            key_dir=args.key_dir or DEFAULT_KEY_DIR, cred_overrides=cred_args, explicit_targets=args.kernel or None,
            attempt_id=args.attempt_id, scan_limit=args.scan_limit,
            read_timeout=args.read_timeout, min_interval=args.min_interval,
        )
        with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state)) as server:
            url = f"http://127.0.0.1:{server.server_address[1]}/"
            print(f"Kaggle live dashboard listening on {url} (127.0.0.1 only, cooldown {state.min_interval:.0f}s; Ctrl+C to stop)", flush=True)
            if args.open:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nKaggle dashboard stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

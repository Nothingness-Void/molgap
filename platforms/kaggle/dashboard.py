"""Read-only Kaggle quota and selected kernel dashboard."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platforms/ims"))
from check_queue import atomic_snapshot, make_handler, ThreadingHTTPServer

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kaggle 看板</title><style>body{font:16px system-ui;background:#101923;color:#edf5fa;max-width:55rem;margin:5vh auto;padding:1rem}
section{background:#1d3040;padding:1rem;margin:1rem 0;border-radius:.6rem}button{padding:.5rem}#health{color:#ffcf79}</style>
<h1>Kaggle 看板</h1><button onclick="refresh()">刷新快照</button><p id="health"></p><main></main>
<script>async function refresh(){const h=document.querySelector('#health'),m=document.querySelector('main');try{
const d=await (await fetch('/api/status',{cache:'no-store'})).json();m.replaceChildren();
for(const r of d.accounts){const e=document.createElement('section');e.textContent=r.account+' · '+(r.at?'剩余 '+r.remaining_hours+' / '+r.total_hours+' GPU 小时，已用 '+r.used_hours+'，预留 '+r.reserved_hours+'，重置 '+r.refresh_time_utc:'尚无配额快照')+(r.at?' · '+new Date(r.at*1000).toLocaleString():'')+(r.error?' · 配额读取失败':'');m.append(e)}
for(const k of d.kernels){const e=document.createElement('section');e.textContent=k.label+' · '+k.status;m.append(e)}
h.textContent=(d.stale?'已过期或部分数据不可用':'快照正常')+' · 最近采集 '+(d.at?new Date(d.at*1000).toLocaleString():'尚无可用快照');
}catch{h.textContent='读取失败，显示的数据可能已过期'}}refresh();setInterval(refresh,30000)</script></html>"""


class Reader:
    page = PAGE

    def __init__(self, path):
        self.path = path

    def get(self, *, force=False):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data["accounts"], list) or not isinstance(data["kernels"], list):
                raise ValueError()
            return {**data, "stale": time.time() - data["at"] > 300 or data["partial"]}
        except (OSError, TypeError, KeyError, ValueError):
            return {"accounts": [], "kernels": [], "at": None, "stale": True}


def collect(path, credentials, kernels):
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
        jobs.append({"label": f"Kernel {n}", "status": status if status in {"RUNNING","QUEUED","COMPLETE","ERROR","CANCELLED","PENDING"} else "UNKNOWN"})
    atomic_snapshot(path, {"at": now, "accounts": rows, "kernels": jobs,
                           "partial": any(r.get("error") for r in rows) or any(j["status"] == "UNKNOWN" for j in jobs)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("web", "collect"))
    p.add_argument("snapshot", type=Path)
    p.add_argument("--credentials", type=Path, action="append", default=[])
    p.add_argument("--kernel", action="append", default=[])
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    if args.mode == "web":
        with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Reader(args.snapshot))) as server:
            server.serve_forever()
    elif args.credentials:
        collect(args.snapshot, [(f"Account {i}", v) for i,v in enumerate(args.credentials, 1)], args.kernel)
    else:
        p.error("collect requires at least one --credentials")

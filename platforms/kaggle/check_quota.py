"""Read-only Kaggle GPU quota and reset-time monitor for configured accounts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import requests


DEFAULT_KEY_DIR = Path(os.environ.get("MOLGAP_KAGGLE_KEY_DIR", "D:/\u4e0b\u8f7d/Key"))
DEFAULT_ACCOUNTS = (
    ("kaggle1", ".kaggle1_default/kaggle.json"),
    ("kaggle2", ".kaggle2_cli/kaggle.json"),
    ("kaggle3", ".kaggle3_cli/kaggle.json"),
)
QUOTA_URL = "https://www.kaggle.com/api/v1/kernels/quota"


def _duration_seconds(entry: dict | None) -> float:
    if not isinstance(entry, dict):
        raise ValueError("Missing Kaggle quota duration")
    seconds = float(entry.get("seconds", 0)) + float(entry.get("nanos", 0)) / 1e9
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Malformed Kaggle quota duration")
    return seconds


def _format_hm(hours: float) -> str:
    total_minutes = max(0, int(round(hours * 60)))
    h, m = divmod(total_minutes, 60)
    return f"{h}h {m:02d}m"


def parse_quota_payload(label: str, username: str, payload: dict, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    gpu = payload.get("gpuQuota")
    if not isinstance(gpu, dict):
        raise ValueError("Missing Kaggle GPU quota")
    total_s = _duration_seconds(gpu.get("totalTimeAllowed"))
    used_s = _duration_seconds(gpu.get("timeUsed"))
    reserved_s = _duration_seconds(gpu.get("timeReserved"))
    remaining_s = max(0.0, total_s - used_s - reserved_s)

    refresh_raw = payload.get("quotaRefreshTime")
    refresh_utc_str = "-"
    refresh_local_str = "-"
    countdown_str = "-"
    reset_in_hours = None
    if isinstance(refresh_raw, str) and refresh_raw:
        refresh_dt = datetime.fromisoformat(refresh_raw.replace("Z", "+00:00"))
        refresh_utc_str = refresh_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        local_dt = refresh_dt.astimezone()
        tz_off = local_dt.strftime("%z")
        if len(tz_off) == 5:
            tz_off = f"UTC{tz_off[:3]}:{tz_off[3:]}"
        refresh_local_str = f"{local_dt.strftime('%Y-%m-%d %H:%M')} {tz_off}".strip()
        delta_h = max(0.0, (refresh_dt - now).total_seconds() / 3600.0)
        reset_in_hours = round(delta_h, 2)
        countdown_str = _format_hm(delta_h)

    return {
        "account": label,
        "username": username,
        "remaining_hours": round(remaining_s / 3600.0, 2),
        "remaining_hm": _format_hm(remaining_s / 3600.0),
        "used_hours": round(used_s / 3600.0, 2),
        "reserved_hours": round(reserved_s / 3600.0, 2),
        "total_hours": round(total_s / 3600.0, 2),
        "refresh_time_utc": refresh_utc_str,
        "refresh_time_local": refresh_local_str,
        "reset_in": countdown_str,
        "reset_in_hours": reset_in_hours,
    }


def fetch_account_quota(label: str, cred_path: Path, *, session: requests.Session | None = None, timeout=(10, 20)) -> dict:
    raw = json.loads(Path(cred_path).read_text(encoding="utf-8"))
    username = raw.get("username", "")
    key = raw.get("key", "")
    if not username or not key:
        raise ValueError(f"Incomplete credentials for {label}")
    http = session or requests.Session()
    response = http.get(QUOTA_URL, auth=(username, key), timeout=timeout)
    response.raise_for_status()
    return parse_quota_payload(label, username, response.json())


def format_table(rows: list[dict]) -> str:
    headers = ("Account", "Username", "Remaining GPU", "Used / Total", "Reset Time (Local / In)")
    table_rows = []
    for r in rows:
        if "error" in r:
            table_rows.append((r["account"], r.get("username", "-"), f"ERROR: {r['error']}", "-", "-"))
        else:
            rem = f"{r['remaining_hours']:6.2f} h ({r['remaining_hm']})"
            used_tot = f"{r['used_hours']:5.2f} / {r['total_hours']:4.1f} h"
            if r["reserved_hours"] > 0:
                used_tot += f" (+{r['reserved_hours']:.1f}h rsv)"
            reset = f"{r['refresh_time_local']} (in {r['reset_in']})"
            table_rows.append((r["account"], r["username"], rem, used_tot, reset))

    widths = [max(len(str(row[i])) for row in [headers, *table_rows]) for i in range(len(headers))]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = lambda row: "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
    lines = [sep, fmt(headers), sep]
    for row in table_rows:
        lines.append(fmt(row))
    lines.append(sep)
    return "\n".join(lines)


def _resolve_targets(cred_args: list[str] | None, key_dir: Path) -> list[tuple[str, Path]]:
    if cred_args:
        targets = []
        for item in cred_args:
            if "=" in item:
                label, path_str = item.split("=", 1)
                targets.append((label.strip(), Path(path_str.strip())))
            else:
                p = Path(item.strip())
                targets.append((p.parent.name, p))
        return targets
    return [(label, key_dir / rel) for label, rel in DEFAULT_ACCOUNTS]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-dir", type=Path, default=DEFAULT_KEY_DIR, help="Directory containing Kaggle key subfolders")
    parser.add_argument("--credentials", action="append", help="Optional label=path/to/kaggle.json override")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    targets = _resolve_targets(args.credentials, args.key_dir)
    results = []
    with requests.Session() as session:
        for label, path in targets:
            try:
                results.append(fetch_account_quota(label, path, session=session))
            except Exception as exc:
                results.append({"account": label, "error": type(exc).__name__})

    if args.json:
        print(json.dumps({"accounts": results, "observed_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2))
    else:
        print(format_table(results))
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

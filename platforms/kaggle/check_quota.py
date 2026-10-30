"""Read-only Kaggle GPU quota and reset-time monitor for configured accounts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from molgap.common.kaggle import api_for_credentials
from molgap.kaggle_submission import sdk_quota_payload


# There is intentionally no desktop-specific credential directory fallback.
DEFAULT_KEY_DIR = None
DEFAULT_ACCOUNTS = (
    ("kaggle1", ".kaggle1_default/kaggle.json"),
    ("kaggle2", ".kaggle2_cli/kaggle.json"),
    ("kaggle3", ".kaggle3_cli/kaggle.json"),
)


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
        if refresh_dt.tzinfo is None:
            # Kaggle reports the refresh time as naive UTC.
            refresh_dt = refresh_dt.replace(tzinfo=timezone.utc)
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


def fetch_account_quota(label: str, cred_path: Path) -> dict:
    raw = json.loads(Path(cred_path).read_text(encoding="utf-8"))
    username = raw.get("username", "")
    key = raw.get("key", "")
    if not username or not key:
        raise ValueError(f"Incomplete credentials for {label}")
    return parse_quota_payload(label, username, sdk_quota_payload(api_for_credentials(cred_path)))


def format_table(rows: list[dict]) -> str:
    headers = ("Account", "Username", "Remaining GPU", "Used / Total", "Reset Time (Local / In)")
    table_rows = []
    for row in rows:
        if "error" in row:
            table_rows.append((row["account"], row.get("username", "-"), f"ERROR: {row['error']}", "-", "-"))
        else:
            rem = f"{row['remaining_hours']:6.2f} h ({row['remaining_hm']})"
            used_tot = f"{row['used_hours']:5.2f} / {row['total_hours']:4.1f} h"
            if row["reserved_hours"] > 0:
                used_tot += f" (+{row['reserved_hours']:.1f}h rsv)"
            reset = f"{row['refresh_time_local']} (in {row['reset_in']})"
            table_rows.append((row["account"], row["username"], rem, used_tot, reset))

    widths = [max(len(str(row[i])) for row in [headers, *table_rows]) for i in range(len(headers))]
    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    fmt = lambda row: "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
    lines = [sep, fmt(headers), sep]
    for row in table_rows:
        lines.append(fmt(row))
    lines.append(sep)
    return "\n".join(lines)


def _resolve_targets(cred_args: list[str] | None, key_dir: Path | None) -> list[tuple[str, Path]]:
    if cred_args:
        targets = []
        labels = set()
        for item in cred_args:
            if "=" in item:
                label, path_str = item.split("=", 1)
                label, path_str = label.strip(), path_str.strip()
                if not label or not path_str:
                    raise ValueError("Credentials must be given as label=path")
            else:
                p = Path(item.strip())
                label, path_str = p.parent.name, str(p)
            if label in labels:
                raise ValueError("Duplicate Kaggle account label")
            labels.add(label)
            targets.append((label, Path(path_str)))
        return targets
    if key_dir is None:
        configured = os.environ.get("MOLGAP_KAGGLE_KEY_DIR")
        if not configured:
            raise ValueError("Kaggle credentials require --key-dir or MOLGAP_KAGGLE_KEY_DIR")
        key_dir = Path(configured)
    return [(label, key_dir / relative) for label, relative in DEFAULT_ACCOUNTS]


class _FiniteParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("Invalid quota arguments")


def main(argv=None) -> int:
    parser = _FiniteParser(description=__doc__)
    parser.add_argument("--key-dir", type=Path,
                        help="Directory containing Kaggle key subfolders; may also be set by MOLGAP_KAGGLE_KEY_DIR")
    parser.add_argument("--credentials", action="append", help="Optional label=path/to/kaggle.json override")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    try:
        args = parser.parse_args(argv)
        targets = _resolve_targets(args.credentials, args.key_dir)
    except Exception as exc:
        result = {"status": "BLOCKED", "error": type(exc).__name__}
        print(json.dumps(result, ensure_ascii=False, indent=2) if getattr(locals().get("args", None), "json", False)
              else "Quota check blocked: " + result["error"])
        return 2

    results = []
    for label, path in targets:
        try:
            results.append(fetch_account_quota(label, path))
        except Exception as exc:
            # Never serialize credential contents, paths, or HTTP bodies.
            results.append({"account": label, "error": type(exc).__name__})

    if args.json:
        print(json.dumps({"accounts": results, "observed_at": datetime.now(timezone.utc).isoformat()},
                         ensure_ascii=False, indent=2))
    else:
        print(format_table(results))
    return 0 if all("error" not in row for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

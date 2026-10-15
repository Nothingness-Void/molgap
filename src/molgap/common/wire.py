"""Stable canonical fingerprints; existing owners import this implementation."""
import hashlib
import json
from collections.abc import Mapping


def canonical_fingerprint(value: Mapping) -> str:
    """Return a stable SHA-256 for a JSON-compatible mapping."""
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

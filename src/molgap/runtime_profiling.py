"""Stage timing primitives required before changing a V5 scientific contract."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


PROFILE_FORMAT = "molgap-runtime-profile-v1"
PROFILE_STAGES = (
    "loader_collate",
    "h2d",
    "forward_loss",
    "backward_optimizer",
    "validation",
    "checkpoint_hash_archive",
    "allocation",
)


class RuntimeProfile:
    """Collect separate stage samples without assuming low VRAM means low use."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {stage: [] for stage in PROFILE_STAGES}
        self.metadata: dict[str, Any] = {}

    def add(self, stage: str, duration_seconds: float) -> None:
        if stage not in self.samples:
            raise ValueError(f"Unknown profiling stage: {stage}")
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Profiling duration must be finite and non-negative")
        self.samples[stage].append(duration)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - started)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PROFILE_FORMAT,
            "stages": {
                stage: {
                    "samples_seconds": list(values),
                    "count": len(values),
                    "total_seconds": sum(values),
                }
                for stage, values in self.samples.items()
            },
            "metadata": dict(self.metadata),
        }

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        temporary: Optional[Path] = None
        try:
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

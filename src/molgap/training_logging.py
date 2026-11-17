"""Shared training telemetry; canonical traces and checkpoints own acceptance.

Reporting reads Python counters and already computed pass metrics. It never
evaluates a model, reads labels, synchronizes a tensor or advances an RNG.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from .training_reproducibility import atomic_json


LOG_FORMAT = "molgap-training-log-v1"
LOG_PREFIX = "TRAIN_LOG "


@dataclass(frozen=True)
class TrainingLogPolicy:
    mode: str = "epoch"
    interval: float | int | None = None

    def __post_init__(self):
        if self.mode == "epoch":
            valid = self.interval is None
        elif self.mode == "batches":
            valid = type(self.interval) is int and self.interval > 0
        elif self.mode == "seconds":
            valid = (type(self.interval) in (int, float) and
                     math.isfinite(self.interval) and self.interval > 0)
        else:
            valid = False
        if not valid:
            raise ValueError("Logging requires epoch/null, batches/positive integer or seconds/finite positive number")

    @classmethod
    def from_dict(cls, value):
        if value is None:
            return cls()
        if type(value) is not dict or set(value) != {"mode", "interval"}:
            raise ValueError("Expected logging mode and interval only")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


class TrainingLogger:
    """One schema for every arm, frequency and platform using the shared session."""

    def __init__(self, root, context, recipe, policy, *, observations=(), clock=None, emit=None,
                 observation_train_unit="normalized-Gap"):
        if type(policy) is not TrainingLogPolicy:
            raise TypeError("Expected TrainingLogPolicy")
        self.root, self.context, self.recipe, self.policy = Path(root), context, recipe, policy
        self.clock = clock or time.perf_counter
        self.emit = emit or (lambda line: print(line, flush=True))
        self.started = self.last_emitted_at = self.last_pass_at = self.clock()
        self.start_step = self.checkpoint_step = 0
        self.started_midpass = False
        self.last_emitted = None
        self.pending_pass = None
        self.metric_pass = self.best_pass = None
        self.train_mae = self.dev_mae = self.best_mae = None
        if observation_train_unit not in {"normalized-Gap", "eV"}:
            raise ValueError("Unsupported retained training metric unit")
        for row in observations:
            if row.get("event") != "observation":
                continue
            train = row.get("live_train_metric")
            dev = row.get("ema_dev_metric" if recipe.selection.endswith("ema") else "live_dev_metric")
            if train is not None:
                self._metrics(row["epoch_or_pass"], train, dev, train_unit=observation_train_unit)

    def _metrics(self, pass_index, train, development, *, train_unit="normalized-Gap"):
        values = (train,) if development is None else (train, development)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Training telemetry requires finite nonnegative MAE")
        self.metric_pass = pass_index
        self.train_mae = train if train_unit == "eV" else train * self.recipe.target_std_eV
        self.dev_mae = development
        if development is not None and (self.best_mae is None or development < self.best_mae):
            self.best_mae, self.best_pass = development, pass_index

    def start(self, progress):
        """Timing covers this invocation; retained trace metrics keep their pass."""
        self.started = self.last_emitted_at = self.last_pass_at = self.clock()
        self.start_step = self.checkpoint_step = progress.optimizer_step
        self.started_midpass = progress.next_batch != 0

    def pass_finished(self, progress, metrics):
        evaluation = metrics["development"]
        self._metrics(progress.pass_index, metrics["normalized_train_gap_mae"],
                      None if evaluation is None else evaluation["mae_eV"])
        self.pending_pass = progress.optimizer_step

    def checkpoint_finished(self, progress):
        self.checkpoint_step = progress.optimizer_step

    def progress(self, progress, *, learning_rates):
        now = self.clock()
        epoch = self.pending_pass == progress.optimizer_step
        terminal = progress.status != "running"
        due = ((self.policy.mode == "seconds" and now - self.last_emitted_at >= self.policy.interval) or
               (self.policy.mode == "batches" and progress.optimizer_step % self.policy.interval == 0))
        if not (epoch or terminal or due) or self.last_emitted == (progress.optimizer_step, progress.status):
            return
        elapsed = max(0., now - self.started)
        updates = progress.optimizer_step - self.start_step
        rate = updates / elapsed if elapsed > 0 and updates > 0 else None
        weights = ("clean_ema" if getattr(self.recipe, "clean_bn", None) is not None else
                   "ema" if self.recipe.selection.endswith("ema") else "live")
        record = {
            "format": LOG_FORMAT, "event": "epoch" if epoch else "terminal" if terminal else "progress",
            "arm_id": self.context.arm_id, "family": self.recipe.family, "weight_state": weights,
            "status": progress.status, "completed_passes": progress.pass_index,
            "planned_passes": self.recipe.optimizer_steps / self.recipe.loop_plan.steps_per_pass,
            "next_batch": progress.next_batch, "optimizer_steps": progress.optimizer_step,
            "planned_optimizer_steps": self.recipe.optimizer_steps,
            "sample_presentations": progress.sample_presentations,
            "train_live_mae_eV": self.train_mae, "development_mae_eV": self.dev_mae,
            "best_development_mae_eV": self.best_mae, "metric_pass": self.metric_pass,
            "best_metric_pass": self.best_pass, "learning_rates": list(learning_rates),
            "elapsed_seconds": elapsed, "updates_per_second": rate,
            "eta_seconds": max(0, self.recipe.optimizer_steps - progress.optimizer_step) / rate if rate else None,
            "pass_wall_seconds": max(0., now - self.last_pass_at) if epoch else None,
            "pass_timing_scope": ("invocation_partial_pass" if self.started_midpass else "full_pass") if epoch else None,
            "checkpoint_optimizer_steps": self.checkpoint_step,
        }
        line = LOG_PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        atomic_json(self.root / "training_progress.json", record)
        self.emit(line)
        self.last_emitted, self.last_emitted_at = (progress.optimizer_step, progress.status), now
        if epoch:
            self.pending_pass = None
            self.last_pass_at, self.started_midpass = now, False


def forward_training_logs(path, offset, *, arm_id, emit=None):
    """Forward complete structured worker lines once, including a child's last line.

    Ordinary worker diagnostics remain in train.log. Incomplete writes are read
    again on the next poll. Historical log bytes are skipped by the caller.
    """
    emit = emit or (lambda line: print(line, flush=True))
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        while True:
            begin = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                return begin
            if line.startswith(LOG_PREFIX.encode()):
                try:
                    record = json.loads(line[len(LOG_PREFIX):])
                    rendered = LOG_PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
                except (ValueError, UnicodeError):
                    continue
                if type(record) is dict and record.get("format") == LOG_FORMAT and record.get("arm_id") == arm_id:
                    emit(rendered)

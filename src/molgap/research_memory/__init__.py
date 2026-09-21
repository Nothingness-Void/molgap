"""Deterministic, evidence-linked research memory for MolGap."""

from .compiler import compile_research_memory, rebuild_research_memory
from .validate import validate_repository_records
from .trace import RMLTraceRecorder, canonicalize_trace, load_canonical_trace, trace_digest, validate_canonical_trace
from .finalize import finalize
from .plan import plan
from .recovery import recover_trace
from .pipeline import finalize_rebuild_backtest
from .terminal_wiring import (
    build_default_trace_manifest,
    close_terminal_arm,
    close_terminal_multi_arm,
    inspect_trace_retention_evidence,
    is_trace_artifact,
    resolve_trace_for_terminal_arm,
)

__all__ = [
    "compile_research_memory",
    "rebuild_research_memory",
    "validate_repository_records",
    "RMLTraceRecorder",
    "canonicalize_trace",
    "load_canonical_trace",
    "trace_digest",
    "validate_canonical_trace",
    "finalize",
    "plan",
    "recover_trace",
    "finalize_rebuild_backtest",
    "build_default_trace_manifest",
    "close_terminal_arm",
    "close_terminal_multi_arm",
    "inspect_trace_retention_evidence",
    "is_trace_artifact",
    "resolve_trace_for_terminal_arm",
]

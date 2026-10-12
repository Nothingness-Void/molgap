"""Execution declaration checks; no molecular models, training or network."""
import ast
import inspect
from pathlib import Path

from molgap.gptrans_execution_acceptance import expected_execution_fields
from molgap.gptrans_execution_prepare import freeze
from molgap.gptrans_execution_tuning import ExecutionTuning
from dataclasses import asdict


def test_original_fields_are_unchanged_by_tuning():
    tuning = asdict(ExecutionTuning(independent_models=1, total_cpu_budget=2))
    assert expected_execution_fields("execution_original", tuning=tuning) == expected_execution_fields("execution_original")


def test_tuning_identity_is_frozen_without_model_execution():
    tuning = ExecutionTuning(independent_models=1, total_cpu_budget=2)
    old = expected_execution_fields("execution_optimized")
    new = expected_execution_fields("execution_optimized", tuning=asdict(tuning))
    assert new.pop("execution_backend") == "optimized-tuning-v3:" + tuning.identity
    old.pop("execution_backend")
    assert old == new


def test_preparer_keeps_historical_default_and_has_explicit_new_scope():
    signature = inspect.signature(freeze)
    assert signature.parameters["base"].default == "experiments/pcqm_gptrans_execution40"
    assert signature.parameters["tuning"].default is None


def test_thin_entrypoints_and_completion_report_binding():
    root = Path(__file__).parents[1]
    for name in ("freeze.py", "accept.py"):
        tree = ast.parse((root / "experiments/pcqm_gptrans_execution_tuning40" / name).read_text())
        assert not any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
    text = (root / "src/molgap/gptrans_execution_study.py").read_text()
    assert 'completion["files"][report_path.name] = sha256_file(report_path)' in text
    assert "torch.cuda.max_memory_allocated()" in text

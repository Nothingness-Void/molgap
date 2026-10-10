"""Retained _save plus synthetic metadata: formatting changes only trace bytes."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from molgap.training_reproducibility import (
    assert_finite_state_dict, atomic_json, atomic_torch_save, sha256_file,
)
from molgap.edge_state_training_core import EpochPermutationBatchSampler
from test_k1_sender_sampler_reuse import SOURCE, save_node


def execute(output, *, compact):
    node = save_node(False)
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "atomic_json"
             and "trace.json" in ast.unparse(n.args[0])]
    assert len(calls) == 1 and not calls[0].keywords
    if compact:
        calls[0].keywords.append(ast.keyword(arg="compact", value=ast.Constant(value=True)))
    arms = {name: dict(epoch=1, offset=3, steps=10, samples=1280, best=0.12,
                       best_epoch=0, trace=[dict(epoch=0, calibrated_mae_eV=0.12)],
                       step_trace=[dict(step=1, loss=1.2345678901234567,
                                        seconds=0.125, allocation_seconds=1.0)])
            for name in ("reference", "ema999")}
    namespace = {
        "_snapshot": lambda arm: {**copy.deepcopy(arm), "model": {"fixture": torch.tensor([1.0])}},
        "_sampler": lambda graphs, epoch, offset: EpochPermutationBatchSampler(
            len(graphs), 128, seed=42, epoch=epoch, start_batch=offset),
        "assert_finite_state_dict": assert_finite_state_dict,
        "atomic_json": atomic_json, "atomic_torch_save": atomic_torch_save,
        "sha256_file": sha256_file, "FORMAT": "synthetic-fixture",
        "time": SimpleNamespace(time=lambda: 100.0), "json": json,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace["_save"](output, arms, {"binding": "fixture"},
                              {"runtime": "fixture"}, 999, "RUNNING", 10.0)


def test_compact_trace_updates_actual_receipt_hash_without_changing_checkpoint(tmp_path):
    before_rng = torch.get_rng_state().clone()
    before_cuda = torch.cuda.is_initialized()
    original, candidate = tmp_path / "original", tmp_path / "candidate"
    receipts = [execute(original, compact=False), execute(candidate, compact=True)]
    traces = [json.loads((root / "trace.json").read_text()) for root in (original, candidate)]
    assert traces[0] == traces[1]
    assert sha256_file(original / "last.pt") == sha256_file(candidate / "last.pt")
    assert (candidate / "trace.json").stat().st_size < (original / "trace.json").stat().st_size
    assert receipts[0]["trace_sha256"] != receipts[1]["trace_sha256"]
    for root, receipt in zip((original, candidate), receipts):
        assert receipt["trace_sha256"] == sha256_file(root / "trace.json")
        assert receipt == json.loads((root / "last.json").read_text())
    assert {k: v for k, v in receipts[0].items() if k != "trace_sha256"} == {
        k: v for k, v in receipts[1].items() if k != "trace_sha256"}
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.cuda.is_initialized() == before_cuda

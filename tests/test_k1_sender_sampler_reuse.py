"""Synthetic _save metadata parity; no import or execution of a trainer."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap.edge_state_training_core import (
    EpochPermutationBatchSampler,
    epoch_permutation_state,
)


SOURCE = (Path(__file__).resolve().parents[1] / "experiments/pcqm_k1_execution_efficiency/"
          "checkpoint_io_attempt_001/source_snapshot/colab_k1_screen.py")
SHA256 = "ac13918013015c27b1f46020b0eabf2b4f7ad0eb18bb091a9e6033b94948dc34"


def save_node(candidate):
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    node = next(n for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == "_save")
    if candidate:
        loop = node.body[1]
        assert isinstance(loop, ast.For)
        original = copy.deepcopy(loop.body)
        loop.body[:2] = ast.parse(
            'snapshots[name]["sampler"] = epoch_permutation_state('
            'train_rows, owner.BS, seed=42, epoch=arm["epoch"], next_batch=arm["offset"])'
        ).body
        assert ast.dump(loop.body[-1]) == ast.dump(original[-1])
    return ast.fix_missing_locations(node)


def exercise(candidate, epochs, offsets):
    from molgap.edge_state_training_core import _epoch_permutation_metadata

    _epoch_permutation_metadata.cache_clear()
    events = []
    arms = {}
    for name, epoch, offset in zip(("reference", "ema999"), epochs, offsets):
        arms[name] = dict(epoch=epoch, offset=offset, steps=epoch * 7 + offset,
                          samples=(epoch * 7 + offset) * 128, best=0.1,
                          best_epoch=0, step_trace=[{"step": 1, "loss": 0.25}],
                          trace=[{"epoch": e, "calibrated_mae_eV": 0.1 + e}
                                 for e in range(epoch)])

    def record(path, value):
        events.append((str(path), copy.deepcopy(value)))

    def sampler(graphs, epoch, offset=0):
        return EpochPermutationBatchSampler(len(graphs), 128, seed=42,
                                            epoch=epoch, start_batch=offset)

    namespace = {
        "_snapshot": lambda arm: {**copy.deepcopy(arm), "model": {"fixture": 1.0}},
        "_sampler": sampler, "epoch_permutation_state": epoch_permutation_state,
        "owner": SimpleNamespace(BS=128),
        "assert_finite_state_dict": lambda state, *, label: None,
        "atomic_torch_save": record, "atomic_json": record,
        "FORMAT": "synthetic-fixture", "sha256_file": lambda path: "a" * 64,
        "time": SimpleNamespace(time=lambda: 100.0),
        "json": SimpleNamespace(dumps=lambda value: "synthetic receipt"),
    }
    exec(compile(ast.Module(body=[save_node(candidate)], type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    first = namespace["_save"](Path("synthetic-output"), arms, {"binding": "fixture"},
                               {"runtime": "fixture"}, 999, "RUNNING", 10.0)
    second = namespace["_save"](Path("synthetic-output"), arms, {"binding": "fixture"},
                                {"runtime": "fixture"}, 999, "RUNNING", 10.0)
    return first, second, events, arms


@pytest.mark.parametrize("epochs", [(0, 0), (1, 0), (2, 1), (60, 60)])
@pytest.mark.parametrize("offsets", [(0, 0), (3, 7), (7, 7)])
def test_sender_checkpoint_receipts_and_artifact_payloads_unchanged(epochs, offsets, monkeypatch):
    original_randperm = torch.randperm
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original_randperm(*args, **kwargs)

    monkeypatch.setattr(torch, "randperm", counted)
    original = exercise(False, epochs, offsets)
    assert len(calls) == 4
    calls.clear()
    candidate = exercise(True, epochs, offsets)
    assert len(calls) == len(set(epochs))
    assert original == candidate

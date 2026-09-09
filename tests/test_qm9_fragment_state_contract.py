from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "qm9_fragment_state.py"
CLI = ROOT / "experiments" / "qm9_architecture" / "qm9_fragment_state.py"
PROTOCOL = ROOT / "experiments" / "qm9_architecture" / "fragment_state_protocol.md"
REMOTE = ROOT / "platforms" / "scnet" / "qm9_fragment_state"


def test_fragment_screen_contract_is_explicit_and_sealed():
    source = MODULE.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert 'BATCH_SIZE = 128' in source
    assert "FRAGMENT_BLOCKS = (3, 6, 9)" in source
    assert "FRAGMENT_ALGORITHM = \"brics-heavy-atom-components-v1\"" in source
    assert "official_pcqm_roles_read" in source
    assert "test_role_materialized" in source
    assert "candidate validation Gap MAE" in protocol
    assert "no QM9 test" in protocol and "graph is materialized" in protocol
    assert "pretrained weight" in protocol


def test_remote_sequence_requires_cpu_acceptance_then_preflight():
    accept = (REMOTE / "accept_cache_xian.slurm").read_text(encoding="utf-8")
    preflight = (REMOTE / "preflight_xian.slurm").read_text(encoding="utf-8")
    train = (REMOTE / "train_xian.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --gres" not in accept
    assert "test -s \"$CACHE_ROOT/acceptance.json\"" in preflight
    assert "test -s \"$ROOT/preflight/preflight.json\"" in train
    assert "--cache-sha256" in preflight and "--cache-sha256" in train
    assert "--batch-size" not in train
    assert "xahdtest" in accept and "xahdtest" in preflight and "xahdtest" in train


def test_fragment_batch_offsets_are_not_default_node_offsets():
    source = MODULE.read_text(encoding="utf-8")
    assert 'key in {"fragment_id", "fragment_edge_index"}' in source
    assert "fragment edge crosses molecules" in source
    assert "fragment counts do not match fragment features" in source


def test_cli_is_thin():
    source = CLI.read_text(encoding="utf-8")
    assert "from molgap.qm9_fragment_state import" in source
    assert "class OGBFragmentState" not in source

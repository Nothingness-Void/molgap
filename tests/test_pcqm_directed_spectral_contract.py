from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/molgap/pcqm_directed_spectral.py"
BASE = ROOT / "src/molgap/pcqm_gap_architecture.py"
RUNNER = ROOT / "src/molgap/pcqm_local_global_runner.py"
EXPERIMENT = ROOT / "experiments/pcqm_gap_architecture"
KERNELS = EXPERIMENT / "kaggle_pcqm_gap100k"


def test_directed_and_spectral_sources_parse_and_are_zero_start():
    paths = (
        MODEL,
        BASE,
        RUNNER,
        EXPERIMENT / "accept_pcqm100k_signnet_lappe_cache.py",
        EXPERIMENT / "accept_pcqm100k_directed_spectral_seed42.py",
        KERNELS / "signnet_lappe_cache/run_lappe_cache.py",
    )
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source = MODEL.read_text(encoding="utf-8")
    assert "incoming.index_add_(0, second, edge_state[first])" in source
    assert "phi(+v, lambda)" in source
    assert "self.signnet_phi(positive) + self.signnet_phi(negative)" in source
    assert "nn.init.zeros_(block[-1].weight)" in source
    assert "nn.init.zeros_(self.lappe_to_atom.weight)" in source


def test_runner_freezes_paired_seed42_t4_contracts():
    source = RUNNER.read_text(encoding="utf-8")
    for token in (
        '"directed_bond_graphstate"',
        '"signnet_lappe_graphstate"',
        "3_741_265",
        "3_673_109",
        "SEARCH_BUDGET_S = 14_400",
        'EXPECTED_GPU_TOKEN = "T4"',
        '"directed_bond_memory"',
        '"spectral_position_encoding"',
    ):
        assert token in source


def test_lappe_cache_is_cpu_only_role_sealed_and_hashable():
    metadata = json.loads(
        (KERNELS / "signnet_lappe_cache/kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["id"].startswith("kaseichou/")
    assert metadata["enable_gpu"] == "false"
    assert metadata["is_private"] == "true"
    source = (KERNELS / "signnet_lappe_cache/run_lappe_cache.py").read_text(
        encoding="utf-8"
    )
    assert "torch.linalg.eigh" in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "atomic_torch_save" in source
    acceptance = (
        EXPERIMENT / "accept_pcqm100k_signnet_lappe_cache.py"
    ).read_text(encoding="utf-8")
    assert "import torch" not in acceptance
    assert "model_inference_executed" in acceptance


def test_protocols_do_not_broaden_seed_or_roles():
    for name in (
        "directed_bond_graphstate_seed42_protocol.md",
        "signnet_lappe_graphstate_seed42_protocol.md",
    ):
        source = (EXPERIMENT / name).read_text(encoding="utf-8")
        assert "seed 42" in source
        assert "official validation" in source
        assert "test-dev" in source
        assert "does not authorize" in source

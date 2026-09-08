from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_gap_architecture"
MODEL = ROOT / "src/molgap/pcqm_relative_value.py"
RUNNER = ROOT / "src/molgap/pcqm_local_global_runner.py"
PROTOCOL = EXPERIMENT / "relative_value_graphstate_seed42_protocol.md"
KERNEL = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k/relative_value_graphstate_seed42/kernel-metadata.json"
)
ENTRY = KERNEL.with_name("run_screen.py")


def test_sources_parse() -> None:
    for path in (MODEL, RUNNER, ENTRY):
        ast.parse(path.read_text(encoding="utf-8"))


def test_model_contract_is_relative_key_value_attention() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert "path_key" in source
    assert "path_value" in source
    assert "path_bias" in source
    assert "softmax(score, target" in source
    assert "torch.log1p(count)" in source
    assert "nn.init.zeros_(self.output_value.weight)" in source
    assert "PATH_BLOCKS" in source


def test_runner_and_protocol_lock_roles_and_budget() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert '"relative_value_graphstate"' in runner
    assert "3_734_977" in runner
    assert "official validation and test-dev: unread" in protocol
    assert "seed: 42" in protocol
    assert "precision: FP32" in protocol
    assert "parameter ceiling: 4,000,000" in protocol
    assert "paired gain at least `0.001 eV`" in protocol


def test_kernel_requests_t4x2_capable_shape_and_frozen_inputs() -> None:
    metadata = json.loads(KERNEL.read_text(encoding="utf-8"))
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["is_private"] == "true"
    assert metadata["dataset_sources"] == [
        "nothingnessvoid/molgap-pcqm-relative-value-source"
    ]
    assert metadata["kernel_sources"] == [
        "nothingnessvoid/molgap-pcqm-hop-path-cache-s42"
    ]
    entry = ENTRY.read_text(encoding="utf-8")
    assert "MOLGAP_LOCAL_GLOBAL_RUN_MODE" in entry
    assert "relative_value_graphstate" in entry
    assert "__FROZEN_SOURCE_COMMIT__" in entry


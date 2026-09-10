import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "qm9_cardinality.py"
EXPERIMENT = ROOT / "experiments" / "qm9_cardinality_channel"
RUNNER = EXPERIMENT / "gpu_seed42" / "run_screen.py"
METADATA = RUNNER.with_name("kernel-metadata.json")


def test_cardinality_source_and_runner_parse():
    ast.parse(MODULE.read_text(encoding="utf-8"))
    ast.parse(RUNNER.read_text(encoding="utf-8"))
    ast.parse((EXPERIMENT / "accept.py").read_text(encoding="utf-8"))


def test_frozen_batch_and_roles_are_static():
    source = MODULE.read_text(encoding="utf-8")
    assert "BATCH_SIZE = 128" in source
    assert "TRAIN_ROWS = 30_000" in source
    assert "VALIDATION_ROWS = 3_000" in source
    assert 'MAX_HOPS = 3' in source
    assert 'CHANNEL_LAYERS = (3, 6, 9)' in source
    assert 'MIN_GAIN_VS_BASELINE_EV = 0.003' in source
    assert 'MIN_GAIN_VS_SIZE_CONTROL_EV = 0.001' in source
    assert 'test_role_read": False' in source
    assert "official_pcqm_roles_read" in source


def test_candidate_uses_unnormalized_same_support_sum():
    source = MODULE.read_text(encoding="utf-8")
    assert '"bij,bjhd->bihd"' in source
    assert "support.to(hidden.dtype)" in source
    assert "torch.log1p" in source
    assert "softmax" not in source
    assert "nn.init.zeros_(self.output.weight)" in source


def test_zero_return_preflight_allows_only_tight_cuda_roundoff():
    source = MODULE.read_text(encoding="utf-8")
    assert "torch.allclose(" in source
    assert "zero_return_atol = 1e-7" in source
    assert "zero_return_rtol = 1e-6" in source
    assert '"zero_return_max_abs_difference"' in source
    assert "torch.equal(output, baseline_output)" not in source


def test_kernel_requests_t4x2_source_and_accepted_cache():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-qm9-cardinality-channel-s42"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] == "true"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-qm9-cardinality-channel-source",
        "kaseichou/molgap-qm9-gape-lite-cache",
    ]
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'torch.cuda.device_count() != 2' in runner
    assert 'CUDA_VISIBLE_DEVICES"] = "0" if role == "baseline" else "1"' in runner


def test_protocol_excludes_desktop_and_privileged_inputs():
    protocol = (EXPERIMENT / "protocol.md").read_text(encoding="utf-8")
    assert "physical batch 128" in protocol
    assert "desktop 304-wide" in protocol
    assert "No held-out/test graph" in protocol
    assert "teacher distillation" in protocol
    assert "PCQM-100K" in protocol

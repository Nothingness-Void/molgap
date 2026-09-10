import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "qm9_fourier_edge.py"
EXPERIMENT = ROOT / "experiments" / "qm9_fourier_edge"
RUNNER = EXPERIMENT / "gpu_seed42" / "run_screen.py"
METADATA = RUNNER.with_name("kernel-metadata.json")


def test_fourier_edge_source_and_runner_parse():
    ast.parse(MODULE.read_text(encoding="utf-8"))
    ast.parse(RUNNER.read_text(encoding="utf-8"))
    ast.parse((EXPERIMENT / "accept.py").read_text(encoding="utf-8"))


def test_frozen_contract_is_static():
    source = MODULE.read_text(encoding="utf-8")
    assert "BATCH_SIZE = 128" in source
    assert "TRAIN_ROWS = 30_000" in source
    assert "VALIDATION_ROWS = 3_000" in source
    assert "HARMONICS = 1" in source
    assert "MIN_GAIN_VS_FULL_GPS_EV = 0.003" in source
    assert "MIN_GAIN_VS_K1_EV = 0.001" in source
    assert 'test_role_read": False' in source
    assert "official_pcqm_roles_read" in source


def test_only_edge_proposal_uses_single_harmonic_fourier_kan():
    source = MODULE.read_text(encoding="utf-8")
    assert "edge_update.update = nn.Sequential(" in source
    assert "_FourierKANFactory.make(" in source
    assert '"d...ik,doik->...o"' in source
    assert "torch.cos(phase)" in source
    assert "torch.sin(phase)" in source
    assert "omit_edge_proposal=True" in source
    assert "manual_equation_match" in source


def test_kernel_requests_t4x2_and_reuses_accepted_cache():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-qm9-fourier-edge-s42"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] == "true"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-qm9-fourier-edge-source",
        "kaseichou/molgap-qm9-gape-lite-cache",
    ]
    runner = RUNNER.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in runner
    assert 'CUDA_VISIBLE_DEVICES"] = "0" if role == "anchor" else "1"' in runner


def test_protocol_excludes_confounded_kagnn_inputs():
    protocol = (EXPERIMENT / "protocol.md").read_text(encoding="utf-8")
    assert "physical batch 128" in protocol
    assert "non-covalent edge" in protocol
    assert "No QM9 held-out/test graph" in protocol
    assert "official PCQM validation/test-dev" in protocol
    assert "3D/ETKDG" in protocol

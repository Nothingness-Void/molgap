import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "qm9_neural_atom.py"
EXPERIMENT = ROOT / "experiments" / "qm9_neural_atom_mixer"
RUNNER = EXPERIMENT / "gpu_seed42" / "run_screen.py"
METADATA = RUNNER.with_name("kernel-metadata.json")


def test_neural_atom_source_and_runner_parse():
    ast.parse(MODULE.read_text(encoding="utf-8"))
    ast.parse(RUNNER.read_text(encoding="utf-8"))
    ast.parse((EXPERIMENT / "accept.py").read_text(encoding="utf-8"))


def test_frozen_batch_roles_and_attempt_are_static():
    source = MODULE.read_text(encoding="utf-8")
    assert "BATCH_SIZE = 128" in source
    assert "TRAIN_ROWS = 30_000" in source
    assert "VALIDATION_ROWS = 3_000" in source
    assert "MIXER_LAYERS = (3, 6, 9)" in source
    assert "MAX_SLOTS = 4" in source
    assert "LATENT_CHANNELS = 64" in source
    assert "MIN_GAIN_VS_BASELINE_EV = 0.003" in source
    assert "MIN_GAIN_VS_ONE_SLOT_EV = 0.001" in source
    assert 'test_role_read": False' in source
    assert "official_pcqm_roles_read" in source


def test_candidate_replaces_global_attention_with_parameter_matched_slots():
    source = MODULE.read_text(encoding="utf-8")
    assert "del self.convs" in source
    assert "self.local_blocks" in source
    assert 'active_slots = 1 if mode == "neural_atom_k1" else MAX_SLOTS' in source
    assert "self.slot_seed[: self.active_slots]" in source
    assert '"bkn,bnd->bkd"' in source
    assert '"bkn,bkd->bnd"' in source
    assert "nn.init.zeros_(self.return_projection.weight)" in source
    assert 'initial_sha != candidate_initial_sha' in source
    assert 'parameter_count != candidate_parameter_count' in source


def test_remote_preflight_uses_direct_invariants():
    source = MODULE.read_text(encoding="utf-8")
    assert "def compute_update(" in source
    assert "torch.count_nonzero(update)" in source
    assert "assignment.sum(dim=-1)" in source
    assert "finite_return_projection_gradients" in source
    assert "torch.equal(output, baseline_output)" not in source


def test_kernel_requests_t4x2_source_and_accepted_cache():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-qm9-neural-atom-s42"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] == "true"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-qm9-neural-atom-source",
        "kaseichou/molgap-qm9-gape-lite-cache",
    ]
    runner = RUNNER.read_text(encoding="utf-8")
    assert "torch.cuda.device_count() != 2" in runner
    assert 'CUDA_VISIBLE_DEVICES"] = "0" if role == "baseline" else "1"' in runner


def test_protocol_excludes_closed_or_privileged_inputs():
    protocol = (EXPERIMENT / "protocol.md").read_text(encoding="utf-8")
    assert "physical batch 128" in protocol
    assert "desktop 304-wide" in protocol
    assert "No held-out/test graph" in protocol
    assert "teacher distillation" in protocol
    assert "PCQM-100K" in protocol
    assert "attempt" in protocol.lower()

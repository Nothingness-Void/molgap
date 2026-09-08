import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "qm9_local_hierarchy.py"
PROTOCOL = ROOT / "experiments" / "qm9_local_hierarchy" / "protocol.md"


def _tree():
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_track_c_sizes_and_equal_compute_are_frozen():
    namespace = {}
    for node in _tree().body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(
                node.value, (ast.Constant, ast.Dict)
            ):
                namespace[target.id] = ast.literal_eval(node.value)
    assert namespace["TRAIN_ROWS"] == 30_000
    assert namespace["VALIDATION_ROWS"] == 3_000
    assert namespace["HELD_OUT_ROWS"] == 3_000
    assert namespace["SCRATCH_EPOCHS"] == 40
    assert namespace["PRETRAIN_EPOCHS"] + namespace["FINETUNE_EPOCHS"] == 40
    assert len(namespace["FUNCTIONAL_GROUP_SMARTS"]) == 12


def test_local_supervision_is_not_graph_histogram_surrogate():
    source = MODULE.read_text(encoding="utf-8")
    assert "functional_group_y[node_mask]" in source
    assert "original_x[node_mask, column]" in source
    assert "original_edge[edge_mask, column]" in source
    assert "histogram" not in source.lower()
    assert "moments" not in source.lower()


def test_cache_path_uses_model_free_qm9_data_module():
    source = MODULE.read_text(encoding="utf-8")
    assert "from .qm9_data import fixed_split" in source
    assert "from .qm9_screen import" not in source


def test_sealed_roles_and_dynamic_drift_gate_are_explicit():
    source = MODULE.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert '"test_role_read": False' in source
    assert '"official_pcqm_roles_read": False' in source
    assert "max(MIN_GAIN_EV, 2.0 * control_spread)" in source
    assert '"rng": _rng_state(train_loader)' in source
    assert '_restore_rng(checkpoint["rng"]' in source
    assert "unmaterialized" in protocol
    assert "does not authorize" in protocol


def test_slurm_separates_cpu_cache_and_dcu_training():
    cpu = (PROTOCOL.parent / "build_cache_kunshan.slurm").read_text(
        encoding="utf-8"
    )
    gpu = (PROTOCOL.parent / "train_kunshan.slurm").read_text(
        encoding="utf-8"
    )
    preflight = (PROTOCOL.parent / "preflight_kunshan.slurm").read_text(
        encoding="utf-8"
    )
    setup = (PROTOCOL.parent / "setup_cache_env_kunshan.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=kshctest02" in cpu
    assert "--gres=dcu" not in cpu
    assert "#SBATCH --partition=kshdtest" in gpu
    assert "#SBATCH --gres=dcu:Hygon:1" in gpu
    assert 'test -s "$ROOT/cache/acceptance.json"' in gpu
    assert "--gres=dcu:Hygon:1" in preflight
    assert "--time=00:15:00" in preflight
    assert '--preflight "$ROOT/preflight/preflight.json"' in gpu
    assert "cpu-cache-py311/bin/python" in setup
    assert "--no-index --no-deps" in setup
    assert 'WHEELHOUSE="$ROOT/wheelhouse-d834486"' in setup
    assert 'PYTHONPATH="$ROOT/cpu-deps:$CODE/src"' in cpu


def test_remote_preflight_checks_exact_model_and_gradients():
    source = MODULE.read_text(encoding="utf-8")
    assert "def run_preflight" in source
    assert '"finite_forward_backward": finite' in source
    assert '"inference_parameter_count"' in source
    assert '"geometry_valid": False' in source
    assert "loss.backward()" in source


def test_acceptance_never_imports_or_runs_model():
    source = (PROTOCOL.parent / "accept.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "make_encoder" not in source
    assert "model_inference_executed" in source

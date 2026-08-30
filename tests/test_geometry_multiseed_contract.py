from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"
PACKAGE = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "geometry_bottom_fusion_multiseed"
)
RUNNER = PACKAGE / "run_confirmation.py"
METADATA = PACKAGE / "kernel-metadata.json"
ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_geometry_multiseed.py"
PROTOCOL = EXPERIMENT / "geometry_bottom_fusion_multiseed_protocol.md"


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values


def test_geometry_multiseed_metadata_is_one_private_gpu_task() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-pcqm-geometry-confirm-s4344"
    assert metadata["code_file"] == "run_confirmation.py"
    assert metadata["enable_gpu"] == "true"
    assert metadata["is_private"] == "true"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-pcqm-geometry-fusion-source"
    ]
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pcqm-geometry-cache-s42"
    ]


def test_geometry_multiseed_runner_freezes_only_the_winner_and_comparator() -> None:
    values = _literal_assignments(RUNNER)
    assert values["COMPARATOR"] == "ogb_sparse_triangle_edge_state_gps9"
    assert values["CANDIDATE"] == (
        "ogb_distance_angle_triangle_edge_state_gps9"
    )
    source = RUNNER.read_text(encoding="utf-8")
    assert "CANDIDATES = (COMPARATOR, CANDIDATE)" in source
    assert "ogb_distance_triangle_edge_state_gps9" not in source
    assert "ogb_angle_triangle_edge_state_gps9" not in source


def test_geometry_multiseed_training_contract_is_frozen() -> None:
    values = _literal_assignments(RUNNER)
    assert values["SEEDS"] == (43, 44)
    assert values["BATCH_SIZE"] == 48
    assert values["LEARNING_RATE"] == 1.6e-4
    assert values["WEIGHT_DECAY"] == 1.0e-6
    assert values["MAX_EPOCHS"] == 40
    assert values["PATIENCE"] == 8
    assert values["PARAMETER_BUDGET"] == 5_200_000
    assert values["SEARCH_BUDGET_S"] == 39_600
    source = RUNNER.read_text(encoding="utf-8")
    assert "torch.Generator().manual_seed(seed)" in source
    assert '"target": "gap"' in source
    assert '"precision": "fp32"' in source


def test_geometry_multiseed_cache_and_seed42_references_are_frozen() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "e083bee19ee6a13cd9f72e91229752a9d5f56389" in source
    assert "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21" in source
    assert "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406" in source
    assert "0.13790177369117737" in source
    assert "0.13559719920158386" in source


def test_geometry_multiseed_preserves_atomic_outputs_and_pascal_support() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "os.replace(temporary, path)" in source
    for artifact in (
        '"best_model.pt"',
        '"checkpoint.pt"',
        '"validation_payload.pt"',
        '"trace.json"',
        '"metrics.json"',
        '"progress.json"',
        '"selection.json"',
        '"failure.json"',
    ):
        assert artifact in source
    assert '"torch==2.7.1"' in source
    assert '"sm_60" in set(torch.cuda.get_arch_list())' in source
    assert '"train_generator_state": train_generator.get_state()' in source
    assert '"torch_rng_state": torch.get_rng_state()' in source
    assert '"cuda_rng_state_all": torch.cuda.get_rng_state_all()' in source
    assert 'train_generator.set_state(checkpoint["train_generator_state"])' in source


def test_geometry_multiseed_keeps_sealed_roles_closed() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "piero0/pcqm4mv2" not in source
    assert "/lustre" not in source


def test_geometry_multiseed_acceptance_is_no_inference_and_exact() -> None:
    source = ACCEPTANCE.read_text(encoding="utf-8")
    assert "model_inference_executed" in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "4_878_257" in source
    assert "4_891_057" in source
    assert "torch.load" not in source
    assert "make_pcqm_gap_encoder" not in source


def test_geometry_multiseed_protocol_requires_all_seed_directions() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "seeds 42, 43, and 44",
        "arithmetic mean",
        "Only one Kaggle GPU kernel",
        "It does not\n   authorize full-data training",
    ):
        assert required in text


def test_geometry_multiseed_pairing_requires_every_seed_to_improve() -> None:
    spec = importlib.util.spec_from_file_location("geometry_confirmation", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passing = []
    for seed in (43, 44):
        passing.extend(
            [
                {
                    "seed": seed,
                    "candidate": module.COMPARATOR,
                    "validation_gap_mae_eV": 0.14,
                },
                {
                    "seed": seed,
                    "candidate": module.CANDIDATE,
                    "validation_gap_mae_eV": 0.13,
                },
            ]
        )
    pairs, passed = module.paired_summary(passing)
    assert passed is True
    assert [row["seed"] for row in pairs] == [42, 43, 44, "mean"]

    failing = [dict(row) for row in passing]
    failing[-1]["validation_gap_mae_eV"] = 0.15
    _, passed = module.paired_summary(failing)
    assert passed is False

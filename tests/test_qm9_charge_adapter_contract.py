from pathlib import Path

import pytest
import torch
from rdkit import Chem

from molgap.qm9_charge_adapter import (
    ADAPTER_RANK,
    BATCH_SIZE,
    CHARGE_CHANNELS,
    MIN_GAIN_EV,
    _gasteiger_features,
    _shared_initialization_exact,
    make_charge_encoder,
    make_control_encoder,
)


ROOT = Path(__file__).parents[1]


def test_frozen_screen_contract():
    assert BATCH_SIZE == 128
    assert CHARGE_CHANNELS == 2
    assert ADAPTER_RANK == 16
    assert MIN_GAIN_EV == pytest.approx(0.001)


def test_gasteiger_features_are_finite_and_atom_aligned():
    molecule = Chem.MolFromSmiles("CC(=O)N")
    values = _gasteiger_features(molecule)
    assert values.shape == (molecule.GetNumAtoms(), 2)
    assert torch.isfinite(values).all()
    total = values.sum().item()
    assert total == pytest.approx(0.0, abs=1e-5)


def test_zero_start_candidate_matches_control_prediction():
    torch.manual_seed(42)
    control = make_control_encoder().eval()
    torch.manual_seed(42)
    candidate = make_charge_encoder([0.0, 0.0], [1.0, 1.0]).eval()
    assert _shared_initialization_exact(control, candidate)

    x = torch.zeros((4, 9), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    edge_attr = torch.zeros((6, 3), dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    rwse = torch.zeros((4, 16))
    charge = torch.tensor([[0.1, 0.0], [-0.2, 0.1], [0.2, 0.0], [-0.1, 0.0]])
    with torch.no_grad():
        control_prediction = control(x, edge_index, edge_attr, batch, rwse)
        candidate_prediction = candidate(
            x, edge_index, edge_attr, batch, rwse, charge
        )
    assert torch.equal(control_prediction, candidate_prediction)
    normalized = (charge - candidate.charge_mean) / candidate.charge_std
    assert torch.count_nonzero(candidate.charge_adapter(normalized)) == 0


def test_preflight_uses_bounded_numeric_equivalence():
    source = (ROOT / "src/molgap/qm9_charge_adapter.py").read_text()
    assert "adapter_output_exact_zero" in source
    assert "zero_start_prediction_max_abs_diff" in source
    assert "atol=1e-6" in source
    assert "torch.equal(\n            forward_encoder" not in source


def test_remote_package_lists_complete_runtime():
    package = (
        ROOT / "platforms/scnet/qm9_charge_adapter/package_runtime.ps1"
    ).read_text(encoding="utf-8")
    for required in (
        "src/molgap",
        "src/molgap/constants.py",
        "src/molgap/qm9_data.py",
        "src/molgap/qm9_local_hierarchy.py",
        "src/molgap/qm9_charge_adapter.py",
    ):
        assert required in package


def test_slurm_chain_keeps_cpu_and_dcu_roles_separate():
    directory = ROOT / "platforms/scnet/qm9_charge_adapter"
    cache = (directory / "build_cache_kunshan.slurm").read_text()
    preflight = (directory / "preflight_kunshan.slurm").read_text()
    train = (directory / "train_kunshan.slurm").read_text()
    assert "#SBATCH --partition=kshctest02" in cache
    assert "--gres=" not in cache
    assert "#SBATCH --partition=kshdtest" in preflight
    assert "#SBATCH --gres=dcu:Hygon:1" in preflight
    assert "#SBATCH --partition=kshdtest" in train
    assert "#SBATCH --gres=dcu:Hygon:1" in train
    assert "last_checkpoint.pt" in (
        ROOT / "src/molgap/qm9_charge_adapter.py"
    ).read_text()

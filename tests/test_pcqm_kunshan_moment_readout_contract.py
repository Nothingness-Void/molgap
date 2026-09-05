"""Static K2 contract tests; no model execution."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/molgap/pcqm_moment_readout.py"
RUNNER = ROOT / "src/molgap/pcqm_kunshan_screen.py"
REMOTE = ROOT / "tests/remote_pcqm_moment_readout.py"
SLURM = ROOT / "experiments/pcqm_gap_architecture/kunshan_moment_readout.slurm"
PROTOCOL = ROOT / "experiments/pcqm_gap_architecture/kunshan_moment_readout_protocol.md"
ACCEPT = ROOT / "experiments/pcqm_gap_architecture/accept_kunshan_moment_readout.py"


def test_k2_sources_parse_and_freeze_one_readout_mechanism() -> None:
    for path in (MODEL, RUNNER, REMOTE, ACCEPT):
        ast.parse(path.read_text(encoding="utf-8"))
    source = MODEL.read_text(encoding="utf-8")
    assert "MOMENT_CHANNELS = 32" in source
    assert "MOMENT_PARAMETER_COUNT = BASELINE_PARAMETER_COUNT + MOMENT_PARAMETER_DELTA" in source
    assert "global_mean_pool(projected, batch)" in source
    assert "raw_second - first.square()" in source
    assert "nn.init.zeros_(self.moment_return.weight)" in source
    assert "MultiheadAttention" not in source


def test_k2_runner_and_remote_gate_are_frozen() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    remote = REMOTE.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert 'choices=("vector", "moment_readout")' in runner
    assert 'format_name = "molgap-kunshan-moment-readout-screen-v1"' in runner
    assert "3_684_753" in remote
    assert "shared initialization changed" in remote
    assert "zero-return readout changed initial prediction" in remote
    assert "--screen moment_readout" in slurm
    assert "#SBATCH --gres=dcu:Hygon:1" in slurm
    assert "#SBATCH --time=12:00:00" in slurm
    assert "Official validation and test-dev stay unread" in protocol
    assert "at least `0.001 eV`" in protocol

"""No model construction: panel identity and execution boundaries only."""
import ast
from pathlib import Path

from molgap.k1_representation_audit import panel_ids


def test_panel_is_stable_unique_and_disjoint_from_training():
    ids = panel_ids()
    assert ids == panel_ids() == sorted(set(ids))
    assert len(ids) == 1024
    assert min(ids) >= 500000 and max(ids) < 550000


def test_no_optimizer_or_training_constructor():
    path = Path("src/molgap/k1_representation_audit.py")
    tree = ast.parse(path.read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not {"backward", "step", "train", "AdamW", "Adam", "SGD"}.intersection(calls)
    assert "autograd.grad" in path.read_text()
    assert 'os.environ.get("SLURM_JOB_ID")' in path.read_text()


def test_source_parses():
    ast.parse(Path("src/molgap/k1_representation_audit.py").read_text())

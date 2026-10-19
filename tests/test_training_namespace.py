"""Relocation compatibility only; Luna runs this and the existing CPU regressions."""
import ast
import importlib
import inspect
import pickle
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("core", "state", "session")


@pytest.mark.parametrize("part", PARTS)
def test_old_and_new_paths_share_the_module_and_all_symbols(part):
    old = importlib.import_module("molgap.family_training_" + part)
    new = importlib.import_module("molgap.training." + part)
    assert old is new
    assert old.__dict__ is new.__dict__
    assert Path(new.__file__).resolve() == ROOT / "src" / "molgap" / "training" / (part + ".py")
    assert all(getattr(old, name) is value for name, value in vars(new).items())


@pytest.mark.parametrize("part", PARTS)
def test_owned_types_and_functions_keep_historical_pickle_names(part):
    old_name = "molgap.family_training_" + part
    module = importlib.import_module("molgap.training." + part)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            value = getattr(module, node.name)
            assert value.__module__ == old_name
            assert pickle.loads(pickle.dumps(value)) is value
            if inspect.isfunction(value):
                assert value.__globals__ is module.__dict__


@pytest.mark.parametrize("part,name,args", [
    ("core", "LoopPlan", (4, 2, 3, 1)),
    ("core", "LoopProgress", (1, 2, 0, 1, "paused")),
    ("core", "BatchPlan", (2,)),
    ("core", "BatchResult", (1, 2, None)),
    ("core", "UpdateObserver", ()),
    ("state", "CalibrationPolicy", ("1" * 64, "2" * 64, "3" * 64, 1)),
])
def test_value_pickles_resolve_original_module(part, name, args):
    module = importlib.import_module("molgap.training." + part)
    value = getattr(module, name)(*args)
    payload = pickle.dumps(value, protocol=4)
    assert ("molgap.family_training_" + part).encode() in payload
    restored = pickle.loads(payload)
    assert type(restored) is type(value)
    assert restored == value


@pytest.mark.parametrize("part,global_name,owner", [
    ("core", "train_batches", "run_training_loop"),
    ("state", "_digest", "CalibrationPolicy.__post_init__"),
    ("session", "run_training_loop", "FamilyTrainingSession.run"),
])
def test_old_path_monkeypatch_reaches_implementation_globals(monkeypatch, part, global_name, owner):
    old = importlib.import_module("molgap.family_training_" + part)
    new = importlib.import_module("molgap.training." + part)
    function = new
    for component in owner.split("."):
        function = getattr(function, component)
    marker = object()
    monkeypatch.setattr(old, global_name, marker)
    assert getattr(new, global_name) is marker
    assert function.__globals__[global_name] is marker


@pytest.mark.parametrize("part", PARTS)
def test_old_module_contains_only_a_compatibility_alias(part):
    tree = ast.parse((ROOT / "src" / "molgap" / ("family_training_" + part + ".py")).read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   for node in ast.walk(tree))
    assert len(tree.body) == 4
    assignment = tree.body[-1]
    assert isinstance(assignment, ast.Assign)
    assert ast.unparse(assignment.targets[0]) == "_sys.modules[__name__]"


@pytest.mark.parametrize("first", ("old", "new"))
def test_both_import_orders_in_a_fresh_interpreter(first, tmp_path):
    program = textwrap.dedent("""\
        import importlib, pickle, sys
        sys.path.insert(0, sys.argv[1])
        for part in ('core', 'state', 'session'):
            names = ['molgap.family_training_' + part, 'molgap.training.' + part]
            if sys.argv[2] == 'new':
                names.reverse()
            a, b = [importlib.import_module(name) for name in names]
            assert a is b
        cls = importlib.import_module('molgap.training.core').LoopPlan
        assert pickle.loads(pickle.dumps(cls(4, 2, 3, 1))).__class__ is cls
        assert 'molgap.experiment_cli' not in sys.modules
    """)
    subprocess.run([sys.executable, "-B", "-c", program, str(ROOT / "src"), first],
                   cwd=tmp_path, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("name", ("molgap.training", "molgap.training.core", "molgap.training.state"))
def test_lightweight_imports_do_not_load_training_siblings_or_cli(name, tmp_path):
    program = textwrap.dedent("""\
        import importlib, sys
        sys.path.insert(0, sys.argv[1])
        importlib.import_module(sys.argv[2])
        for forbidden in ('torch', 'torch_geometric', 'molgap.experiment_cli',
                          'molgap.family_training_kernel', 'molgap.family_training_session',
                          'molgap.training.session'):
            assert forbidden not in sys.modules, forbidden
        if sys.argv[2] == 'molgap.training':
            assert 'molgap.training.core' not in sys.modules
            assert 'molgap.training.state' not in sys.modules
    """)
    subprocess.run([sys.executable, "-B", "-c", program, str(ROOT / "src"), name],
                   cwd=tmp_path, check=True, capture_output=True, text=True)

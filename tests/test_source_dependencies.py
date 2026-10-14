"""No code execution or host-package copying during source dependency discovery."""
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from molgap.source_dependencies import discover_source_dependencies
from test_experiment_spec import payload


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_function_local_and_literal_dynamic_imports_are_packed(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/engine.py", """import torch
def run():
    from .addons import attention
    return __import__('molgap.state', fromlist=['resume']).apply()
""")
    _write(tmp_path, "src/molgap/addons/__init__.py", "")
    _write(tmp_path, "src/molgap/addons/attention.py", "from ..helper import Scale\n")
    _write(tmp_path, "src/molgap/helper.py", "class Scale: pass\n")
    _write(tmp_path, "src/molgap/state/__init__.py", "")
    _write(tmp_path, "src/molgap/state/resume.py", "import random\n")
    result = discover_source_dependencies(tmp_path, ["src/molgap/engine.py"])
    assert result.source_files == tuple(sorted((
        "src/molgap/__init__.py", "src/molgap/engine.py", "src/molgap/helper.py",
        "src/molgap/addons/__init__.py", "src/molgap/addons/attention.py",
        "src/molgap/state/__init__.py", "src/molgap/state/resume.py")))
    assert result.external_modules == ("random", "torch")
    assert result.dynamic_imports == ()


def test_import_cycles_and_order_are_deterministic(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/one.py", "from . import two\n")
    _write(tmp_path, "src/molgap/two.py", "from . import one\n")
    first = discover_source_dependencies(tmp_path, ["src/molgap/one.py", "src/molgap/two.py"])
    second = discover_source_dependencies(tmp_path, ["src/molgap/two.py", "src/molgap/one.py"])
    assert first == second


def test_standalone_root_includes_parent_package_startup(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "from .startup import VALUE\n")
    _write(tmp_path, "src/molgap/startup.py", "VALUE = 1\n")
    _write(tmp_path, "src/molgap/leaf.py", "VALUE = 2\n")
    result = discover_source_dependencies(tmp_path, ["src/molgap/leaf.py"], require_static=True)
    assert result.source_files == ("src/molgap/__init__.py", "src/molgap/leaf.py", "src/molgap/startup.py")


def test_literal_relative_importlib_and_builtin_fromlist(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", """from importlib import import_module as load
def run():
    load('.leaf', 'molgap.parts')
    __import__('molgap.parts', {}, {}, ['other'])
    __import__('helper', globals(), locals(), (), 1)
""")
    _write(tmp_path, "src/molgap/parts/__init__.py", "")
    _write(tmp_path, "src/molgap/parts/leaf.py", "")
    _write(tmp_path, "src/molgap/parts/other.py", "")
    _write(tmp_path, "src/molgap/helper.py", "")
    report = discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"], require_static=True)
    assert {"src/molgap/parts/leaf.py", "src/molgap/parts/other.py", "src/molgap/helper.py"}.issubset(report.source_files)


def test_unresolved_relative_package_is_reported(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", "import importlib\ndef run(package): return importlib.import_module('.leaf', package)\n")
    report = discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])
    assert len(report.dynamic_imports) == 1
    with pytest.raises(ValueError, match="Dynamic import"):
        discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"], require_static=True)


def test_package_lazy_export_and_method_imports_are_traced_when_selected(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", """def __getattr__(name):
    from .public import Symbol
    return Symbol
""")
    _write(tmp_path, "src/molgap/plugin.py", "from molgap import Symbol\n")
    _write(tmp_path, "src/molgap/public.py", "class Symbol: pass\n")
    selected = discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])
    assert "src/molgap/public.py" in selected.source_files
    startup = discover_source_dependencies(tmp_path, ["src/molgap/__init__.py"])
    assert "src/molgap/public.py" not in startup.source_files
    _write(tmp_path, "src/molgap/parts/__init__.py", """class Module:
    def forward(self):
        from ..helper import action
        return action()
""")
    _write(tmp_path, "src/molgap/helper.py", "def action(): return None\n")
    methods = discover_source_dependencies(tmp_path, ["src/molgap/parts/__init__.py"])
    assert "src/molgap/helper.py" in methods.source_files


def test_literal_builtin_fromlist_follows_lazy_symbol(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "def __getattr__(name):\n    from .public import Symbol\n    return Symbol\n")
    _write(tmp_path, "src/molgap/plugin.py", "__import__('molgap', fromlist=['Symbol'])\n")
    _write(tmp_path, "src/molgap/public.py", "class Symbol: pass\n")
    assert "src/molgap/public.py" in discover_source_dependencies(
        tmp_path, ["src/molgap/plugin.py"], require_static=True).source_files


@pytest.mark.parametrize("source", [
    "__import__('molgap', fromlist=names)\n",
    "__import__('molgap', fromlist=[name])\n",
    "__import__('molgap', fromlist=['*'])\n",
    "from molgap import *\n",
])
def test_dynamic_or_wildcard_fromlist_is_a_review_obligation(tmp_path, source):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", source)
    report = discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])
    assert len(report.dynamic_imports) == 1
    with pytest.raises(ValueError, match="Dynamic import"):
        discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"], require_static=True)


def test_dynamic_selection_reported_without_execution(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", """import importlib
def run(module):
    return importlib.import_module(module)
raise RuntimeError('discovery must not execute this file')
""")
    report = discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])
    assert len(report.dynamic_imports) == 1
    with pytest.raises(ValueError, match="Dynamic import"):
        discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"], require_static=True)


def test_missing_internal_dependency_fails(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", "from .missing import thing\n")
    with pytest.raises(ValueError, match="Missing internal dependency"):
        discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])


def test_relative_import_escape_fails(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", "from ...credentials import token\n")
    with pytest.raises(ValueError, match="escapes"):
        discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])


def test_source_secret_and_link_boundaries_are_reused(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", "from .secrets import token\n")
    _write(tmp_path, "src/molgap/secrets.py", "token = 'fixture'\n")
    with pytest.raises(ValueError):
        discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])


def test_python_io_owner_is_source_while_data_payload_and_secret_names_remain_blocked(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/plugin.py", "from .my_data import load\n")
    _write(tmp_path, "src/molgap/my_data.py", "def load(): return None\n")
    report = discover_source_dependencies(tmp_path, ["src/molgap/plugin.py"])
    assert "src/molgap/my_data.py" in report.source_files
    for path in ("src/molgap/data/payload.py", "src/molgap/my_data/payload.py",
                 "src/molgap/checkpoint_results/worker.py", "data/graphs.pt", "src/molgap/private_token.py"):
        _write(tmp_path, path, "FIXTURE = 0\n")
        with pytest.raises(ValueError):
            discover_source_dependencies(tmp_path, [path])


def test_actual_foreach_worker_import_closure_contains_repaired_primitive():
    # AST-only and bounded: no Torch import, cache, or remote execution.
    root = Path(__file__).resolve().parents[1]
    report = discover_source_dependencies(root, ["src/molgap/pcqm_gptrans_v4.py"])
    assert "src/molgap/gptrans_execution_hotpath.py" in report.source_files
    assert "src/molgap/gptrans_execution_tuning.py" in report.source_files


def test_discovered_closure_publishes_and_imports_in_isolation(tmp_path, payload):
    from molgap.experiment_package import build_experiment_source_package, inspect_experiment_source_package
    from molgap.experiment_spec import ExperimentSpec
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "src/molgap/__init__.py", "from .startup import BASE\n")
    _write(root, "src/molgap/startup.py", "BASE = 2\n")
    _write(root, "src/molgap/addon.py", "def compute():\n    from .helper import VALUE\n    from molgap import BASE\n    return BASE + VALUE\n")
    # Python prefers helper/__init__.py to helper.py; freezing must preserve it.
    _write(root, "src/molgap/helper/__init__.py", "from .leaf import VALUE\n")
    _write(root, "src/molgap/helper/leaf.py", "VALUE = 5\n")
    _write(root, "src/molgap/helper.py", "raise RuntimeError('wrong module resolution')\n")
    for arguments in (("init",), ("config", "user.name", "Synthetic fixture"),
                      ("config", "user.email", "synthetic@example.invalid"),
                      ("config", "core.autocrlf", "false"), ("add", "."),
                      ("-c", "commit.gpgsign=false", "commit", "-m", "Synthetic source")):
        subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)
    closure = discover_source_dependencies(root, ["src/molgap/addon.py"], require_static=True)
    assert "src/molgap/helper/__init__.py" in closure.source_files
    assert "src/molgap/helper.py" not in closure.source_files
    target = tmp_path / "package"
    manifest = build_experiment_source_package(ExperimentSpec(payload), root, closure.source_files, target)
    assert inspect_experiment_source_package(target, root) == manifest
    extracted = tmp_path / "extracted"
    with tarfile.open(target / "source.tar.gz") as archive:
        assert tuple(archive.getnames()) == closure.source_files
        for member in archive.getmembers():
            assert member.isfile() and member.name in closure.source_files
            output = extracted / member.name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.extractfile(member).read())
    result = subprocess.run([sys.executable, "-I", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); from molgap.addon import compute; print(compute())",
        str(extracted / "src")], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "7"

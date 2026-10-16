"""A/B migration checks use generated files and independent child interpreters."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from molgap.local_runtime import launch_local

ROOT = Path(__file__).resolve().parents[1]


def test_root_conftest_imports_no_cli_or_torch():
    script = ("import runpy,sys; runpy.run_path(sys.argv[1]); "
              "assert not any(n.startswith('cli.') for n in sys.modules); "
              "assert 'torch' not in sys.modules; print('LAZY_COLLECTION_OK')")
    result = subprocess.run([sys.executable,"-I","-c",script,str(ROOT/"tests/conftest.py")],
                            capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LAZY_COLLECTION_OK"


def test_default_collection_ignores_snapshot_trees(tmp_path):
    (tmp_path/"pyproject.toml").write_bytes((ROOT/"pyproject.toml").read_bytes())
    (tmp_path/"tests").mkdir();(tmp_path/"tests/test_probe.py").write_text("def test_probe(): pass\n")
    (tmp_path/"snapshot").mkdir();(tmp_path/"snapshot/test_must_not_collect.py").write_text("raise RuntimeError('snapshot collected')\n")
    result = subprocess.run([sys.executable,"-I","-m","pytest","--collect-only","-q"],
                            cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "1 test collected" in result.stdout and "snapshot collected" not in result.stdout


def test_two_workspaces_select_their_own_sources_not_inherited_editable(tmp_path,monkeypatch,capfd):
    roots=[]
    for name in ("a","b"):
        root=tmp_path/name;package=root/"src/molgap";package.mkdir(parents=True)
        (package/"__init__.py").write_text(f"MARKER={name!r}\n")
        (package/"probe.py").write_text("from molgap import MARKER\nprint(MARKER)\n")
        roots.append(root)
    monkeypatch.setenv("PYTHONPATH",str(roots[1]/"src"))
    for name,root in zip(("a","b"),roots):
        assert launch_local(root,"molgap.probe")==0
        output,logs=capfd.readouterr();assert output.strip()==name
        info=json.loads(logs.strip());assert Path(info["molgap_module"])==root/"src/molgap/__init__.py"
        assert info["package_identity"] is None
    assert "PYTHONPATH" in __import__('os').environ

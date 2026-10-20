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


@pytest.mark.parametrize("exit_code", [0, 7])
def test_child_scratch_is_cleaned_after_success_or_failure(tmp_path,capfd,exit_code):
    import os
    root=tmp_path/"checkout";package=root/"src/molgap";package.mkdir(parents=True)
    (package/"__init__.py").write_text("")
    (package/"probe.py").write_text(
        "import os, sys, tempfile\nfrom pathlib import Path\n"
        "scratch = Path(tempfile.gettempdir())\n"
        "assert all(Path(os.environ[key]) == scratch for key in ('TMP', 'TEMP', 'TMPDIR'))\n"
        "nested = Path(tempfile.mkdtemp())\n(nested / 'payload').write_bytes(b'temporary')\n"
        "print(scratch)\nsys.exit(int(sys.argv[1]))\n")
    before={key:os.environ.get(key) for key in ("TMP","TEMP","TMPDIR")}
    scratch=tmp_path/"external-scratch"
    retained=scratch/"existing-run"/"retained.bin"
    retained.parent.mkdir(parents=True);retained.write_bytes(b"existing artifact")
    assert launch_local(root,"molgap.probe",[exit_code],scratch_root=scratch)==exit_code
    output,_=capfd.readouterr()
    child_scratch=Path(output.strip())
    assert child_scratch.parent==scratch and not child_scratch.exists()
    assert retained.read_bytes()==b"existing artifact"
    assert list(scratch.iterdir())==[retained.parent]
    assert before=={key:os.environ.get(key) for key in before}


def test_child_scratch_is_cleaned_if_process_cannot_start(tmp_path,monkeypatch):
    root=tmp_path/"checkout";package=root/"src/molgap";package.mkdir(parents=True)
    (package/"__init__.py").write_text("")
    scratch=tmp_path/"external-scratch"
    seen=[]

    def fail_to_start(*args,env,**kwargs):
        child_scratch=Path(env["TMP"])
        assert child_scratch.is_dir()
        (child_scratch/"payload").write_bytes(b"temporary")
        seen.append(child_scratch)
        raise OSError("synthetic process start failure")

    monkeypatch.setattr(subprocess,"run",fail_to_start)
    with pytest.raises(OSError,match="synthetic process start failure"):
        launch_local(root,"molgap.probe",scratch_root=scratch)
    assert len(seen)==1 and not seen[0].exists()
    assert scratch.is_dir() and not list(scratch.iterdir())


def test_concurrent_launches_have_independent_scratch(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    root=tmp_path/"checkout";package=root/"src/molgap";package.mkdir(parents=True)
    (package/"__init__.py").write_text("")
    (package/"probe.py").write_text(
        "import sys, tempfile\nfrom pathlib import Path\n"
        "scratch = Path(tempfile.gettempdir())\n"
        "(scratch / 'payload').write_bytes(b'temporary')\n"
        "Path(sys.argv[1]).write_text(str(scratch))\n")
    scratch=tmp_path/"external-scratch"
    barrier=Barrier(2)
    run=subprocess.run

    def overlap_launches(*args,env,**kwargs):
        child_scratch=Path(env["TMP"])
        (child_scratch/"ready").write_bytes(b"temporary")
        barrier.wait(timeout=30)
        assert len(list(scratch.iterdir()))==2
        barrier.wait(timeout=30)
        return run(*args,env=env,timeout=30,**kwargs)

    monkeypatch.setattr(subprocess,"run",overlap_launches)
    receipts=[tmp_path/f"child-{index}.txt" for index in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        launches=[pool.submit(launch_local,root,"molgap.probe",[receipt],scratch_root=scratch)
                  for receipt in receipts]
        assert [launch.result(timeout=30) for launch in launches]==[0,0]
    child_scratch=[Path(receipt.read_text()) for receipt in receipts]
    assert len(set(child_scratch))==2
    assert all(path.parent==scratch and not path.exists() for path in child_scratch)
    assert scratch.is_dir() and not list(scratch.iterdir())


def test_platform_script_uses_selected_checkout_and_cannot_escape(tmp_path,capfd):
    root=tmp_path/"checkout";package=root/"src/molgap";package.mkdir(parents=True)
    (package/"__init__.py").write_text("MARKER='selected-platform-checkout'\n")
    entry=root/"platforms/local/probe.py";entry.parent.mkdir(parents=True)
    entry.write_text("from molgap import MARKER\nprint(MARKER)\n")
    assert launch_local(root,None,script="platforms/local/probe.py")==0
    assert capfd.readouterr().out.strip()=="selected-platform-checkout"
    for script in ("../escape.py","src/molgap/__init__.py","platforms/local/probe.py/.."):
        with pytest.raises(ValueError):launch_local(root,None,script=script)
    with pytest.raises(ValueError):launch_local(root,"molgap.probe",script="platforms/local/probe.py")

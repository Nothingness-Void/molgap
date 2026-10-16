"""B: explicit machine-local storage and no-clobber streaming migration."""
import hashlib
import json
from pathlib import Path
import sys
import pytest
from molgap.common.io import copy_immutable_file
from molgap.common.paths import MachinePaths

def profile(tmp_path):
    return MachinePaths(python=sys.executable,data_root=tmp_path/"data",
                        artifact_root=tmp_path/"artifacts",scratch_root=tmp_path/"scratch")


def test_machine_paths_bind_run_attempt_without_worktree_storage(tmp_path):
    paths=profile(tmp_path).outside_checkout(tmp_path/"code")
    assert paths.run_directory("run-42","attempt-1")==tmp_path/"artifacts/runs/run-42/attempt-1"
    with pytest.raises(ValueError):paths.run_directory("../code","attempt-1")
    with pytest.raises(ValueError):paths.outside_checkout(tmp_path/"artifacts/code")
    raw={"format":"molgap-machine-paths-v1",**{k:str(getattr(paths,k)) for k in
        ("python","data_root","artifact_root","scratch_root")}}
    f=tmp_path/"profile.json";f.write_text(json.dumps(raw,indent=2)+"\n")
    assert MachinePaths.from_file(f).run_directory("run-42","attempt-1")==paths.run_directory("run-42","attempt-1")
    f.write_text('{"format":"x","format":"y"}')
    with pytest.raises(ValueError,match="Duplicate"):MachinePaths.from_file(f)


def test_copy_then_verify_keeps_original_and_is_idempotent(tmp_path):
    source=tmp_path/"code/result.bin";source.parent.mkdir()
    source.write_bytes(b'bounded-fixture'*100000)
    sha=hashlib.sha256(source.read_bytes()).hexdigest();target=tmp_path/"store/result.bin"
    first=copy_immutable_file(source,target,expected_sha256=sha)
    assert first==copy_immutable_file(source,target,expected_sha256=sha)
    assert first["original_retained"] and source.read_bytes()==target.read_bytes()
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):copy_immutable_file(source,target,expected_sha256=sha)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==sha


def test_conflicting_destination_or_reparse_is_not_overwritten(tmp_path,monkeypatch):
    source=tmp_path/"source";source.write_bytes(b'original')
    target=tmp_path/"target";target.write_bytes(b'other')
    with pytest.raises(ValueError,match="conflict"):
        copy_immutable_file(source,target,expected_sha256=hashlib.sha256(b'original').hexdigest())
    assert target.read_bytes()==b'other' and source.read_bytes()==b'original'
    from types import SimpleNamespace
    import stat
    original=Path.lstat
    def fake(path,*a,**k):
        if path.name=="junction":return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
        return original(path,*a,**k)
    monkeypatch.setattr(Path,"lstat",fake)
    with pytest.raises(ValueError,match="reparse"):
        copy_immutable_file(source,tmp_path/"junction/copied",expected_sha256=hashlib.sha256(b'original').hexdigest())

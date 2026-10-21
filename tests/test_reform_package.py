"""Fixed source boundary, batch Git validation and safe archive-only reuse."""
from copy import deepcopy
from dataclasses import replace
import builtins
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from unittest.mock import patch

import pytest

from molgap.experiment_package import build_experiment_source_package,verify_experiment_source_package,_name
from molgap.experiment_source_inventory import runtime_source_files
from molgap.experiment_spec import ExperimentSpec
from molgap.local_storage import SOURCE_CACHE_NAMESPACE
from molgap import experiment_spec as spec_module
from molgap.screen_policy import canonical_fingerprint
from test_experiment_package import repo,payload,git
from test_local_storage import cache_holder


def test_python_data_namespace_is_source_but_payloads_and_credentials_are_not():
    assert _name("src/molgap/data/loader.py")=="src/molgap/data/loader.py"
    assert _name("src/molgap/k1_pair_token_mose.py").endswith(".py")
    for value in ("data/loader.py","src/molgap/data/secret.py","src/molgap/data/cache.pt",
                  "src/molgap/keys/helper.py","src/molgap/auth_token.py","src/molgap/.env/config.py"):
        with pytest.raises(ValueError):_name(value)


def add_runtime(root):
    for name in ("__init__.py","model.py","new_addon.py","data/__init__.py","data/loader.py",
                 "server_control.py","common/kaggle.py", "artifact_inventory.py",
                 "experiment_run.py", "experiment_timings.py"):
        path=root/"src/molgap"/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("VALUE=42\n")
    private=root/"platforms/_records/credentials.json";private.parent.mkdir(parents=True);private.write_text('{}')
    git(root,"add",".");git(root,"-c","commit.gpgsign=false","commit","-m","Fixture runtime")


def test_new_tracked_addon_enters_fixed_distribution_without_ast_or_manual_list(repo):
    add_runtime(repo)
    with patch("molgap.source_dependencies.discover_source_dependencies",side_effect=AssertionError("AST in default path")):
        names=runtime_source_files(repo)
    assert "src/molgap/new_addon.py" in names and "src/molgap/data/loader.py" in names
    assert "src/molgap/server_control.py" not in names
    assert "src/molgap/common/kaggle.py" not in names
    assert not {"src/molgap/"+name+".py" for name in
                ("artifact_inventory","experiment_run","experiment_timings")} & set(names)
    assert not any("_records" in p for p in names)


def test_cache_reuses_archive_but_never_other_spec_package_or_receipt(repo,payload,tmp_path):
    cache=tmp_path/"archive-cache";first=tmp_path/"first";second=tmp_path/"second"
    a=build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],first,source_cache=cache)
    other=deepcopy(payload);other["logical_run_id"]="independent-run"
    with patch("molgap.experiment_package.build_v4_source_bundle",side_effect=AssertionError("warm archive rebuilt")):
        b=build_experiment_source_package(ExperimentSpec(other),repo,["README.md"],second,source_cache=cache)
    assert a["archive_sha256"]==b["archive_sha256"]
    assert a["package_identity"]!=b["package_identity"] and a["spec_identity"]!=b["spec_identity"]
    assert len(list((cache/SOURCE_CACHE_NAMESPACE).glob("[0-9a-f]"*64)))==1
    assert verify_experiment_source_package(second,repo)==b
    assert not (second/"launch_receipt.json").exists()


def test_uncached_package_never_imports_local_storage(repo, payload, tmp_path, monkeypatch):
    original = builtins.__import__
    def import_without_cache(name, *args, **kwargs):
        if name == "local_storage" or name == "molgap.local_storage":
            raise ImportError("legacy source bundle excludes host cache management")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", import_without_cache)
    output = tmp_path / "uncached"
    manifest = build_experiment_source_package(ExperimentSpec(payload), repo, ["README.md"], output)
    assert verify_experiment_source_package(output, repo) == manifest


def test_busy_cache_builds_independent_package_without_waiting(repo, payload, tmp_path, cache_holder):
    cache, process, connection = cache_holder
    output = tmp_path / "busy-cache-package"
    manifest = build_experiment_source_package(
        ExperimentSpec(payload), repo, ["README.md"], output, source_cache=cache)
    assert process.is_alive()  # Holder remains blocked until fixture teardown.
    assert verify_experiment_source_package(output, repo) == manifest
    assert {p.name for p in (cache / SOURCE_CACHE_NAMESPACE).iterdir()} == {".lock"}


def test_normal_packaging_automatically_bounds_cache_to_eight(repo, payload, tmp_path, monkeypatch):
    monkeypatch.delenv("MOLGAP_SOURCE_CACHE_MAX_ENTRIES", raising=False)
    cache = tmp_path / "bounded-cache"
    legacy = cache / ("f" * 64)
    legacy.mkdir(parents=True)
    (legacy / "source.tar.gz").write_bytes(b"legacy builder cache")
    packages = []
    for index in range(10):
        (repo / "README.md").write_text(f"Synthetic source variant {index}\n")
        git(repo, "add", "README.md")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-m", f"Variant {index}")
        output = tmp_path / f"package-{index}"
        manifest = build_experiment_source_package(
            ExperimentSpec(payload), repo, ["README.md"], output, source_cache=cache)
        packages.append((output, manifest))
    entries = list((cache / SOURCE_CACHE_NAMESPACE).glob("[0-9a-f]" * 64))
    assert len(entries) == 8
    assert (legacy / "source.tar.gz").read_bytes() == b"legacy builder cache"
    # Eviction does not invalidate independent packages retained by their runs.
    for output, manifest in packages:
        assert verify_experiment_source_package(output) == manifest


def test_git_verification_process_count_is_constant_not_per_file(repo,payload,tmp_path):
    for i in range(80):(repo/f"module_{i}.py").write_text(f"VALUE={i}\n")
    git(repo,"add",".");git(repo,"-c","commit.gpgsign=false","commit","-m","Many source modules")
    package=tmp_path/"many"
    build_experiment_source_package(ExperimentSpec(payload),repo,[f"module_{i}.py" for i in range(80)],package)
    calls=[];run=subprocess.run
    def observe(command,*a,**kw):
        calls.append(command);return run(command,*a,**kw)
    with patch("subprocess.run",observe):verify_experiment_source_package(package,repo)
    assert len(calls)==3,calls
    assert any("--batch" in c for c in calls)


@pytest.mark.parametrize("generated", [False, True])
def test_verification_reads_and_decompresses_one_archive_snapshot(repo,payload,tmp_path,generated):
    spec=ExperimentSpec(payload);names=["README.md"];options={}
    if generated:
        config_name="scientific.json"
        config={"format":"molgap-scientific-experiment-v1","experiment_id":payload["experiment_id"]}
        config_bytes=json.dumps(config,sort_keys=True,separators=(",", ":")).encode()
        (repo/config_name).write_bytes(config_bytes)
        git(repo,"add",config_name)
        git(repo,"-c","commit.gpgsign=false","commit","-m","Synthetic configuration provenance")
        digest=hashlib.sha256(config_bytes).hexdigest();commit=git(repo,"rev-parse","HEAD")
        namespace=canonical_fingerprint({"input":digest,"commit":commit})
        derived=f"experiments/{payload['experiment_id']}/generated/{namespace}/recipe.json"
        path=repo/derived;path.parent.mkdir(parents=True);path.write_bytes(b'{}')
        names.extend([config_name,derived])
        options={"generated_records":{derived:b'{}'},"configuration_source":{
            "path":config_name,"sha256":digest,"compiler":config["format"]}}
    package=tmp_path/"package"
    manifest=build_experiment_source_package(spec,repo,names,package,**options)
    archive=package/"source.tar.gz";reads=[];path_open=Path.open;file_open=builtins.open
    def observe(path,*args,**kwargs):
        if path==archive:reads.append((args,kwargs))
        return path_open(path,*args,**kwargs)
    def observe_file(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path)==archive:reads.append((args,kwargs))
        return file_open(path,*args,**kwargs)
    with patch.object(Path,"open",observe), patch("builtins.open",observe_file), patch(
            "tarfile.bltn_open",observe_file), patch(
            "gzip._GzipReader",wraps=gzip._GzipReader) as gzip_reader, patch(
            "molgap.experiment_package.tarfile.open",wraps=tarfile.open) as tar_open:
        assert verify_experiment_source_package(package,repo)==manifest
    assert len(reads)==1,reads
    assert tar_open.call_count==1
    assert gzip_reader.call_count==1


def test_clean_filter_or_index_flag_cannot_hide_changed_source_on_cache_hit(repo,payload,tmp_path):
    cache=tmp_path/"cache"
    build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"before",source_cache=cache)
    git(repo,"update-index","--assume-unchanged","README.md")
    (repo/"README.md").write_text("changed actual bytes\n")
    with pytest.raises(ValueError,match="payload mismatch"):
        build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"after",source_cache=cache)


def test_cache_corruption_never_publishes_complete_package(repo,payload,tmp_path):
    cache=tmp_path/"cache"
    build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"before",source_cache=cache)
    next((cache/SOURCE_CACHE_NAMESPACE).glob("*/source.tar.gz")).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"after",source_cache=cache)
    assert not (tmp_path/"after/package_manifest.json").exists()


def unrelated_commit(root):
    (root/"unrelated.md").write_bytes(b"Unrelated documentation only\n")
    git(root,"add","unrelated.md")
    git(root,"-c","commit.gpgsign=false","commit","-m","Unrelated documentation")
    return git(root,"rev-parse","HEAD")


def test_identical_full_payload_reuses_archive_across_commits_with_fresh_sidecars(repo,payload,tmp_path):
    cache=tmp_path/"cache";first=tmp_path/"first";second=tmp_path/"second"
    names=["README.md","src/example.py"];spec=ExperimentSpec(payload)
    a=build_experiment_source_package(spec,repo,names,first,source_cache=cache)
    cached=next((cache/SOURCE_CACHE_NAMESPACE).glob("[0-9a-f]"*64))
    retained={p.name:p.read_bytes() for p in cached.iterdir()}
    commit=unrelated_commit(repo)
    with patch("molgap.experiment_package.build_v4_source_bundle",side_effect=AssertionError("archive rebuilt")):
        b=build_experiment_source_package(spec,repo,names,second,source_cache=cache)
    assert a["source_commit"]!=b["source_commit"]==commit
    assert a["archive_sha256"]==b["archive_sha256"]
    assert a["package_identity"]!=b["package_identity"]
    assert a["inventory_sha256"]!=b["inventory_sha256"]
    assert a["spec_identity"]==b["spec_identity"]
    assert (second/"SOURCE_COMMIT.txt").read_bytes()==(commit+"\n").encode()
    assert json.loads((second/"SOURCE_FILES.json").read_bytes())["source_commit"]==commit
    assert (first/"source.tar.gz").read_bytes()==(second/"source.tar.gz").read_bytes()
    assert {p.name:p.read_bytes() for p in cached.iterdir()}==retained
    assert len(list((cache/SOURCE_CACHE_NAMESPACE).glob("[0-9a-f]"*64)))==1
    assert verify_experiment_source_package(first,repo)==a
    assert verify_experiment_source_package(second,repo)==b
    # A cold build at the new HEAD must produce exactly the same six files.
    cold=tmp_path/"cold"
    assert build_experiment_source_package(spec,repo,names,cold)==b
    assert {p.name:p.read_bytes() for p in second.iterdir()}=={p.name:p.read_bytes() for p in cold.iterdir()}


def test_seed_only_synthetic_spec_reuses_code_archive_with_distinct_identity(repo,payload,tmp_path,monkeypatch):
    # Test-only parser contract; no seed expansion or scientific run is authorized.
    monkeypatch.setattr(spec_module,"FAMILIES",{
        key:replace(contract,allow_any_seed=True) for key,contract in spec_module.FAMILIES.items()})
    other=deepcopy(payload);other["arms"][0]["initialization"]["seed"]=43
    cache=tmp_path/"cache";names=["src/example.py"]
    a=build_experiment_source_package(ExperimentSpec(payload),repo,names,tmp_path/"first",source_cache=cache)
    with patch("molgap.experiment_package.build_v4_source_bundle",side_effect=AssertionError("archive rebuilt")):
        b=build_experiment_source_package(ExperimentSpec(other),repo,names,tmp_path/"second",source_cache=cache)
    assert a["archive_sha256"]==b["archive_sha256"]
    assert a["spec_identity"]!=b["spec_identity"] and a["package_identity"]!=b["package_identity"]
    assert verify_experiment_source_package(tmp_path/"second",repo)==b
    assert len(list((cache/SOURCE_CACHE_NAMESPACE).glob("[0-9a-f]"*64)))==1


@pytest.mark.parametrize("fault",["dirty","hidden_dirty","source.tar.gz","SOURCE_COMMIT.txt",
                                  "SOURCE_ARCHIVE_SHA256.txt","SOURCE_FILES.json"])
def test_cross_commit_cache_rejects_dirty_payloads_and_corrupt_objects(repo,payload,tmp_path,fault):
    cache=tmp_path/"cache";spec=ExperimentSpec(payload);names=["README.md","src/example.py"]
    build_experiment_source_package(spec,repo,names,tmp_path/"first",source_cache=cache)
    unrelated_commit(repo)
    if fault in {"dirty","hidden_dirty"}:
        if fault=="hidden_dirty":git(repo,"update-index","--assume-unchanged","src/example.py")
        (repo/"src/example.py").write_bytes(b"Changed actual file bytes\n")
    else:
        (next((cache/SOURCE_CACHE_NAMESPACE).glob("[0-9a-f]"*64))/fault).write_bytes(b"corrupt")
    with patch("molgap.experiment_package.build_v4_source_bundle",side_effect=AssertionError("unsafe rebuild")):
        with pytest.raises((ValueError,RuntimeError)):
            build_experiment_source_package(spec,repo,names,tmp_path/"second",source_cache=cache)
    assert not (tmp_path/"second/package_manifest.json").exists()


def test_real_committed_recipe_change_changes_full_archive_at_same_paths(repo,payload,tmp_path):
    recipe=repo/"recipe.json";recipe.write_bytes(b'{"seed":42}')
    git(repo,"add","recipe.json")
    git(repo,"-c","commit.gpgsign=false","commit","-m","Synthetic recipe")
    cache=tmp_path/"cache";spec=ExperimentSpec(payload);names=["src/example.py","recipe.json"]
    a=build_experiment_source_package(spec,repo,names,tmp_path/"first",source_cache=cache)
    recipe.write_bytes(b'{"seed":43}')
    git(repo,"add","recipe.json")
    git(repo,"-c","commit.gpgsign=false","commit","-m","Change synthetic recipe bytes")
    b=build_experiment_source_package(spec,repo,names,tmp_path/"second",source_cache=cache)
    assert a["archive_sha256"]!=b["archive_sha256"] and a["package_identity"]!=b["package_identity"]
    assert len(list((cache/SOURCE_CACHE_NAMESPACE).glob("[0-9a-f]"*64)))==2
    assert verify_experiment_source_package(tmp_path/"second",repo)==b


def test_head_change_during_cache_copy_cannot_publish_stale_commit(repo,payload,tmp_path):
    from molgap.common.io import copy_immutable_file
    cache=tmp_path/"cache";spec=ExperimentSpec(payload)
    build_experiment_source_package(spec,repo,["README.md"],tmp_path/"first",source_cache=cache)
    def change_head(*args,**kwargs):
        unrelated_commit(repo)
        return copy_immutable_file(*args,**kwargs)
    with patch("molgap.common.io.copy_immutable_file",side_effect=change_head):
        with pytest.raises(ValueError,match="HEAD changed"):
            build_experiment_source_package(spec,repo,["README.md"],tmp_path/"second",source_cache=cache)
    assert not (tmp_path/"second/package_manifest.json").exists()

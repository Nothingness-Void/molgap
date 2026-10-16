"""Fixed source boundary, batch Git validation and safe archive-only reuse."""
from copy import deepcopy
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from molgap.experiment_package import build_experiment_source_package,verify_experiment_source_package,_name
from molgap.experiment_source_inventory import runtime_source_files
from molgap.experiment_spec import ExperimentSpec
from test_experiment_package import repo,payload,git


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
    assert len(list(cache.iterdir()))==1
    assert verify_experiment_source_package(second,repo)==b
    assert not (second/"launch_receipt.json").exists()


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
    next(cache.glob("*/source.tar.gz")).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"after",source_cache=cache)
    assert not (tmp_path/"after/package_manifest.json").exists()

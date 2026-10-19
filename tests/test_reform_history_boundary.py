"""Default worker payload retirement with explicit historical source compatibility."""
from pathlib import Path
import tarfile

import pytest

from molgap.experiment_package import (
    PACKAGE_FORMAT,
    build_experiment_source_package,
    inspect_experiment_source_package,
    read_packaged_text,
    verify_experiment_source_package,
)
from molgap.experiment_source_inventory import runtime_source_files
from molgap.experiment_spec import ExperimentSpec
from test_experiment_package import git, payload, repo


ACTIVE = (
    "src/molgap/__init__.py",
    "src/molgap/packed_graph_dataset.py",
    "src/molgap/training/new_optimizer.py",
    "src/molgap/models/new_family.py",
    "src/molgap/data/new_loader.py",
    "src/molgap/compat/legacy_constants.py",
    "src/molgap/phase8_supported.py",
    "src/molgap/v4_runtime.py",
    "src/molgap/archive_tools.py",
    "src/molgap/archive_v2/__init__.py",
    "platforms/kaggle/run_experiment.py",
)
ARCHIVED = (
    "src/molgap/archive/__init__.py",
    "src/molgap/archive/phase8_r01_router/late_blend.py",
    "src/molgap/archive/phase8_r03_three_expert/hetero_moe.py",
    "src/molgap/archive/phase8_r04_static_dual2d/dual2d.py",
    "src/molgap/archive/phase8_r05_physics_consistency/fusion.py",
    "src/molgap/archive/phase8_r06_structural_gps_adapter/structural_features.py",
    "src/molgap/archive/retained/nested_helper.py",
)
HOST_ONLY = (
    "src/molgap/common/kaggle.py",
    "src/molgap/kaggle_credentials.py",
    "src/molgap/common/transport.py",
    "src/molgap/local_runtime.py",
    "src/molgap/server_control.py",
    "src/molgap/kaggle_submission.py",
    "src/molgap/kaggle_output_retrieval.py",
)


@pytest.fixture
def history_repo(repo):
    for name in (*ACTIVE, *ARCHIVED, *HOST_ONLY):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"VALUE = 42\n")
    git(repo, "add", ".")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic runtime and history")
    return repo


def test_default_runtime_excludes_only_explicit_archive_subtree(history_repo):
    names = runtime_source_files(history_repo)
    assert names == sorted(ACTIVE)
    assert not set(ARCHIVED) & set(names)
    assert not set(HOST_ONLY) & set(names)


def test_real_checkout_archive_sources_are_outside_default_distribution():
    from molgap.v4_bundle import _tracked_paths

    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_paths(root)
    archived = {name for name in tracked if name.startswith("src/molgap/archive/")
                and name.endswith(".py")}
    names = set(runtime_source_files(root))
    assert not archived & names
    assert {
        "src/molgap/v4_runtime.py",
    } <= names


def test_new_default_package_physically_omits_archived_sources(history_repo, payload, tmp_path):
    target = tmp_path / "new-package"
    names = runtime_source_files(history_repo)
    manifest = build_experiment_source_package(ExperimentSpec(payload), history_repo, names, target)
    assert verify_experiment_source_package(target, history_repo) == manifest
    with tarfile.open(target / "source.tar.gz", "r:gz") as archive:
        assert set(archive.getnames()) == set(ACTIVE)


def test_explicit_historical_package_still_verifies_and_reads(history_repo, payload, tmp_path):
    target = tmp_path / "historical-package"
    manifest = build_experiment_source_package(
        ExperimentSpec(payload), history_repo, [*ACTIVE, *ARCHIVED], target,
    )
    assert manifest["format"] == PACKAGE_FORMAT == "molgap-experiment-source-package-v1"
    assert not set(ARCHIVED) & set(runtime_source_files(history_repo))
    # Frozen packages are self-contained even when the original checkout is unavailable.
    assert verify_experiment_source_package(target) == manifest
    assert inspect_experiment_source_package(target) == manifest
    assert verify_experiment_source_package(target, history_repo) == manifest
    for name in ARCHIVED:
        assert read_packaged_text(target, name) == "VALUE = 42\n"


def test_explicit_source_selection_can_still_request_history(history_repo):
    names = runtime_source_files(history_repo, records=ARCHIVED)
    assert set(names) == set(ACTIVE) | set(ARCHIVED)


@pytest.mark.parametrize("name", [
    "src/molgap/credentials.py",
    "src/molgap/data/private_key.py",
    "src/molgap/auth_token.py",
    "src/molgap/tokens/helper.py",
    "src/molgap/.env/config.py",
])
def test_runtime_still_rejects_sensitive_tracked_python(history_repo, monkeypatch, name):
    from molgap import v4_bundle

    tracked = v4_bundle._tracked_paths(history_repo) | {name}
    monkeypatch.setattr(v4_bundle, "_tracked_paths", lambda root: tracked)
    with pytest.raises(ValueError, match="reserved or sensitive"):
        runtime_source_files(history_repo)


@pytest.mark.parametrize("name", [
    "src/molgap/archive/../credentials.py",
    "src/molgap/archive/credentials.py",
    "src/molgap/archive/auth_token.py",
    "src/molgap/archive/cache.pt",
])
def test_historical_reader_keeps_path_and_credential_denials(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe source path|reserved or sensitive"):
        read_packaged_text(tmp_path / "absent-package", name)

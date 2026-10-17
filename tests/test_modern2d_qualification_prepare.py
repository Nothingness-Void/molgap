"""Local preparation gates over generated metadata and tiny CPU initial states.

These fixtures preserve typed defaults and shared package/launch validation.
They do not open graph payloads or qualify the full default models on a device.
"""
from copy import deepcopy
from dataclasses import asdict
from importlib import import_module
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
import torch

from molgap import modern2d_qualification_prepare as owner
from molgap.common.wire import canonical_fingerprint
from molgap.data.indexed_inputs import AcceptedIndexedInputPermit
from molgap.experiment_spec import ExperimentSpec, FAMILIES
from molgap.family_training_runner import METADATA, read_recipe, validate_screen_recipe
from molgap.modern2d_input_preparation import METADATA_FORMAT
from molgap.modern2d_qualification_runtime import LAUNCH_FILENAME, validate_launch
from molgap.research_memory.trace import file_digest, json_bytes


ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def _read(path):
    return json.loads(path.read_bytes())


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _snapshot(root):
    return {p.relative_to(root).as_posix(): file_digest(p)
            for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}


def _role_binding(role, acceptance_sha):
    manifest_sha = ("2" if role == "train_100k" else "3") * 64
    authority = dict(scope="ACCEPTED_INDEXED_INPUT_V1", role=role,
        acceptance_sha256=acceptance_sha, manifest_sha256=manifest_sha,
        version_identity="4" * 64, operations=["prediction_input", "labels_read"],
        protected=False, max_chunk_bytes=16 * 1024**2)
    history = dict(role=role, manifest_sha256=manifest_sha,
        usage="training" if role == "train_100k" else "selection-used")
    permit = AcceptedIndexedInputPermit(role, acceptance_sha, manifest_sha, "4" * 64,
        canonical_fingerprint(authority), canonical_fingerprint(history), 16 * 1024**2)
    return dict(permit=asdict(permit), authority=authority, role_history=history)


@pytest.fixture
def preparation_case(tmp_path, monkeypatch):
    root, mounted = tmp_path / "repo", tmp_path / "mounted"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Qualification fixture")
    _git(root, "config", "core.autocrlf", "false")
    authority = dict(purpose="EXECUTION_QUALIFICATION_ONLY", user_authorized=True,
        arms=list(owner.ARMS), seeds=[42], physical_batch_size=128,
        scientific_training_authorized=False, protected_role_access_authorized=False,
        extra_seed_authorized=False, automatic_successor_authorized=False,
        gpu=dict(platform="kaggle2", owner=owner.ACCOUNT, accelerator="NvidiaTeslaT4",
                 allocated_devices=2, wall_ceiling_seconds=1800, allocated_t4_hours_ceiling=1))
    _write(root / owner.BASE / "authority.json", authority)
    _write(root / owner.BASE / "cpu_acceptance.json", {"scope": "generated-fixture"})
    _write(root / "experiments/modern2d_model_integration/integration_attempt_report.json", {})
    entry = root / owner.ENTRY
    entry.parent.mkdir(parents=True)
    entry.write_bytes(
        b"EXPECTED_LAUNCH_SHA256 = None\n"
        b"EXPECTED_SOURCE_DATASET = None\n"
        b"MOUNT_RESOLVER_SOURCE = None\n"
    )
    helper = root / "src/molgap/kaggle_mounts.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "src/molgap/kaggle_mounts.py").read_bytes())
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "Generated preparation inputs")

    asset = mounted / owner.MOUNT
    receipt = dict(status="DATA_ROLE_ACCEPTED", scope="OFFICIAL_TRAIN_CPU_PREPARATION_ONLY",
        rule_identity="4" * 64, roles={role: {
            "sha256": ("2" if role == "train_100k" else "3") * 64,
            "logical_content_sha256": ("5" if role == "train_100k" else "6") * 64}
            for role in ("train_100k", "common_dev")})
    acceptance_path = asset / "indexed_assets/acceptance.json"
    _write(acceptance_path, receipt)
    acceptance_sha = file_digest(acceptance_path)
    binding = dict(format="molgap-indexed-training-input-v1", data_root="indexed_assets",
        scope="ACCEPTED_INDEXED_INPUT_V1", train=_role_binding("train_100k", acceptance_sha),
        development=_role_binding("common_dev", acceptance_sha))
    binding_path = asset / "training_input.json"
    _write(binding_path, binding)
    models = {}
    for arm, interface in owner.ARMS.items():
        feature = {"fixture": "generated-metadata-only", "family": list(interface), "role": "train_100k"}
        models[arm] = dict(family=list(interface),
            model_config=asdict(import_module(FAMILIES[interface].source_module).CONFIG_TYPE()),
            feature_bindings={"train_100k": feature},
            feature_binding_sha256={"train_100k": canonical_fingerprint(feature)})
    metadata = dict(format=METADATA_FORMAT, status="INPUT_METADATA_PREPARED",
        training_input=dict(path="training_input.json", sha256=file_digest(binding_path),
                            bytes=binding_path.stat().st_size), data_root="indexed_assets",
        export_acceptance_sha256=acceptance_sha, parent_acceptance_sha256="7" * 64,
        inventory_sha256="8" * 64, target_mean_eV=1.5, target_std_eV=0.75,
        normalization=dict(role="train_100k", dtype="float64", correction=1, std_floor=1e-6),
        roles={role: dict(rows=rows, source_idx_tensor_sha256="9" * 64)
               for role, rows in (("train_100k", 100000), ("common_dev", 50000))},
        models=models, development_label_statistics_exposed=False,
        scientific_training_authorized=False, publication_status="NOT_PUBLISHED")
    metadata_path = asset / "qualification_inputs.json"
    _write(metadata_path, metadata)
    calls = []

    def tiny_factory(adapter):
        def construct(record):
            calls.append((adapter.family, deepcopy(record)))
            return torch.nn.Linear(2, 2)
        return construct

    monkeypatch.setattr("molgap.shared_model_adapter.resolve_model_factory", tiny_factory)
    monkeypatch.setattr(owner, "PARAMETERS", {arm: 6 for arm in owner.ARMS})

    def no_graphs(*args, **kwargs):
        raise AssertionError("Local declaration preparation opened graph payloads")

    monkeypatch.setattr("molgap.family_training_data.load_training_data", no_graphs)
    monkeypatch.setattr("molgap.data.indexed_inputs.IndexedRoleDataset.__init__", no_graphs)
    return SimpleNamespace(root=root, mounted=mounted, asset=asset, metadata=metadata,
        metadata_path=metadata_path, initial=tmp_path / "initial", output=tmp_path / "prepared",
        calls=calls)


def _freeze(case):
    return owner.freeze_qualification_declarations(case.root, case.metadata_path,
        file_digest(case.metadata_path), init_root=case.initial)


@pytest.fixture
def frozen_case(preparation_case):
    preparation_case.result = _freeze(preparation_case)
    return preparation_case


@pytest.fixture
def namespaced_frozen_case(preparation_case):
    namespaced = preparation_case.mounted / "datasets" / owner.ACCOUNT / owner.MOUNT
    namespaced.parent.mkdir(parents=True)
    preparation_case.asset.rename(namespaced)
    preparation_case.asset = namespaced
    preparation_case.metadata_path = namespaced / "qualification_inputs.json"
    preparation_case.result = _freeze(preparation_case)
    return preparation_case


def _commit(case):
    _git(case.root, "add", ".")
    _git(case.root, "-c", "commit.gpgsign=false", "commit", "-m", "Freeze generated qualification")


def _package(case, expected=None):
    return owner.prepare_package(case.root, case.output, init_root=case.initial,
        mounted_root=case.mounted, expected_spec_identity=expected or case.result["spec_identity"])


def test_train_only_catalog_and_frozen_metadata_schema(frozen_case):
    case = frozen_case
    spec = ExperimentSpec(_read(case.root / owner.BASE / "experiment_spec.json"))
    assert spec.identity == case.result["spec_identity"]
    assert METADATA == {"acceptance_requirements", "train_role_identity", "development_role_identity"}
    assert (case.root / owner.BASE / "qualification_inputs.json").read_bytes() == case.metadata_path.read_bytes()
    assert (case.root / owner.BASE / "training_input.json").read_bytes() == (case.asset / "training_input.json").read_bytes()
    for arm in spec.to_dict()["arms"]:
        interface = (arm["family"]["name"], arm["family"]["version"])
        assert FAMILIES[interface].roles == ("train",)
        assert FAMILIES[interface].optional_roles == ("development",)
        assert [r["role"] for r in arm["data"]["roles"]] == ["train"]
        record, recipe = read_recipe(case.root / owner.BASE / arm["arm_id"] / "contract.json")
        validate_screen_recipe(spec, arm["arm_id"], record)
        assert recipe.dataset.development is False and recipe.development_rows == 0
        assert recipe.dataset.manifest == "training_input.json"
        assert recipe.qualification.scratch_updates_per_model == 6
        assert recipe.target_mean_eV == 1.5 and recipe.target_std_eV == 0.75
        assert record["development_role_identity"] is None
        assert record["acceptance_requirements"]["source_idx_sha256"] is None
    for interface in (("ppgt_small", "1"), ("esa_small", "1"), ("esa_small", "2")):
        assert FAMILIES[interface].roles == ("train", "development")
    assert [interface for interface, _ in case.calls] == list(owner.ARMS.values())


@pytest.mark.parametrize("change", ["metadata-pin", "binding-bytes", "default-config", "feature-pin"])
def test_freeze_rejects_changed_pins_or_defaults(preparation_case, change):
    case = preparation_case
    expected = file_digest(case.metadata_path)
    if change == "metadata-pin":
        expected = "0" * 64
    elif change == "binding-bytes":
        path = case.asset / "training_input.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        values = case.metadata["models"]["ppgt_author_v2"]
        if change == "default-config":
            values["model_config"]["hidden"] += 1
        else:
            values["feature_binding_sha256"]["train_100k"] = "0" * 64
        _write(case.metadata_path, case.metadata)
        expected = file_digest(case.metadata_path)
    with pytest.raises(ValueError):
        owner.freeze_qualification_declarations(case.root, case.metadata_path, expected, init_root=case.initial)
    assert not (case.root / owner.BASE / "preparation.json").exists()


@pytest.mark.parametrize("change", ["status", "data-root", "development-statistics", "training",
    "row-count", "missing-model", "family", "binding-scope"])
def test_metadata_scope_rejected_before_initialization(preparation_case, change):
    case = preparation_case
    if change == "status":
        case.metadata["status"] = "INCOMPLETE"
    elif change == "data-root":
        case.metadata["data_root"] = "../other-assets"
    elif change == "development-statistics":
        case.metadata["development_label_statistics_exposed"] = True
    elif change == "training":
        case.metadata["scientific_training_authorized"] = True
    elif change == "row-count":
        case.metadata["roles"]["common_dev"]["rows"] = 49999
    elif change == "missing-model":
        del case.metadata["models"]["esa_native_v3"]
    elif change == "family":
        case.metadata["models"]["ppgt_author_v2"]["family"] = ["ppgt_small", "1"]
    else:
        binding_path = case.asset / "training_input.json"
        binding = _read(binding_path)
        binding["scope"] = "CPU_SYNTHETIC_ONLY"
        _write(binding_path, binding)
        case.metadata["training_input"].update(sha256=file_digest(binding_path), bytes=binding_path.stat().st_size)
    _write(case.metadata_path, case.metadata)
    with pytest.raises(ValueError):
        _freeze(case)
    assert case.calls == [] and not case.initial.exists()
    assert not (case.root / owner.BASE / "experiment_spec.json").exists()


@pytest.mark.parametrize("change", ["authority", "seed", "batch", "training", "wall"])
def test_authority_cannot_expand_qualification(preparation_case, change):
    case = preparation_case
    path = case.root / owner.BASE / "authority.json"
    record = _read(path)
    if change == "authority":
        record["user_authorized"] = False
    elif change == "seed":
        record["seeds"] = [42, 43]
    elif change == "batch":
        record["physical_batch_size"] = 64
    elif change == "training":
        record["scientific_training_authorized"] = True
    else:
        record["gpu"]["wall_ceiling_seconds"] = 1801
    _write(path, record)
    with pytest.raises(ValueError, match="human qualification authority"):
        _freeze(case)
    assert case.calls == [] and not case.initial.exists()


@pytest.mark.parametrize("location", ["existing", "repository"])
def test_initial_state_storage_must_be_fresh_and_external(preparation_case, location):
    case = preparation_case
    initial = case.initial if location == "existing" else case.root / "initial"
    sentinel = initial / "preserve.txt"
    if location == "existing":
        initial.mkdir()
        sentinel.write_bytes(b"preserve existing storage")
    with pytest.raises(ValueError, match="fresh external E:"):
        owner.freeze_qualification_declarations(case.root, case.metadata_path,
            file_digest(case.metadata_path), init_root=initial)
    if location == "existing":
        assert sentinel.read_bytes() == b"preserve existing storage"
    else:
        assert not initial.exists()
    assert case.calls == []


def test_frozen_and_incomplete_declarations_cannot_be_overwritten(frozen_case):
    case = frozen_case
    preparation = case.root / owner.BASE / "preparation.json"
    preparation.unlink()
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match="already frozen"):
        owner.freeze_qualification_declarations(case.root, case.metadata_path,
            file_digest(case.metadata_path), init_root=case.initial.parent / "second-initial")
    with pytest.raises(FileNotFoundError):
        _package(case)
    assert _snapshot(case.root) == before
    assert not case.output.exists()


def test_package_uses_shared_commit_guard_before_staging(frozen_case):
    with pytest.raises((ValueError, RuntimeError), match="untracked|committed"):
        _package(frozen_case)
    assert not (frozen_case.output / "inputs").exists()


def test_package_rejects_changed_committed_source(frozen_case):
    case = frozen_case
    _commit(case)
    entry = case.root / owner.ENTRY
    entry.write_bytes(entry.read_bytes() + b"# changed after freeze\n")
    with pytest.raises(RuntimeError, match="committed"):
        _package(case)
    assert entry.read_bytes().endswith(b"# changed after freeze\n")
    assert not (case.output / "inputs").exists()


def test_package_rejects_changed_spec_before_output(frozen_case):
    case = frozen_case
    with pytest.raises(ValueError, match="Spec identity differs"):
        _package(case, expected="0" * 64)
    assert not case.output.exists()


def test_package_retains_incomplete_output_and_refuses_duplicate(frozen_case):
    case = frozen_case
    _commit(case)
    state = case.initial / "ppgt_author_v2.pt"
    state.write_bytes(state.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="initial state artifact changed"):
        _package(case)
    before = _snapshot(case.output)
    assert "package/package_manifest.json" in before
    with pytest.raises(ValueError, match="fresh external output root"):
        _package(case)
    assert _snapshot(case.output) == before


def test_package_validates_exact_launch_then_shared_release_and_plan_many(frozen_case, monkeypatch):
    case = frozen_case
    _commit(case)
    events = []
    runtime = import_module("molgap.modern2d_qualification_runtime")

    def checked_launch(**kwargs):
        result = validate_launch(**kwargs)
        events.append("launch")
        return result

    def release(spec, package, **kwargs):
        events.append("release")
        assert kwargs["launch_config"] is None
        assert set(kwargs["initial_states"]) == set(owner.ARMS)
        assert kwargs["entry_script"] == case.output / "run_modern2d_qualification.py"
        assert kwargs["recipe_files"] == {arm: owner.BASE + "/" + arm + "/contract.json" for arm in owner.ARMS}
        return {"status": "LOCAL_RELEASE_INPUTS_VERIFIED", "scope": "mocked-generic-release-only"}

    def publish(root, plans):
        events.append("plan_many")
        assert root == case.root and len(plans) == 2
        for arm, plan in zip(owner.ARMS, plans):
            assert plan["spec"] == _read(case.root / owner.BASE / arm / "plan_input.json")
            assert plan["output"] == owner.BASE + "/" + arm + "/rml"
            assert plan["spec"]["costs"][0]["category"] == "preflight"
        return {"status": "PLANNED", "batch": {"size": 2}, "results": []}

    def no_rebuild(*args, **kwargs):
        raise AssertionError("Preparation rebuilt RML derived files")

    monkeypatch.setattr(runtime, "validate_launch", checked_launch)
    monkeypatch.setattr("molgap.experiment_preflight.check_release_inputs", release)
    monkeypatch.setattr(import_module("molgap.research_memory.plan"), "plan_many", publish)
    monkeypatch.setattr("molgap.research_memory.compiler.rebuild_research_memory", no_rebuild)
    monkeypatch.setattr("molgap.experiment_prospective.rebuild_research_memory", no_rebuild)
    report = _package(case)
    assert events == ["launch", "release", "plan_many"]
    assert report["status"] == "QUALIFICATION_PACKAGE_PREPARED"
    assert report["qualification_launch_validated"] is True
    assert report["rml_rebuilt"] is report["scientific_training_authorized"] is report["submission_performed"] is False
    launch = _read(case.output / "inputs" / LAUNCH_FILENAME)
    assert launch["max_wall_seconds"] == 1800
    assert [(j["arm_id"], j["device"]) for j in launch["jobs"]] == [(arm, i) for i, arm in enumerate(owner.ARMS)]
    assert launch["private_asset"]["acceptance_sha256"] == case.metadata["export_acceptance_sha256"]
    assert report["launch_sha256"] == file_digest(case.output / "inputs" / LAUNCH_FILENAME)
    entry = (case.output / "run_modern2d_qualification.py").read_text(encoding="utf-8")
    assert "EXPECTED_LAUNCH_SHA256 = " + repr(report["launch_sha256"]) in entry
    assert "EXPECTED_SOURCE_DATASET = " + repr(owner.SOURCE_DATASET) in entry
    helper_bytes = (ROOT / "src/molgap/kaggle_mounts.py").read_bytes()
    assert "MOUNT_RESOLVER_SOURCE = " + repr(helper_bytes.decode("utf-8")) in entry
    with tarfile.open(case.output / "package/source.tar.gz", "r:gz") as archive:
        helpers = [member for member in archive.getmembers()
                   if member.name == "src/molgap/kaggle_mounts.py"]
        assert len(helpers) == 1 and helpers[0].isfile() and not helpers[0].issym()
        assert archive.extractfile(helpers[0]).read() == helper_bytes
    for root in (case.output, case.root / owner.BASE):
        for path in root.rglob("*.json"):
            assert b"E:" not in path.read_bytes().upper(), path


def test_package_accepts_owner_namespaced_dataset_mount(namespaced_frozen_case, monkeypatch):
    case = namespaced_frozen_case
    _commit(case)
    runtime = import_module("molgap.modern2d_qualification_runtime")
    observed = []

    def checked_launch(**kwargs):
        observed.append(kwargs["mounted_root"])
        return validate_launch(**kwargs)

    def release(*args, **kwargs):
        return {"status": "LOCAL_RELEASE_INPUTS_VERIFIED", "scope": "mocked-generic-release-only"}

    def publish(root, plans):
        return {"status": "PLANNED", "batch": {"size": len(plans)}, "results": []}

    def no_rebuild(*args, **kwargs):
        raise AssertionError("Preparation rebuilt RML derived files")

    monkeypatch.setattr(runtime, "validate_launch", checked_launch)
    monkeypatch.setattr("molgap.experiment_preflight.check_release_inputs", release)
    monkeypatch.setattr(import_module("molgap.research_memory.plan"), "plan_many", publish)
    monkeypatch.setattr("molgap.research_memory.compiler.rebuild_research_memory", no_rebuild)
    monkeypatch.setattr("molgap.experiment_prospective.rebuild_research_memory", no_rebuild)

    report = _package(case)
    assert report["status"] == "QUALIFICATION_PACKAGE_PREPARED"
    assert observed == [case.mounted]
    launch = _read(case.output / "inputs" / LAUNCH_FILENAME)
    assert launch["private_asset"]["dataset_source"] == owner.DATASET
    assert launch["private_asset"]["mount"] == owner.MOUNT

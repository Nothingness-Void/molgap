"""Declaration-only fixtures; typed Recipe/builders are mocked, never torch.

Execution is delegated to Luna. These checks do not qualify a GPU, train a
model, publish prospective RML or turn a planned reference into accepted evidence.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from molgap import modern2d_training_prepare as owner
from molgap.common.wire import canonical_fingerprint
from molgap.experiment_spec import ExperimentSpec, FAMILIES
from molgap.modern2d_feature_binding import validate_indexed_feature_binding
from molgap.research_memory.trace import file_digest, json_bytes


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def _read(path):
    return json.loads(path.read_bytes())


@dataclass
class _Dataset:
    manifest: str
    manifest_sha256: str
    development: bool


@dataclass
class _Schedule:
    kind: str
    total_units: int
    minimum_lr: float
    warmup_units: int
    reference_epochs: int


class _Recipe:
    def __init__(self, **values):
        self.values = values
        self.__dict__.update(values)


def _bindings(recipe):
    def ref(name):
        return {"name": name, "version": "1", "sha256": canonical_fingerprint({"name": name, "family": recipe.family})}
    return {"base": ref(recipe.family), "sampler": ref("frozen-family-pass-order"),
        "transform": ref("frozen-train-mean-sample-std"), "objective": ref("normalized-gap-l1")}


@pytest.fixture
def case(tmp_path, monkeypatch):
    root, initial, roles = tmp_path / "repo", tmp_path / "initial", tmp_path / "roles"
    root.mkdir()
    initial.mkdir()
    old = root / owner.OLD_BASE
    metadata = _read(ROOT / owner.OLD_BASE / "qualification_inputs.json")
    binding = _read(ROOT / owner.OLD_BASE / "training_input.json")
    old_spec = _read(ROOT / owner.OLD_BASE / "experiment_spec.json")
    preparation = _read(ROOT / owner.OLD_BASE / "preparation.json")
    for name, role in (("train", "train_100k"), ("development", "common_dev")):
        row = metadata["roles"][role]
        manifest = {"role": role, "rows": row["rows"], "protected": False, "pure_2d": True,
            "status": "complete", "ordering": "ascending-source-index", "version_identity": "4" * 64,
            "source_index_sha256": row["source_idx_tensor_sha256"],
            "graph_chunks": [{"topology_sha256": "5" * 64}]}
        _write(roles / (role + ".json"), manifest)
        manifest_sha = file_digest(roles / (role + ".json"))
        authority = binding[name]["authority"]
        authority.update(manifest_sha256=manifest_sha, version_identity="4" * 64)
        history = binding[name]["role_history"]
        history["manifest_sha256"] = manifest_sha
        binding[name]["permit"].update(manifest_sha256=manifest_sha, version_identity="4" * 64,
            authority_sha256=canonical_fingerprint(authority), role_history_sha256=canonical_fingerprint(history))
        for model in metadata["models"].values():
            feature = model["feature_bindings"][role]
            feature.update(base_dataset_manifest_sha=manifest_sha,
                role_history_sha=canonical_fingerprint(history),
                source_topology_sha=canonical_fingerprint({"chunks": ["5" * 64]}))
            model["feature_binding_sha256"][role] = canonical_fingerprint(feature)
    _write(old / "training_input.json", binding)
    metadata["training_input"].update(sha256=file_digest(old / "training_input.json"),
        bytes=(old / "training_input.json").stat().st_size)
    _write(old / "qualification_inputs.json", metadata)
    preparation["metadata_sha256"] = file_digest(old / "qualification_inputs.json")
    for arm_id in owner.ARMS:
        contract = _read(ROOT / owner.OLD_BASE / arm_id / "contract.json")
        _write(old / arm_id / "contract.json", contract)
        arm = next(a for a in old_spec["arms"] if a["arm_id"] == arm_id)
        arm["training"]["recipe"]["sha256"] = file_digest(old / arm_id / "contract.json")
        # Plain bytes with a .pt suffix prove the freezer never deserializes it.
        state = initial / (arm_id + ".pt")
        state.write_bytes(("not a checkpoint: " + arm_id).encode())
        next(j for j in preparation["jobs"] if j["arm_id"] == arm_id)["initial_state"]["sha256"] = file_digest(state)
    _write(old / "experiment_spec.json", old_spec)
    preparation["spec_identity"] = canonical_fingerprint(old_spec)
    _write(old / "preparation.json", preparation)
    permit = binding["development"]["permit"]
    target = tmp_path / "target.json"
    _write(target, {"rows": 50000, "development_target_sha256": "7" * 64,
        "development_source_idx_sha256": metadata["roles"]["common_dev"]["source_idx_tensor_sha256"],
        "target_encoding": owner.TARGET_ENCODING, "role_manifest_sha256": permit["manifest_sha256"],
        "export_acceptance_sha256": permit["acceptance_sha256"]})
    for name, values in {
        "molgap.family_training_data": {"DatasetSelection": _Dataset},
        "molgap.family_training_recipe": {"FamilyRecipe": _Recipe, "ScheduleSpec": _Schedule},
        "molgap.family_training_runner": {"family_arm_bindings": _bindings},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    calls = []

    def build(family, *, configuration, source_idx_sha256, target_sha256,
              train_role_identity, development_role_identity):
        calls.append(configuration)
        values = deepcopy(configuration.values)
        for name in ("dataset", "schedule", "formal_training"):
            values[name] = asdict(values[name])
        return {"format": "mock-typed-recipe", **values,
            "train_role_identity": train_role_identity, "development_role_identity": development_role_identity,
            "acceptance_requirements": {"optimizer_steps": 46860, "completed_passes": 60, "next_batch": 0,
                "sample_presentations": 5998080, "development_rows": 50000,
                "source_idx_sha256": source_idx_sha256, "target_sha256": target_sha256, "precision": "fp32"}}

    monkeypatch.setattr("molgap.experiment_execution.build_family_recipe", build)
    monkeypatch.setattr("molgap.modern2d_feature_binding.indexed_transform_identity", lambda family: "8" * 64)
    snapshot = {"format": "molgap-modern2d-formal-source-identity-v1", "source_commit": COMMIT,
        "source_profile": "runtime-v1", "runtime_files": ["src/molgap/modern2d_training.py"],
        "files": {"src/molgap/modern2d_training.py": "9" * 64}}
    monkeypatch.setattr(owner, "_source_snapshot", lambda *args: deepcopy(snapshot))
    return SimpleNamespace(root=root, initial=initial, roles=roles, target=target,
        metadata=metadata, binding=binding, calls=calls, snapshot=snapshot)


def _freeze(case, **overrides):
    options = dict(authorization="Authorize both full60-pass seed42 screens within aggregate9 T4-hours",
        development_target_manifest=case.target, development_target_manifest_sha256=file_digest(case.target),
        init_root=case.initial, role_manifest_root=case.roles, timestamp="2026-10-18T00:00:00Z")
    options.update(overrides)
    return owner.freeze_formal_screen(case.root, **options)


def test_exact_two_reference_declarations_and_normal_workflow(case):
    old_bytes = {p: p.read_bytes() for p in (case.root / owner.OLD_BASE).rglob("*.json")}
    result = _freeze(case)
    base = case.root / owner.BASE
    spec = ExperimentSpec(_read(base / "experiment_spec.json"))
    declaration = spec.to_dict()
    plan, workflow = _read(base / "acceptance_plan.json"), _read(base / "workflow.json")
    assert result["spec_identity"] == spec.identity and declaration["logical_run_id"] == owner.RUN
    assert [a["scientific_role"] for a in declaration["arms"]] == ["reference", "reference"]
    assert "same_run_replay" not in declaration["prospective"]
    assert set(plan) == {"format", "spec_identity", "authority", "inputs", "arms"}
    assert plan["format"] == owner.PLAN_FORMAT
    assert [a["adapter"] for a in plan["arms"]] == ["modern2d-formal-session-v1"] * 2
    assert workflow["source_profile"] == "runtime-v1" and "source_files" not in workflow
    assert workflow["kaggle"]["kernel"] == owner.KERNEL
    assert workflow["kaggle"]["datasets"] == [owner.SOURCE_DATASET, owner.DATASET]
    assert [j["device"] for j in workflow["arms"]] == [0, 1]
    assert _read(base / "experiment_input.json")["machine_profile"] == owner.MACHINE_PROFILE
    authority = _read(base / "authority.json")
    assert authority["allocated_t4_hours_ceiling"] == authority["gpu"]["allocated_t4_hours_ceiling"] == 9
    assert authority["gpu"]["wall_ceiling_seconds"] == 16200
    assert authority["run_reference"] == owner.KERNEL and authority["source_dataset"] == owner.SOURCE_DATASET
    assert authority["arms"] == list(owner.ARMS) and authority["seeds"] == [42]
    assert all(authority[k] is False for k in ("scale_500k_authorized", "full_training_authorized",
        "protected_role_access_authorized", "extra_seed_authorized", "automatic_successor_authorized"))
    for recipe in case.calls:
        assert recipe.learning_rate == .001 and recipe.weight_decay == 1e-5
        assert recipe.optimizer_steps == 46860 and recipe.checkpoint_every == 128
        assert asdict(recipe.schedule) == dict(kind="four_stage_warmup_cosine_step", total_units=46860,
            minimum_lr=1e-6, warmup_units=4, reference_epochs=60)
        assert recipe.dataset.manifest == "training_input.json" and recipe.dataset.development
        assert recipe.selection == "best-development-live" and recipe.torch_version == "2.7.1"
        arm_id = next(a for a, family in owner.ARMS.items() if family[0] == recipe.family)
        assert recipe.model_config == case.metadata["models"][arm_id]["model_config"]
    metadata = _read(case.root / plan["inputs"]["path"])
    for arm_id, family in owner.ARMS.items():
        assert metadata["models"][arm_id]["family"] == list(family)
        for role, feature in metadata["models"][arm_id]["feature_bindings"].items():
            assert feature["transform_code_sha"] == "8" * 64
            assert validate_indexed_feature_binding(feature, feature) == metadata["models"][arm_id]["feature_binding_sha256"][role]
    for entry in declaration["prospective"]["arms"]:
        proposal = _read(case.root / entry["plan_spec_ref"])
        assert proposal["trajectory"]["state_at_start"]["source_commit"] == COMMIT
        assert proposal["trajectory"]["state_at_start"]["reference_ids"] == []
        assert proposal["decision_state"]["chosen_action"] == "RUN_SCREEN60"
        assert proposal["costs"][0]["measurement"]["device_hours"] == {"status": "measurement_missing", "value": None}
    assert set(result["source_records"]) <= set(result["commit_required"])
    assert all((case.root / name).is_file() for name in result["commit_required"])
    assert (base / "training_input.json").read_bytes() == (case.root / owner.OLD_BASE / "training_input.json").read_bytes()
    assert all(p.read_bytes() == raw for p, raw in old_bytes.items())
    assert result["noReplay"] and result["comparison_scope"] == "CONTEXT_ONLY"
    assert not result["prepared"] and not result["submitted"] and not result["rml_published"]


def _proof(case):
    """Match Beauvoir's actual nested identity-proof schema, without payloads."""
    permit = case.binding["development"]["permit"]
    source = case.metadata["roles"]["common_dev"]["source_idx_tensor_sha256"]
    return {"schema": "molgap-modern2d-common-dev-identity-v1", "status": "COMMON_DEV_IDENTITY_ACCEPTED",
        "source": {"selected_role": "common_dev"},
        "pinned_metadata": {"role_manifest": {"sha256": permit["manifest_sha256"], "rows": 50000,
            "version_identity": permit["version_identity"], "source_index_sha256": source, "chunk_count": 47},
            "root_acceptance": {"sha256": permit["acceptance_sha256"]},
            "training_input": {"sha256": case.metadata["training_input"]["sha256"],
                "authority": case.binding["development"]["authority"],
                "role_history": case.binding["development"]["role_history"]}},
        "source_index_tensor": {"role": "common_dev", "proof_rows": 50000,
            "sha256": source, "encoding": "little-endian int64 raw bytes"},
        "target_tensor": {"role": "common_dev", "proof_rows": 50000, "sha256": "7" * 64,
            "encoding": "little-endian float64 raw bytes", "chunk_count": 47, "chunk_rows": [1000] * 46 + [4000]},
        "execution_boundary": {"role_manifest_and_chunk_hashes_verified": True,
            **{k: False for k in ("training_executed", "models_loaded", "cuda_used", "graphs_rebuilt",
                "official_validation_read", "test_roles_read", "protected_roles_read",
                "train_dataset_instantiated", "audit_label_statistics_exposed")}}}


def test_actual_proof_schema_maps_compact_manifest_and_retains_original(case):
    proof = _proof(case)
    _write(case.target, proof)
    result = _freeze(case)
    base = case.root / owner.BASE
    target = _read(base / "target_manifest.json")
    assert target["development_target_sha256"] == proof["target_tensor"]["sha256"]
    assert target["target_encoding"] == owner.TARGET_ENCODING
    assert target["development_source_idx_sha256"] == proof["source_index_tensor"]["sha256"]
    assert (case.root / target["identity_proof"]["path"]).read_bytes() == case.target.read_bytes()
    assert target["identity_proof"]["sha256"] == file_digest(case.target)
    assert target["identity_proof"]["path"] in result["source_records"]
    for entry in _read(base / "acceptance_plan.json")["arms"]:
        assert entry["target_manifest"]["sha256"] == file_digest(base / "target_manifest.json")


@pytest.mark.parametrize("change", ["target-pin", "target-order", "target-encoding", "target-provenance",
    "chunk-digests-only", "state-file", "changed-config", "feature-pin", "empty-authorization"])
def test_rejects_invalid_bindings_before_any_declaration_write(case, change):
    options = {}
    if change == "target-pin":
        options["development_target_manifest_sha256"] = "0" * 64
    elif change.startswith("target-") or change == "chunk-digests-only":
        target = _read(case.target)
        key = {"target-order": "development_source_idx_sha256", "target-encoding": "target_encoding",
            "target-provenance": "role_manifest_sha256", "chunk-digests-only": "development_target_sha256"}[change]
        target.pop(key)
        _write(case.target, target)
    elif change == "state-file":
        (case.initial / "ppgt_author_v2.pt").write_bytes(b"changed")
    elif change in {"changed-config", "feature-pin"}:
        metadata = deepcopy(case.metadata)
        model = metadata["models"]["ppgt_author_v2"]
        if change == "changed-config":
            model["model_config"]["hidden"] += 1
        else:
            model["feature_binding_sha256"]["common_dev"] = "0" * 64
        old = case.root / owner.OLD_BASE
        _write(old / "qualification_inputs.json", metadata)
        prep = _read(old / "preparation.json")
        prep["metadata_sha256"] = file_digest(old / "qualification_inputs.json")
        _write(old / "preparation.json", prep)
    else:
        options["authorization"] = " "
    with pytest.raises(ValueError):
        _freeze(case, **options)
    assert not (case.root / owner.BASE).exists()
    assert not (case.root / owner.POLICY).exists()


def test_unknown_source_and_existing_declarations_preserve_every_byte(case, monkeypatch):
    def dirty(*args):
        raise ValueError("source not committed")
    monkeypatch.setattr(owner, "_source_snapshot", dirty)
    with pytest.raises(ValueError, match="not committed"):
        _freeze(case)
    assert not (case.root / owner.BASE).exists()
    sentinel = case.root / owner.BASE / "keep.json"
    _write(sentinel, {"other_worker": True})
    with pytest.raises(ValueError, match="already exist"):
        _freeze(case)
    assert _read(sentinel) == {"other_worker": True}


def test_existing_policy_and_source_change_do_not_publish_partial_declarations(case, monkeypatch):
    _write(case.root / owner.POLICY, {"preserve": True})
    with pytest.raises(ValueError, match="already exists"):
        _freeze(case)
    assert not (case.root / owner.BASE).exists()
    assert _read(case.root / owner.POLICY) == {"preserve": True}


def test_source_change_mid_freeze_fails_before_writes(case, monkeypatch):
    snapshots = iter([case.snapshot, {**case.snapshot, "source_commit": "b" * 40}])
    monkeypatch.setattr(owner, "_source_snapshot", lambda *args: next(snapshots))
    with pytest.raises(ValueError, match="changed during"):
        _freeze(case)
    assert not (case.root / owner.BASE).exists()

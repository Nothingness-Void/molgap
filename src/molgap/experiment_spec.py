"""Versioned prospective experiment contracts; no execution or readiness authority.

Registry versions describe this interface, not accepted model generations.
Digests bind declarations only: adapters must verify bytes, fixed recipe values,
role authorization and runtime certificates before using a spec to execute.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PureWindowsPath
from types import MappingProxyType

from .screen_policy import canonical_fingerprint


SCHEMA_VERSION = "molgap-experiment-spec-v1"
SCHEMA_VERSION_V2 = "molgap-experiment-spec-v2"
TERMINAL_PROTOCOL = "molgap-experiment-terminal-descriptor-v1"


@dataclass(frozen=True)
class FamilyContract:
    """Static adapter lookup key and supported declaration contracts."""

    name: str
    version: str
    source_module: str
    recipe: str
    feature_schema: str
    roles: tuple[str, ...]
    sampler: str
    transform: str


FAMILIES = MappingProxyType({
    ("gptrans_t", "1"): FamilyContract(
        "gptrans_t", "1", "molgap.gptrans", "pcqm_gptrans_v4",
        "ogb-atom9-bond3-shortest-path-cap20", ("train", "development"),
        "seed-plus-epoch-global-randperm-v1", "fixed-train-100k-mean-sample-std",
    ),
    ("neural_atom_k1", "1"): FamilyContract(
        "neural_atom_k1", "1", "molgap.qm9_neural_atom", "pcqm_k1_full",
        "ogb-atom9-bond3-rwse16-v1", ("train",),
        "seed42-global-randperm-by-pass-v1", "full-train-mean-sample-std",
    ),
})


@dataclass(frozen=True)
class AddonContract:
    family: str
    exclusive_group: str
    source_module: str


# Each family's replacement group is exclusive; stacking is not supported.
ADDONS = MappingProxyType({
    **{
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_variants")
        for name in ("pair_prenorm", "centered_logits")
    },
    **{
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_memory")
        for name in ("memory_value", "memory_message")
    },
    ("k1_pair_value", "1"): AddonContract(
        "neural_atom_k1", "pair-token-replacement", "molgap.k1_pair_token",
    ),
})


def _object(value, fields: str, path: str) -> dict:
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError(f"{path}: expected exactly fields [{fields}]")
    return value


def _text(value, path: str) -> None:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{path}: expected nonempty trimmed string")


def _digest(value, path: str) -> None:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{path}: expected lowercase SHA-256")


def _identifier(value, path: str) -> None:
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError(f"{path}: expected safe identity")


def _integer(value, path: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path}: expected integer >= {minimum}")


def _choice(value, choices, path: str) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError(f"{path}: unsupported value {value!r}")


def _reference(value, path: str, *, name: str | None = None) -> None:
    _object(value, "name version sha256", path)
    _text(value["name"], path + ".name")
    _choice(value["version"], ("1",), path + ".version")
    _digest(value["sha256"], path + ".sha256")
    if name is not None and value["name"] != name:
        raise ValueError(f"{path}: expected {name}")


def _list(value, path: str, *, nonempty: bool = True) -> None:
    if type(value) is not list or (nonempty and not value):
        raise ValueError(f"{path}: expected {'nonempty ' if nonempty else ''}array")


def _repo_path(value, path: str, *, output: bool = False) -> None:
    _text(value, path)
    if ("\\" in value or ":" in value or value.startswith("/")
            or PureWindowsPath(value).drive or PureWindowsPath(value).root):
        raise ValueError(f"{path}: expected repository-relative POSIX path")
    parts = value.split("/")
    for part in parts:
        if (part in {"", ".", ".."} or part.endswith((" ", "."))
                or any(ord(char) < 32 or char in '<>"|?*' for char in part)
                or PureWindowsPath(part).is_reserved()):
            raise ValueError(f"{path}: unsafe path component")
    if output:
        if len(parts) < 2 or parts[0] != "experiments":
            raise ValueError(f"{path}: output must be below experiments/")
    elif not value.endswith(".json"):
        raise ValueError(f"{path}: expected JSON plan input")


def _arm(arm: dict) -> None:
    _object(arm, "arm_id scientific_role family base initialization data training addons addon_semantics", "arm")
    _identifier(arm["arm_id"], "arm.arm_id")
    _choice(arm["scientific_role"], ("reference", "candidate", "ablation"), "arm.scientific_role")
    family = _object(arm["family"], "name version", "family")
    _text(family["name"], "family.name")
    _text(family["version"], "family.version")
    contract = FAMILIES.get((family["name"], family["version"]))
    if contract is None:
        raise ValueError("Unknown family/version")
    _reference(arm["base"], "base")
    init = _object(arm["initialization"], "kind seed state_sha256", "initialization")
    _choice(init["kind"], ("random", "frozen_state"), "initialization.kind")
    _integer(init["seed"], "initialization.seed", 0)
    _digest(init["state_sha256"], "initialization.state_sha256")

    data = _object(arm["data"], "dataset split roles feature_schema feature_sha256 target", "data")
    _reference(data["dataset"], "data.dataset", name="pcqm4mv2")
    _reference(data["split"], "data.split")
    _choice(data["feature_schema"], (contract.feature_schema,), "data.feature_schema")
    _digest(data["feature_sha256"], "data.feature_sha256")
    _choice(data["target"], ("pcqm4mv2-gap-eV-direct",), "data.target")
    _list(data["roles"], "data.roles")
    roles = []
    for role in data["roles"]:
        _object(role, "role membership_sha256 row_order_sha256 usage_sha256", "data.role")
        _choice(role["role"], contract.roles, "data.role.role")
        for field in ("membership_sha256", "row_order_sha256", "usage_sha256"):
            _digest(role[field], "data.role." + field)
        roles.append(role["role"])
    if len(set(roles)) != len(roles) or set(roles) != set(contract.roles):
        raise ValueError("Data roles must match the recipe exactly, without duplicates")

    training = _object(arm["training"], "recipe overrides objective sampler transform", "training")
    _reference(training["recipe"], "training.recipe", name=contract.recipe)
    # Frozen legacy recipes have no override allowance. A reviewed new recipe
    # version must define typed bounds before any override can become executable.
    _object(training["overrides"], "", "training.overrides")
    if init["seed"] != 42:
        raise ValueError("Frozen recipe initialization requires seed 42")
    _reference(training["objective"], "training.objective", name="normalized-gap-l1")
    _reference(training["sampler"], "training.sampler", name=contract.sampler)
    _reference(training["transform"], "training.transform", name=contract.transform)

    _list(arm["addons"], "addons", nonempty=False)
    expected = "ordered" if arm["addons"] else "baseline"
    _choice(arm["addon_semantics"], (expected,), "addon_semantics")
    groups = set()
    for addon in arm["addons"]:
        _object(addon, "name version config source_sha256", "addon")
        _text(addon["name"], "addon.name")
        _text(addon["version"], "addon.version")
        extension = ADDONS.get((addon["name"], addon["version"]))
        if extension is None:
            raise ValueError("Unknown addon/version")
        if extension.family != contract.name or extension.exclusive_group in groups:
            raise ValueError("Incompatible addon combination/family")
        groups.add(extension.exclusive_group)
        _object(addon["config"], "", "addon.config")
        _digest(addon["source_sha256"], "addon.source_sha256")


def validate_experiment_spec(payload: dict) -> dict:
    """Return a detached strict JSON declaration; never certify runtime/evidence."""
    spec = _object(payload, "schema_version experiment_id logical_run_id arms platform prospective evidence terminal_protocol", "spec")
    _choice(spec["schema_version"], (SCHEMA_VERSION, SCHEMA_VERSION_V2), "schema_version")
    _identifier(spec["experiment_id"], "experiment_id")
    _identifier(spec["logical_run_id"], "logical_run_id")
    _choice(spec["terminal_protocol"], (TERMINAL_PROTOCOL,), "terminal_protocol")
    _list(spec["arms"], "arms")
    for arm in spec["arms"]:
        _arm(arm)
    ids = [arm["arm_id"] for arm in spec["arms"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate arm_id")
    platform = _object(spec["platform"], "name accelerator device_count cpu_cores memory_gib atomic_checkpoints retrievable_chunks", "platform")
    _choice(platform["name"], ("local", "kaggle", "scnet", "ims"), "platform.name")
    _text(platform["accelerator"], "platform.accelerator")
    for field in ("device_count", "cpu_cores", "memory_gib"):
        _integer(platform[field], "platform." + field)
    # Device count declares resources; runner/platform gates must verify arm,
    # process and visible-device isolation before execution.
    for field in ("atomic_checkpoints", "retrievable_chunks"):
        if platform[field] is not True:
            raise ValueError("Durability requirements cannot be disabled")
    if spec["schema_version"] == SCHEMA_VERSION:
        prospective = _object(spec["prospective"], "trajectory_id hypothesis cheapest_falsifier stop_rule budget_sha256", "prospective")
        _identifier(prospective["trajectory_id"], "prospective.trajectory_id")
        for field in ("hypothesis", "cheapest_falsifier", "stop_rule"):
            _text(prospective[field], "prospective." + field)
        _digest(prospective["budget_sha256"], "prospective.budget_sha256")
    else:
        prospective = spec["prospective"]
        if type(prospective) is not dict or set(prospective) not in ({"arms"}, {"arms", "same_run_replay"}):
            raise ValueError("prospective: expected arms and optional same_run_replay")
        _list(prospective["arms"], "prospective.arms")
        mapped_ids = []
        trajectory_ids = set()
        outputs = set()
        for entry in prospective["arms"]:
            _object(entry, "arm_id trajectory_id plan_spec_ref plan_spec_sha256 output", "prospective.arm")
            for field in ("arm_id", "trajectory_id"):
                _identifier(entry[field], "prospective.arm." + field)
            _digest(entry["plan_spec_sha256"], "prospective.arm.plan_spec_sha256")
            _repo_path(entry["plan_spec_ref"], "prospective.arm.plan_spec_ref")
            _repo_path(entry["output"], "prospective.arm.output", output=True)
            mapped_ids.append(entry["arm_id"])
            if entry["trajectory_id"] in trajectory_ids:
                raise ValueError("Duplicate prospective trajectory_id")
            trajectory_ids.add(entry["trajectory_id"])
            output_key = entry["output"].casefold()
            if output_key in outputs:
                raise ValueError("Duplicate prospective output")
            outputs.add(output_key)
        if len(mapped_ids) != len(set(mapped_ids)) or set(mapped_ids) != set(ids):
            raise ValueError("Prospective arm mapping must match spec arms exactly once")
        if "same_run_replay" in prospective:
            paired = _object(prospective["same_run_replay"], "reference_arm_id candidate_arm_ids", "prospective.same_run_replay")
            _identifier(paired["reference_arm_id"], "prospective.same_run_replay.reference_arm_id")
            _list(paired["candidate_arm_ids"], "prospective.same_run_replay.candidate_arm_ids")
            for candidate_id in paired["candidate_arm_ids"]:
                _identifier(candidate_id, "prospective.same_run_replay.candidate_arm_ids")
            if len(set(paired["candidate_arm_ids"])) != len(paired["candidate_arm_ids"]):
                raise ValueError("Duplicate same-run candidate arm")
            roles = {arm["arm_id"]: arm["scientific_role"] for arm in spec["arms"]}
            if roles.get(paired["reference_arm_id"]) != "reference" or any(
                roles.get(candidate_id) != "candidate" for candidate_id in paired["candidate_arm_ids"]
            ):
                raise ValueError("same-run replay arms must declare reference and candidate roles")
    evidence = _object(spec["evidence"], "policy required_artifacts", "evidence")
    _reference(evidence["policy"], "evidence.policy", name="molgap-v5")
    _list(evidence["required_artifacts"], "evidence.required_artifacts")
    required = {"v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"}
    for item in evidence["required_artifacts"]:
        _choice(item, required, "evidence.required_artifacts")
    if set(evidence["required_artifacts"]) != required or len(evidence["required_artifacts"]) != len(required):
        raise ValueError("Evidence requirements must be complete and unique")
    return json.loads(_canonical(spec))


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, init=False)
class ExperimentSpec:
    """Immutable validated canonical snapshot for future adapters and packaging."""

    _canonical_json: str

    def __init__(self, payload: dict):
        object.__setattr__(self, "_canonical_json", _canonical(validate_experiment_spec(payload)))

    @classmethod
    def from_json(cls, text: str) -> ExperimentSpec:
        return cls(json.loads(text, object_pairs_hook=_unique_object))

    def to_dict(self) -> dict:
        return json.loads(self._canonical_json)

    def to_json(self) -> str:
        return self._canonical_json

    @property
    def identity(self) -> str:
        return canonical_fingerprint(self.to_dict())

    def family_contract(self, arm_id: str) -> FamilyContract:
        """Resolve a static contract without importing or running model code."""
        for arm in self.to_dict()["arms"]:
            if arm["arm_id"] == arm_id:
                family = arm["family"]
                return FAMILIES[(family["name"], family["version"])]
        raise ValueError(f"Unknown arm: {arm_id}")

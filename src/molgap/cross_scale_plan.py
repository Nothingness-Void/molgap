"""Strict draft-only comparison declarations; deliberately not ExperimentSpec.

There is no compiler, publisher, trainer or remote entry here. A future reviewed
adapter must bind real evidence and separate authority using the existing owners.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math

from .common.wire import canonical_fingerprint

FEATURE_FORMATS = {
    "ogb-atom9-bond3-rwse16-v1": frozenset({"x", "edge_index", "edge_attr", "random_walk_pe"}),
    "pair-rrwp-v1": frozenset({"rrwp"}),
    "edge-neighbor-mask-v1": frozenset({"edge_neighbor_mask"}),
    "common-cycle-v1": frozenset({"common_cycle"}),
}


def exact(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{label}: missing or unknown fields")
    return value


def digest(value, label="SHA256"):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label}: lowercase SHA256 required")
    return value


def text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label}: nonempty text required")


def number(value, label, *, integer=False, minimum=0):
    if (type(value) not in ((int,) if integer else (int, float)) or
            not math.isfinite(value) or value < minimum):
        raise ValueError(f"{label}: invalid finite number")


@dataclass(frozen=True)
class CohortIdentity:
    name: str
    dataset: str
    role: str
    role_identity: str
    rows: int
    manifest_sha256: str
    row_index_sha256: str
    source_index_sha256: str
    target_sha256: str
    topology_sha256: str
    target: str
    unit: str
    target_encoding: str
    ordering: str
    role_history_sha256: str
    authority_sha256: str
    official_train_rows: int
    protected: bool = False

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith("sha256"):
                digest(value, f.name)
        for name in ("name", "dataset", "role_identity"):
            text(getattr(self, name), name)
        number(self.rows, "rows", integer=True, minimum=1)
        number(self.official_train_rows, "official train boundary", integer=True, minimum=1)
        expected_prefix = "internal_train" if self.role == "train" else "internal_development"
        if (self.role not in {"train", "development"} or self.protected is not False or
                not self.role_identity.startswith(expected_prefix) or
                any(term in self.role_identity.lower() for term in ("official", "test", "ood", "p8", "external", "shadow", "protected")) or
                self.target != "gap" or self.unit != "eV" or
                self.target_encoding != "float64-le-eV" or self.ordering != "canonical-source-order"):
            raise ValueError("Forbidden role, target, unit, encoding or ordering")

    @classmethod
    def from_dict(cls, value):
        return cls(**exact(value, cls.__dataclass_fields__, "cohort"))


@dataclass(frozen=True)
class RecipeIdentity:
    optimizer_sha256: str
    schedule_sha256: str
    schedule_axis: str
    schedule_horizon_steps: int
    target_transform_sha256: str
    transform_train_source_sha256: str
    sampler_sha256: str
    selection: str
    weight_state: str
    ema_sha256: str
    loss_sha256: str
    physical_batch: int
    drop_last: bool
    precision: str
    tf32: bool
    from_scratch: bool

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name.endswith("sha256"):
                digest(getattr(self, name), name)
        number(self.schedule_horizon_steps, "schedule horizon", integer=True, minimum=1)
        if (self.schedule_axis not in {"optimizer_step", "complete_pass"} or
                self.selection not in {"fixed-step", "best-development"} or
                self.weight_state not in {"live", "ema", "clean-ema"} or
                type(self.physical_batch) is not int or self.physical_batch != 128 or
                self.drop_last is not True or self.precision != "fp32" or self.tf32 is not False or
                self.from_scratch is not True):
            raise ValueError("Recipe axis/selection/precision/batch/scratch policy changed")

    @classmethod
    def from_dict(cls, value):
        return cls(**exact(value, cls.__dataclass_fields__, "recipe"))


@dataclass(frozen=True)
class ModelIdentity:
    family: str
    revision: str
    source_sha256: str
    architecture_sha256: str
    fidelity_sha256: str
    feature_sha256: str
    feature_format: str
    feature_approval_sha256: str
    pure_2d: bool
    parameters: int
    parameter_ceiling: int

    def __post_init__(self):
        from .experiment_execution import TRAINING_ADAPTERS
        if (self.family, self.revision) not in TRAINING_ADAPTERS:
            raise ValueError("Unregistered family/revision; a draft cannot make it executable")
        for name in self.__dataclass_fields__:
            if name.endswith("sha256"):
                digest(getattr(self, name), name)
        if self.feature_format not in FEATURE_FORMATS:
            raise ValueError("Unapproved feature format/version")
        for name in ("parameters", "parameter_ceiling"):
            number(getattr(self, name), name, integer=True, minimum=1)
        if self.pure_2d is not True or self.parameters > self.parameter_ceiling:
            raise ValueError("Model exceeds parameter or pure-2D boundary")

    @classmethod
    def from_dict(cls, value):
        return cls(**exact(value, cls.__dataclass_fields__, "model"))


@dataclass(frozen=True)
class DraftComparison:
    comparison_id: str
    question: str
    mode: str
    view: str
    train100k: CohortIdentity
    train500k: CohortIdentity
    development: CohortIdentity
    models: tuple[ModelIdentity, ...]
    recipes: tuple[RecipeIdentity, ...]
    seeds: tuple[int, ...]
    hardware: str
    diagnostic_steps: int
    primary_steps: int
    milestones: tuple[int, ...]
    reference_id: str
    reference_bundle_sha256: str
    native_budget: dict
    finalist_policy: dict
    stop_rules: dict
    source_commit: str
    format: str = "molgap-cross-scale-draft-v1"
    status: str = "DRAFT_ONLY"
    authority: str = "NOT_AUTHORIZED_FOR_TRAINING"

    def __post_init__(self):
        if (self.format != "molgap-cross-scale-draft-v1" or self.status != "DRAFT_ONLY" or
                self.authority != "NOT_AUTHORIZED_FOR_TRAINING"):
            raise ValueError("Cross-scale V1 accepts DRAFT_ONLY, never training authority")
        for name in ("comparison_id", "question", "hardware", "reference_id"):
            text(getattr(self, name), name)
        if self.mode not in {"SCIENTIFIC_COMPARE", "ENGINEERING_PROBE"} or self.view not in {"MATCHED_EXPOSURE", "TUNED_RECIPE"}:
            raise ValueError("Unknown comparison mode/view")
        if (any(type(c) is not CohortIdentity for c in (self.train100k, self.train500k, self.development)) or
                self.train100k.role != "train" or self.train500k.role != "train" or self.development.role != "development" or
                len({c.dataset for c in (self.train100k, self.train500k, self.development)}) != 1 or
                self.development.role_identity in {self.train100k.role_identity, self.train500k.role_identity} or
                self.development.source_index_sha256 in {self.train100k.source_index_sha256, self.train500k.source_index_sha256}):
            raise ValueError("Inconsistent dataset/role identities")
        if (type(self.models) is not tuple or not self.models or any(type(m) is not ModelIdentity for m in self.models) or
                type(self.recipes) is not tuple or len(self.recipes) != 2 or any(type(r) is not RecipeIdentity for r in self.recipes)):
            raise ValueError("Typed models and two scale recipes required")
        if type(self.seeds) is not tuple or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Explicit unique seed list required")
        for seed in self.seeds:
            number(seed, "seed", integer=True)
        for name in ("diagnostic_steps", "primary_steps"):
            number(getattr(self, name), name, integer=True, minimum=1)
        if self.diagnostic_steps > self.primary_steps:
            raise ValueError("Diagnostic exceeds primary horizon")
        if type(self.milestones) is not tuple or not self.milestones or list(self.milestones) != sorted(set(self.milestones)):
            raise ValueError("Unique ordered exact-step milestones required")
        for point in self.milestones:
            number(point, "milestone", integer=True, minimum=1)
            if point > self.primary_steps:
                raise ValueError("Milestone exceeds primary horizon")
        if self.view == "MATCHED_EXPOSURE":
            if self.recipes[0] != self.recipes[1] or self.recipes[0].schedule_axis != "optimizer_step":
                raise ValueError("Matched prefixes require identical step-indexed full LR trajectory")
            if self.recipes[0].transform_train_source_sha256 != self.train100k.source_index_sha256:
                raise ValueError("Shared transform must use authorized common 100K training only")
        elif any(r.transform_train_source_sha256 != c.source_index_sha256
                 for r, c in zip(self.recipes, (self.train100k, self.train500k))):
            raise ValueError("Tuned transforms must remain train-only")
        if any(r.schedule_horizon_steps < self.primary_steps for r in self.recipes):
            raise ValueError("Diagnostic cutoff must not shorten full LR horizon")
        digest(self.reference_bundle_sha256, "reference bundle")
        if type(self.source_commit) is not str or len(self.source_commit) != 40 or any(c not in "0123456789abcdef" for c in self.source_commit):
            raise ValueError("Source commit pin required")
        exact(self.native_budget, {"hardware", "unit", "ceiling", "job_count", "approval_sha256"}, "native budget")
        if self.native_budget["hardware"] != self.hardware or self.native_budget["unit"] not in {"T4_device_hours", "A100_hours", "DCU_hours", "CPU_hours"}:
            raise ValueError("Hardware-native budget mismatch")
        number(self.native_budget["ceiling"], "budget", minimum=0.000001)
        number(self.native_budget["job_count"], "job count", integer=True, minimum=1)
        digest(self.native_budget["approval_sha256"])
        exact(self.finalist_policy, {"minimum_seeds", "material_gain_eV", "approval_sha256", "tuning_budget_per_system"}, "finalist policy")
        number(self.finalist_policy["minimum_seeds"], "finalist seeds", integer=True, minimum=2)
        number(self.finalist_policy["material_gain_eV"], "material effect", minimum=0.000001)
        number(self.finalist_policy["tuning_budget_per_system"], "equal tuning budget", minimum=0.000001)
        digest(self.finalist_policy["approval_sha256"])
        exact(self.stop_rules, {"quality_stop_on_100k_score", "automatic_stqs", "automatic_futility", "hard_failures"}, "stop rules")
        if any(self.stop_rules[k] is not False for k in ("quality_stop_on_100k_score", "automatic_stqs", "automatic_futility")):
            raise ValueError("Uncalibrated quality-based training control forbidden")
        allowed = {"INVALID_CODE", "NONFINITE", "CORRUPT_INPUT", "UNAUTHORIZED_ROLE", "CHECKPOINT_FAILURE", "DECLARED_COST_CEILING"}
        if type(self.stop_rules["hard_failures"]) is not list or set(self.stop_rules["hard_failures"]) != allowed:
            raise ValueError("Only explicit implementation/integrity/cost stops are permitted")
        json.dumps(asdict(self), allow_nan=False)

    @classmethod
    def from_dict(cls, value):
        value = dict(exact(value, cls.__dataclass_fields__, "draft"))
        for name in ("train100k", "train500k", "development"):
            value[name] = CohortIdentity.from_dict(value[name])
        for name, owner in (("models", ModelIdentity), ("recipes", RecipeIdentity)):
            if type(value[name]) is not list:
                raise ValueError("Draft arrays must be lists")
            value[name] = tuple(owner.from_dict(item) for item in value[name])
        for name in ("seeds", "milestones"):
            if type(value[name]) is not list:
                raise ValueError("Draft arrays must be lists")
            value[name] = tuple(value[name])
        return cls(**value)

    @property
    def identity(self):
        return canonical_fingerprint(asdict(self))

    def execution_request(self):
        raise ValueError("DRAFT_ONLY: no compiler/execution/role authority in cross-scale V1")


def review_draft_inputs(root, plan, *, memberships, reference_binding):
    """Review typed cohorts and a pinned V5 reference; this never releases work."""
    from .comparison_readiness import validate_reference_bundle
    from .cross_scale_analysis import read_bound_json
    from .cross_scale_inputs import validate_nested_cohorts
    if type(plan) is not DraftComparison:
        raise TypeError("DraftComparison required")
    if reference_binding.get("sha256") != plan.reference_bundle_sha256:
        raise ValueError("Wrong reference bundle SHA")
    reference = validate_reference_bundle(read_bound_json(root, reference_binding))
    if reference["reference_id"] != plan.reference_id:
        raise ValueError("Wrong accepted reference identity")
    predictions = reference["prediction_manifest"]
    if (predictions["source_idx_sha256"] != plan.development.source_index_sha256 or
            predictions["target_sha256"] != plan.development.target_sha256 or
            predictions["row_count"] != plan.development.rows or
            predictions["evaluation_role_identity"] != plan.development.role_identity):
        raise ValueError("Reference prediction rows/targets/role differ")
    cohorts = validate_nested_cohorts(plan.train100k, plan.train500k, plan.development, memberships=memberships)
    return {"status": "DRAFT_REVIEWED_ONLY", "cohorts": cohorts, "reference_bundle_pinned": True,
            "reference_runtime_and_artifact_acceptance": "NOT_EVALUATED",
            "training_authorized": False, "prospective_published": False}

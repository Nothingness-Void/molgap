"""Small declarations only; session/optimizer/sampler fixtures stay with owners."""
from dataclasses import asdict
import hashlib

from molgap.cross_scale_inputs import index_digest, target_digest
from molgap.cross_scale_plan import CohortIdentity, ModelIdentity, RecipeIdentity
from molgap.comparison_readiness import STRICT_IDENTITY_FIELDS


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def cohorts():
    result, members = [], {}
    for name, role, start, stop in (("synthetic100k", "train", 0, 256),
                                   ("synthetic500k", "train", 0, 512),
                                   ("syntheticdev", "development", 512, 516)):
        ids = list(range(start, stop))
        targets = [i / 17. for i in ids]
        rows = list(range(len(ids)))
        result.append(CohortIdentity(name, "synthetic-PCQM", role,
            ("internal_train_" if role == "train" else "internal_development_") + name, len(ids), sha("manifest"+name), index_digest(rows),
            index_digest(ids), target_digest(targets), sha("topology"+name), "gap", "eV",
            "float64-le-eV", "canonical-source-order", sha("history"+name), sha("authority"+name),3378606))
        members[name] = {"row_indices": rows, "source_indices": ids, "targets": targets}
    return result, members


def plan_dict():
    c, _ = cohorts()
    recipe = asdict(RecipeIdentity(sha("AdamW"), sha("cosine full horizon"), "optimizer_step", 234360,
        sha("common train transform"), c[0].source_index_sha256, sha("sampler"), "fixed-step",
        "live", sha("no EMA"), sha("L1"), 128, True, "fp32", False, True))
    model = asdict(ModelIdentity("gptrans_t", "2", sha("source"), sha("architecture"),
        sha("fidelity-stub-not-qualified"), sha("features"), "ogb-atom9-bond3-rwse16-v1",
        sha("draft-feature-approval"), True, 10, 100))
    return {"format": "molgap-cross-scale-draft-v1", "status": "DRAFT_ONLY",
        "authority": "NOT_AUTHORIZED_FOR_TRAINING", "comparison_id": "DRAFT-SYNTHETIC-ONLY",
        "question": "Synthetic shape/guard validation, not molecular training", "mode": "SCIENTIFIC_COMPARE",
        "view": "MATCHED_EXPOSURE", "train100k": asdict(c[0]), "train500k": asdict(c[1]),
        "development": asdict(c[2]), "models": [model], "recipes": [dict(recipe), dict(recipe)],
        "seeds": [42], "hardware": "CPU-synthetic", "diagnostic_steps": 31240, "primary_steps": 234360,
        "milestones": [3906, 7812, 15624, 31240, 62480, 117180, 234360],
        "reference_id": "DRAFT-PLACEHOLDER-NOT-AN-ACCEPTED-REFERENCE", "reference_bundle_sha256": sha("draft reference"),
        "native_budget": {"hardware": "CPU-synthetic", "unit": "CPU_hours", "ceiling": 1., "job_count": 2,
                          "approval_sha256": sha("draft budget not release")},
        "finalist_policy": {"minimum_seeds": 2, "material_gain_eV": .003,
                            "approval_sha256": sha("draft policy not release"), "tuning_budget_per_system": 1.},
        "stop_rules": {"quality_stop_on_100k_score": False, "automatic_stqs": False,
                       "automatic_futility": False, "hard_failures": ["INVALID_CODE", "NONFINITE", "CORRUPT_INPUT",
                       "UNAUTHORIZED_ROLE", "CHECKPOINT_FAILURE", "DECLARED_COST_CEILING"]},
        "source_commit": "0"*40}


def identity():
    result = {key: "synthetic:"+key for key in STRICT_IDENTITY_FIELDS}
    result.update(seed=42, physical_batch_per_device=128, gradient_accumulation_steps=1,
        optimizer_steps=234360, sample_presentations=234360*128, tf32_enabled=False,
        deterministic_algorithms=True, ema_enabled=False, ema_decay="none", ema_update_frequency="none",
        sampler_identity="sampler42", schedule_axis="optimizer_step", target_unit="eV",
        evaluation_row_sha256=sha("dev rows"), evaluation_target_sha256=sha("dev targets"),
        evaluation_row_count=50000, bn_calibration_identity="not_applicable")
    return result


def curve(values, steps=(1, 2, 3)):
    return [{"optimizer_step": step, "sample_presentations": step*128, "complete_passes": 0,
             "learning_rate": .001, "value": value, "metric": "live_dev_metric", "unit": "eV",
             "weights": "live", "evaluation_role": "internal_development", "metric_kind": "development"}
            for step, value in zip(steps, values)]

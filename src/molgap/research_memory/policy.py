"""Explicit versioned deterministic rules; approval is never a replay effect."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import load_json_object
from .schemas import SHA256_PATTERN, validate_id

POLICY_TYPES = {"screening", "early_stop", "scale_up", "research_action"}
ACTIONS = {"screening": {"PROMOTE", "STOP", "DEFER"},
           "early_stop": {"CONTINUE", "STOP", "DEFER"},
           "scale_up": {"SCALE", "ATTRIBUTION", "STOP", "DEFER"}}


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema") != "molgap-policy-v1":
        raise ValueError("unsupported policy schema")
    validate_id(policy.get("policy_id"), "policy_id")
    validate_id(policy.get("version"), "policy.version")
    kind = policy.get("policy_type")
    if kind not in POLICY_TYPES or policy.get("status") not in {"draft", "candidate", "approved", "retired"}:
        raise ValueError("invalid policy type/status")
    selector = policy.get("comparability_selector")
    if not isinstance(selector, dict) or not selector.get("scientific_contract"):
        raise ValueError("policy must select an explicit scientific contract")
    from .backtest import BASE_COMPARABILITY_FIELDS
    if set(selector) - {*BASE_COMPARABILITY_FIELDS, "architecture_identity", "reference_id"}:
        raise ValueError("unknown comparability selector field")
    if any(not isinstance(v, str) or not v for v in selector.values()):
        raise ValueError("invalid selector value")
    if not isinstance(policy.get("required_observable_fields"), list) or any(
        not isinstance(v, str) or not v for v in policy["required_observable_fields"]
    ):
        raise ValueError("required_observable_fields must be explicit")
    if len(set(policy["required_observable_fields"])) != len(policy["required_observable_fields"]):
        raise ValueError("duplicate required observable fields")
    if not SHA256_PATTERN.fullmatch(str(policy.get("created_from_source_digest", ""))):
        raise ValueError("policy source digest required")
    if not isinstance(policy.get("approval"), dict):
        raise ValueError("explicit approval metadata required")
    if policy["status"] == "approved" and any(not policy["approval"].get(k) for k in ("approved_by", "approved_at", "authority_ref")):
        raise ValueError("approved policy requires external approval authority")
    if not isinstance(policy.get("cost_model"), dict) or policy["cost_model"].get("kind") != "measured_only":
        raise ValueError("v1 replay supports measured_only cost model")
    if not isinstance(policy["cost_model"].get("assumptions"), list):
        raise ValueError("cost assumptions must be explicit")
    if kind in {"early_stop", "screening"}:
        point = policy.get("observation_point")
        if not isinstance(point, dict) or point.get("axis") not in {"optimizer_step", "sample_presentations"}:
            raise ValueError("prefix needs observed optimizer/sample axis")
        value = point.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("observation point must be nonnegative integer")
        if point.get("selection") != "exact":
            raise ValueError("v1 policies use exact observed decision points, no interpolation")
    elif policy.get("observation_point") is not None:
        raise ValueError("scale/research actions do not consume training prefixes")
    rule_fields = {"early_stop": "early_stop_rule", "screening": "promotion_rule",
                   "scale_up": "promotion_rule", "research_action": "action_rule"}
    for field in ("promotion_rule", "early_stop_rule"):
        if field not in policy:
            raise ValueError(f"missing explicit {field}")
    active = rule_fields[kind]
    for field in ("early_stop_rule", "promotion_rule", "action_rule"):
        rule = policy.get(field)
        if field != active:
            if rule is not None:
                raise ValueError("early-stop, scale-up and research rules must stay separate")
            continue
        if not isinstance(rule, dict) or set(rule) != {"field", "operator", "threshold", "action"}:
            raise ValueError("rule must explicitly bind field/operator/threshold/action")
        if rule["field"] not in policy["required_observable_fields"]:
            raise ValueError("rule field must be a required observable")
        if rule["operator"] not in {"lt", "le", "gt", "ge", "eq"}:
            raise ValueError("unsupported deterministic operator")
        threshold = rule["threshold"]
        if not isinstance(threshold, (str, int, float, bool)) or (
            isinstance(threshold, float) and not math.isfinite(threshold)
        ):
            raise ValueError("rule threshold must be a finite literal")
        if rule["operator"] != "eq" and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))):
            raise ValueError("ordered threshold must be numeric")
        expected = {"early_stop": "STOP", "screening": "PROMOTE", "scale_up": "SCALE"}
        if kind in expected and rule["action"] != expected[kind]:
            raise ValueError("rule action incompatible with policy type")
        if not isinstance(rule["action"], str) or not rule["action"]:
            raise ValueError("rule action must be nonempty text")
    if kind in ACTIONS and policy.get("borderline_action") not in ACTIONS[kind]:
        raise ValueError("invalid borderline action")
    if kind == "research_action" and (not isinstance(policy.get("borderline_action"), str) or not policy["borderline_action"]):
        raise ValueError("research action needs explicit fallback")
    semantics = policy.get("metric_semantics", {})
    if kind in {"early_stop", "screening"}:
        from .trace import METRICS, FIELDS
        if set(policy["required_observable_fields"]) - set(FIELDS):
            raise ValueError("prefix rules may only read canonical observation fields")
        for field in set(policy["required_observable_fields"]) & set(METRICS):
            if not isinstance(semantics.get(field), dict):
                raise ValueError("policy must bind exact metric semantics")
    return policy


def load_policy_registry(repo_root: str | Path) -> list[dict[str, Any]]:
    policies = []
    seen = set()
    for path in sorted((Path(repo_root) / "research_memory" / "policies").glob("*.json")):
        policy = validate_policy(load_json_object(path))
        key = policy["policy_id"], policy["version"]
        if key in seen:
            raise ValueError(f"duplicate policy version: {key}")
        seen.add(key)
        policies.append(policy)
    return sorted(policies, key=lambda p: (p["policy_id"], p["version"]))


def decide(policy: dict[str, Any], observable: dict[str, Any]) -> dict[str, Any]:
    """Receives only the decision-time view; no trajectory/terminal accessor."""
    missing = [field for field in policy["required_observable_fields"] if observable.get(field) is None]
    if missing:
        return {"action": None, "reason": "missing_observables", "missing": missing}
    kind = policy["policy_type"]
    rule = policy[{"early_stop": "early_stop_rule", "screening": "promotion_rule",
                   "scale_up": "promotion_rule", "research_action": "action_rule"}[kind]]
    value, threshold = observable[rule["field"]], rule["threshold"]
    if rule["operator"] != "eq" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return {"action": None, "reason": "observable_type_mismatch"}
    if isinstance(value, float) and not math.isfinite(value):
        return {"action": None, "reason": "nonfinite_observable"}
    import operator
    matched = {"lt": operator.lt, "le": operator.le, "gt": operator.gt,
               "ge": operator.ge, "eq": operator.eq}[rule["operator"]](value, threshold)
    return {"action": rule["action"] if matched else policy["borderline_action"],
            "reason": "rule_matched" if matched else "borderline"}

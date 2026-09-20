# Versioned policy registry

Place explicit policy records here as `<policy_id>.<version>.json`, following
`../schemas/policy-v1.schema.json`. No default thresholds or approved policies
are supplied. Scientific rule content and approval belong to external authority.

Types are screening, early_stop, scale_up, and research_action. Statuses are
draft, candidate, approved, and retired. Retired records are not backtested.
Each rule has field, operator (`lt`, `le`, `gt`, `ge`, `eq`), literal threshold,
and action. Set the unused promotion_rule/early_stop_rule to null; research_action
uses action_rule. Missing observables never invoke the borderline action.
Prefix metric rules must bind exact canonical metric_semantics. V1 cost_model
is measured_only with explicit assumptions; no hardware or scale extrapolation.

Create a new version when rule content changes. Preserve executed snapshots.
Only external explicit approval may set status=approved, with approved_by,
approved_at and authority_ref metadata. Neither replay nor rebuild writes here;
all generated reports keep policy_activated=false. A backtest is retrospective
evaluation, not out-of-sample validation of a policy selected using those runs.

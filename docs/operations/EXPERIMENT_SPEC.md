# Prospective experiment specification v1

`molgap.experiment_spec.ExperimentSpec(payload)` is the Stage 1 entry point.
`from_json`, `to_json`, `to_dict`, and `identity` provide strict loading,
canonical serialization, detached export, and the existing screen-policy
SHA-256 fingerprint convention. The synthetic fixture in
`tests/test_experiment_spec.py` is an exhaustive payload example.

The schema is `molgap-experiment-spec-v1`. The reserved terminal protocol is
`molgap-experiment-terminal-descriptor-v1`; it is a future descriptor identity,
not an implemented terminal API or a replacement for RML terminal wiring.

Each ordered arm binds its scientific role, family/version, base digest,
initial state digest/seed, dataset/split/membership/row-order/usage digests,
feature schema, recipe and objective/sampler/transform identities. An empty
addon list requires `addon_semantics: baseline`; nonempty lists require
`ordered`. Array order is significant and preserved; object key order is not.
Canonical strings are immutable snapshots, independent of caller mutations.

The static, immutable registries expose `FamilyContract` and `AddonContract`.
`spec.family_contract(arm_id)` resolves the future adapter lookup without
importing model code or dispatching a callable supplied by an author.
Family version `1` is an interface version, not a promoted model generation.
GPTrans-T binds the `pcqm_gptrans_v4` recipe and K1 binds `pcqm_k1_full`.
These names point to the existing package modules. No historical experiment
identity is copied. V1 recipe overrides must be exactly `{}`: these frozen
recipes have no approved override range. Seed 42 is required.
Future overrides need a reviewed recipe version with typed bounds.

Platform `device_count` is a positive-integer resource declaration, not a
per-arm recipe restriction. Stage 1 accepts `device_count: 2`, including a
physical run declaring two independent arms under V5, and does not cap the
declaration at one device. Booleans and non-integer values are rejected.
The future runner/platform layer must validate the runtime relationship among
arm count, GPU visibility, processes, and declared device count, including
dual-arm isolation. This schema implements no runner and grants no proof of
real dual-GPU execution or runtime acceptance.

The two parameter-free `gptrans_variants.py` addons are registered:
`pair_prenorm` and `centered_logits`. Both require empty config and a source
digest; they are mutually exclusive attention replacements and GPTrans-only.
K1 additionally registers `k1_pair_value/1` with empty config; its model-only
mapping and migration boundary are in [K1_ADAPTER.md](K1_ADAPTER.md).
Other addons and arbitrary addon configuration are rejected. New combinations
need explicit registry/code review and a contract version change.

All object fields are required and unknown fields are rejected recursively;
duplicate JSON keys are rejected too. There is no author-supplied readiness
field. Prospective requirements and evidence requirements are declarations,
never proof that those requirements were met. No spec grants protected-role
access, full-run admission, execution, scientific acceptance, or replay/desktop
readiness. Those remain under existing V5/RML authorities.

This layer checks declared identity syntax and compatibility only. Digests
must be supplied for base, initialization (including random initial state),
recipes, policy and data identities. It does not read or authenticate their
underlying bytes. Family adapters must verify those bindings against actual
source, recipe values, accepted graph manifests, role history and authority,
runtime certificates, and source bundles. A structurally valid addon spec is
not executable by the frozen reference adapter, which rejects variants;
future adapter work must explicitly handle this boundary.

Next stages own family adapters, packaging, graph preflight, runners and the
terminal descriptor translation into existing `close_terminal_multi_arm`.
They must preserve the spec identity and cannot derive readiness from this
validator. This module does not modify any scientific or RML gate.

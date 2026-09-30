# Clean-input chemical auxiliary supervision

Owner: desktop, branch `codex/exp/gptrans-chemical-aux`.

Question: can chemical auxiliary supervision improve terminal Gap accuracy
while preserving the exported GPTrans inference graph?

Read [STATUS.md](STATUS.md) for submission routing and remaining gates,
[training_protocol.md](training_protocol.md) for the frozen independent
descriptor/fingerprint pair, and `experiment_spec_v3.json` for machine identity.
Per-arm prospective RML records live in `training_prospective_v3/`; submission
and release observations live in `submission_v3/`. Scientific acceptance is pending.

The original CPU/static feasibility is preserved in `protocol.md`,
`feasibility.md` and `prospective/`. The failed strict-cache attempt is preserved
in `cache_decision.md` and `cache_prospective/rml_finalized/`. The explicit
repair policy lives in `label_policy.md`; accepted full component caches and
their CPU-only closure live in `cache_retry_observation/` and
`cache_retry_prospective/rml_finalized/`. These are separate evidence milestones.

`pair_contract_draft.md` is a superseded unexecuted design. It is not the
submitted recipe. Implementation notes remain in `implementation_status.md`;
the dated repair check is in `repair_verification.md`.

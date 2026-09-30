# Clean-input chemical auxiliary supervision feasibility

Owner: desktop, branch `codex/exp/gptrans-chemical-aux`.

Question: can graph-level chemical supervision improve terminal Gap accuracy
without changing the exported GPTrans inference graph and with <=5% added
training wall time? This is a new, unproven adaptation, not a reproduction claim.

Read `protocol.md` for the bounded local action, `pair_contract_draft.md` for
the conditional two-arm design, and `feasibility.md` for observed checks.
See `implementation_status.md` for the later integration and remaining input gates.
The attempted full train-label cache has a confirmed input blocker; see
`cache_decision.md` and the finalized `cache_prospective/rml_finalized/` records.
No Kaggle training submission followed that failed gate.
See `label_policy.md` for the subsequent explicit adapter repair. It does not
qualify a new full cache or alter the failed strict-attempt evidence.
The `prospective/` record is for CPU/static feasibility only. It is not either
training arm. Future training needs its own Spec v2 and per-arm plans.

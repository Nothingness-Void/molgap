# Strict production closure, 2026-10-24 JST

Kaggle1 kernel140391249/version1; frozen source9dd7145c, package
08d420753b6b2b2fe4fddf16b27722b802453dabcb04dad120b73e03f5dab8fa.
The source was not changed remotely. Shared post-training acceptance repair
1ef378ff was reviewed and cherry-picked as cd934440 on the experiment owner.

| Check | Observed result |
|---|---|
| Exact version/source/private T4/data mount | VERIFIED |
| Both40-pass traces,31240 updates,3998720 presentations | VERIFIED |
| Selected Live weights, finite aligned50000 predictions and target identity | VERIFIED |
| Original repeated/serialized-resume native proofs, independently per arm | VERIFIED |
| Observed role audits and measured allocation retention | VERIFIED |
| Learned versus fixed strict comparison | STRICT_CAUSAL |
| Two canonical replay memberships | complete, no exclusions |
| Ordinary repeated strict entry | CANONICAL_READ_ONLY; executed=false |
| Repeated entry elapsed / experiment mutation |11.719seconds /112 files unchanged |
| RML check --frozen | passed |

The [scientific decision](terminal_decision.md) is independent: fixed CLOSED,
learned NEGATIVE_UNDER_CONTRACT. No READY, adoption or transfer claim is made.
Canonical finalizations are under
`runs/pcqm-k1-message-norm-100k-s42/prospective/{msgnorm_fixed,msgnorm_learned}/rml_finalized/`.
The [compact qualification](terminal_acceptance/qualification_summary.json)
and [closure receipt](terminal_acceptance/closure_receipt.json) bind the result.

The first closure exposed an identical target-transform asset retained at two
paths; [incident](terminal_acceptance/incident_asset_reuse.md) preserves the
failed report and repair handoff. The shared fix admits identical content,
retains conflict rejection and distinguishes attempted execution from completed
closure. The repair conversation exercised the affected existing workflow.
Here verification used the actual complete pair: repaired closure88.131seconds,
then the ordinary read-only entry above. No additional synthetic test suite,
training, inference or remote retry was run during this acceptance.

Repaired closure called the existing `accept_run_workflow` with explicit reviewed
per-arm outcomes, observed role audits, strict_inputs={}, original run/attempt,
execution retention and physical job cost bindings. This supported callable
route was needed for different arm outcomes; no custom validator was introduced.
Read-only replay through the ordinary CLI:

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' platforms/local/accept_run.py --profile D:/molgap-machine-profile.json --run-id pcqm-k1-message-norm-100k-s42 --attempt-id v1 --strict
```

This command requires the owning source/catalog checkout and original external
run package/retrieval mapping. Desktop retains canonical evidence; restore the
archived owner for runtime-specific reinspection. Detailed reports and original
failure/success logs remain under
`D:/molgap-artifacts/reconciliation/k1-message-norm-100k/`.

The [retrieval inventory](terminal_acceptance/retrieval_inventory.json) records
36 retained files totaling593.895MiB, exclusively on D:. Required selected/final
artifacts, native repeat/resume proofs and small execution metadata were retained;
no full output download, historical resume chunks or dependency environment.

## General replay grouping boundary

The strict same-run comparison binds full recipe/data/runtime/weight identities.
An additional review found that generic derived replay comparability fields
still use several `unspecified_in_contract` values and `ema_semantics=none`.
That generic grouping is not used here to claim cross-experiment equivalence.
The compact `replay_identity_review.json` in the external reconciliation root
and the full strict comparison were handed to the authorized infrastructure
conversation for a separate shared-generator audit. The complete per-arm
retention qualification is not blanket permission to pool unrelated experiments.

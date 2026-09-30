# Repaired component-cache release, 2026-09-30

The user authorized resubmitting GPTrans chemical auxiliary supervision after
the adapter repair at d9fe9820. Reuse the accepted train-only export pinned in
`cache_observation/export_receipt_attempt2.json` and the policy in
`label_policy.md`. This diagnostic precedes accelerator publication.

Build independent descriptor-only and fingerprint-only caches for all 100,000
frozen V4 training rows. Both use the existing PCQM topology parser. Descriptors
use explicit raw/CDF failure masking; fingerprints require no descriptor labels.
Preserve exact source membership and order. Check known exceptional rows first
and stop immediately on a failed required label, retaining the failure manifest.
No graph reconstruction, geometry generation or protected target read is needed.

The standing two-new-direction instruction selects descriptor MSE weight 0.1
and fingerprint BCE weight 0.1 as separate training arms; auxiliary width32 and
seed42 remain fixed. The historical baseline-plus-joint proposal in
`pair_contract_draft.md` remains an unexecuted design. Reuse the accepted
GPTrans-T 100K V4 reference; no baseline training is authorized by this retry.

CPU wall time is measured for this preparation; CPU busy time is unknown unless
observed. No accelerator cost applies to local labels. Complete component-cache
acceptance permits freezing the two training-arm plans. It does not establish
accuracy, runtime qualification, overhead or replay readiness. Any cache failure
blocks the GPU launch and identifies the next missing discriminator.

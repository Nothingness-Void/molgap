# First production closure: target-transform asset collision

The first strict finalization on2026-10-23 JST used retained real Kaggle1
kernel140391249/version1 outputs. No new training, inference or synthetic
evidence was used. Both mechanical and native proof checks passed; the same-run
candidate comparison was STRICT_CAUSAL with no comparison blockers.

The fixed reference finalization failed because target_transform_asset_id
`b3950836b7fded7445e3b7425f5059570ef028e639c0c9e0c36cfa347a2cb04e`
was published again under this question's comparison_assets while the same
asset already existed under pcqm_k1_norm_mechanisms_100k/comparison_assets.
The inspected records have identical content and asset SHA
`8f1048d1479b50e514d3d844f11d7ab169eac839abe796b6e7e76e3baae8ba3f`.
The candidate consequently failed because its same-run reference was not
terminally accepted. No finalization.json was created for either arm.

The top-level result reported BLOCKED with executed=true; nested results
reported FINALIZATION_FAILED. The factual result is an attempted but failed
closure, never replay readiness. Do not delete historical assets, change their
identity, or relax qualification to bypass this failure.

Evidence and reproduction are in
`D:/molgap-artifacts/reconciliation/k1-message-norm-100k/`:
`strict_acceptance.json`, `close.log`, `close.py`, and the exact retained outputs.
The production call took36.416seconds, including full diagnosis/qualification.
Human-authorized repair routing was sent to conversation
`01a14421-e9d1-73a1-9059-3723d77d0f02` (重构基建). Resolution and final evidence
qualification are recorded separately; the original failure report is retained.

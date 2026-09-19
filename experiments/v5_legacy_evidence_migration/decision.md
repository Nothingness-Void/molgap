# V5 legacy evidence migration decision

Decision date: 2026-09-17

## Question

Are the historical V4 screening artifacts still available, and can their
accepted results be represented under V5?

## Local artifact audit

The local record store contains 21 arms whose embedded benchmark is
`pcqm4mv2-ogb-fixed-100k-gap-v4`. All 21 retain the prediction payload, selected
model, last checkpoint, runtime certificate, trace, and completion manifest.
No V4 100K arm in this inventory is reduced to a scalar-only result. The compact
count record is `inventory_summary.json`; large files remain in ignored
`platforms/_records` storage.

The old K1 500K payloads, models, and checkpoints also remain available. That
run used the historical paired-v3 partial-tail contract, so its later audit
correctly marks it ineligible as a reusable strict V4 reference. V5 does not
change that scientific incompatibility.

At audit time, PairToken matched60-v4 500K job `122312462` had durable candidate
checkpoints and trace files. The matching K1 reference was
available locally only as a frozen scalar; its prediction bundle and complete
matched60-v4 reference contract were not present. V5 terminal acceptance was
therefore constrained to leave `comparison_status=PENDING` unless that exact
reference evidence was retrieved. A scalar MAE comparison cannot be promoted
to a strict paired V5 comparison.

## Migration rule

Historical files are never rewritten in place. A V5 migration is a sidecar
that:

1. names the original V4 acceptance and scientific contract;
2. verifies retained prediction alignment and artifact hashes without model
   inference;
3. separates execution, artifact, comparison, scientific, transfer, budget,
   and full-handoff status;
4. records incomplete historical cost scope and reused-role limitations; and
5. remains `PENDING` wherever required evidence cannot be recovered.

The first migrated result is the positive PairToken 100K comparison in
`../pcqm_k1_pair_token_100k/results/v5_evidence_overlay.json`. It accepts the
strict V4 comparison evidence under the V5 state model but does not create a
new experiment, independent holdout, scale result, or desktop handoff.

## K1-v4 reference recovery

The retained K1-v4 100K reference was subsequently recovered into a reusable
repository-bound V5 bundle without training or model inference. The recovery
verified the fixed-cache shard hashes and row order, recomputed the original
100K sample target transform, bound the aligned 50K prediction payload, and
verified the retained model, checkpoint, runtime certificate, role events,
cost record and trace. The compact authority is
`k1_v4_100k_reference/reference_bundle.json`; large tensors remain in ignored
platform records. Training stochasticity remains explicitly unavailable and
was not inferred from platform or account identity.

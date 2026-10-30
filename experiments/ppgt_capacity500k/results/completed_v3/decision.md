# Width-capacity endpoint — 2026-10-31

The exact Kaggle2 kernel `kaseichou/molgap-ppgt-capacity500k-s42-r2`,
ID141694880/version1, completed both authorized arms. Saved-artifact analysis
found a strongly negative width result under the frozen recipe. Strict RML
qualification remained blocked; this record did not claim Replay admission.

## Endpoint and trajectory

The existing directory diagnosis verified identical50,000 source IDs and target
values. Both arms completed15,620 optimizer updates,1,999,360 presentations and
20 development evaluations; both selected their final Live checkpoint. See
[paired analysis](paired_diagnosis/analysis_summary.json) and
[runtime summary](runtime_summary.json).

| Arm | Factory-verified parameters | Best/final Common-Dev MAE, eV | Final online train MAE, normalized | Process seconds |
|---|---:|---:|---:|---:|
|Width192|2,577,729|0.1714706253|0.1270860234|18,781.758|
|Width384|9,933,825|0.3680308035|0.2859581091|19,089.695|

Prediction recomputation used float64 accumulation; native trace/model scalars
differed by less than2e-8eV. Parameter-shaped selected-state tensor counts also
equalled the independently factory-verified parameter counts.

Candidate minus reference MAE was+0.1965601782eV; the2,000-draw paired row
bootstrap95% interval was[+0.1939050830,+0.1993640090]. The candidate was worse
at all20 matched observations. Its last-five-observation mean disadvantage was
0.1978542000eV; only25.324% of rows improved. No preregistered0.005eV capacity
lead condition passed. Row bootstrap did not measure training-seed uncertainty.

## Attribution supported by the saved evidence

- Larger width failed to improve training fit as well as development error.
  Its normalized online training error worsened during warmup, while its
  development MAE rose from0.395523 at781 updates to0.765903 at3,905 updates,
  around the1e-3 learning-rate peak. The small arm improved through that region.
  This supported an optimization/parameterization mismatch hypothesis, not a
  claim that excess representational capacity caused ordinary overfitting.
  Gradient/activation witnesses sufficient to identify a particular culprit
  were not retained in the endpoint diagnosis.
- Both models still improved late: the final five evaluation intervals reduced
  development MAE by0.0100262 and0.0171136eV respectively. Their final single
  interval gains were0.0003744 and0.0003830eV at an LR floor of1e-6. This did
  not establish either family's ultimate ceiling; it also did not justify
  extending this floor schedule to erase a0.19656eV gap.
- Every reported target-value stratum regressed. Optimal post-hoc constant
  shifting removed only0.0247% of the disadvantage, so a simple output offset
  did not explain the failure. These label-based strata were diagnostic and
  were not an inference-time routing rule.
- Width multiplied parameters by3.8537 but measured process time rose only
  1.64%. These end-to-end times included evaluation and IO. The retained
  two-update native profile measured allocated-memory peaks of547.32 and
  881.50MiB; these were not full-training peaks or exclusive training/GPU
  throughput measurements. Structural atom/ring/degree grouping was unavailable
  in the retrieved endpoint payloads.

## Mechanical acceptance and evidence boundary

The registered exact-version all-arm retriever retained model/checkpoint,
selected/final predictions, contracts, canonical traces, runtime certificates,
native repeat/resume proofs and the execution-retention ledger. Both workers
exited0 and both arms passed `MECHANICALLY_VERIFIED` through the existing
`accept_run.py --strict --diagnose` entry.

That same strict entry returned `BLOCKED` for both arms with the exact missing
key `'reference_artifacts'`, reporting required evidence
`target_transform_asset`. The separately generated role drafts also preserved
protected-role states as `unknown`: native role audits declared `untouched`,
but the retriever did not retain a file-open provenance artifact that would
mechanically establish that declaration. No declaration was upgraded to proof.

The descriptive directory diagnosis was an existing supported fallback; its
output explicitly said scientific acceptance and Replay readiness were not
evaluated. It did not replace the blocked registered strict gate. No frozen
reference, role event, evidence envelope or implementation was rewritten.

Raw artifacts remained under
`E:/MolGap/artifacts/retrieved/ppgt-capacity500k-k2-s42-v1/v3/version1/`;
the original receipt and retrieval mappings remained under
`E:/MolGap/artifacts/runs/ppgt-capacity500k-k2-s42-v1/v3/operations/`.
Compact original-byte manifests, traces, native reports and role audits were
retained in [retained/](retained/). RML terminal closure and Replay admission
were pending, with the cause preserved in [runtime_summary.json](runtime_summary.json).

## Allocation and disposition

The allocation ledger measured19,507.063 session seconds (5.41863h),
39,014.125 physical T4-device seconds (10.83726h) for two allocated devices.
Adding the previously reserved120-second failed-attempt debit gave an observed
aggregate19,627.063s, below the32,400s authorization ceiling. The native retry
ledger alone did not include that earlier attempt. Queue, pre-Python provisioning
and platform-billed hours were not measured; the ledger was not a billing claim.

The decision was to stop width-only expansion under this recipe and retain
the negative endpoint with its qualification blocker. Neither a longer run,
extra seed, new module, official role nor successor was submitted. A future
research proposal would have to address width-dependent optimization with
observable diagnostics, rather than assume that more parameters or more steps
would rescue this endpoint. Superiority or inferiority to K1/GPTrans was not
established by this within-family pair.

The acceptance session performed saved-tensor/JSON analysis only: no training,
model inference or protected dataset access.

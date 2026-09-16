# Status

Stage 1 is authorized for the new Kunshan account. It reuses the accepted
fixed 100K/50K graph identity and 20 already-frozen same-contract K1 prediction
payloads. It performs no model inference or training and reads no sealed role.

Stage 2 and any successor architecture remain conditional on the Stage-1
decision required by `protocol.md`.

The first Stage-1 submission (`122243216`) failed before reading a payload:
the Kunshan CPU node could not import the vendor DTK PyTorch wheel because
`libhsakmt.so.1` is available only with a DCU allocation. The unchanged
analysis is therefore resubmitted on `kshdtest`; the accelerator is required
only to load PyTorch serialization and no model or training step is executed.

That DCU submission (`122249373`) reached the accepted runtime but failed while
deserializing the graph cache because the uploaded minimal source bundle omitted
the cache's recorded `molgap.pcqm_wedge.WedgeData` class. No prediction payload
or scientific result was read. The next infrastructure-only retry keeps the
analysis contract unchanged, ships the complete tracked `src/molgap/` tree,
and requires a graph/payload preflight in the same job before Stage 1 begins.

Infrastructure retry `122250328` completed and passed mechanical acceptance.
The paired map found coherent deficits at small/sparse/weakly conjugated and
highly cyclic/high-RWSE topology extremes, while no existing variant cleared
the overall material-gain gate. Stage 2 is informative but remains a
no-training frozen-checkpoint causal audit; no architecture successor has been
released. The complete interpretation is in `results/stage1_decision.md`.

The user authorized a bounded three-round causal sequence. Round 1 is frozen
checkpoint job `122253291`, first observed running on Kunshan node `e06r4n09`.
It performs layer-3/6/9 exchange interventions and internal-state measurement
without training. Rounds 2 and 3 remain conditional on accepted causal evidence.

Round-1 job `122253291` failed during diagnostic aggregation because CUDA
`bincount` has no deterministic implementation under the frozen deterministic
runtime. It produced no scientific artifact and did not train a model. The
instrumentation now moves descriptor and diagnostic reductions to CPU while
leaving frozen-checkpoint inference on the accelerator. The unchanged audit was
submitted once as job `122268096` from source commit `a18dc2a`; it was observed
running on node `e10r4n11`. No Round-2 or Round-3 candidate is released until
this retry passes mechanical acceptance and receives a scientific decision.

The unchanged retry `122268096` completed and passed every mechanical check.
Round 1 found that all three exchanges are necessary, but layer 6 has the
strongest topology-dependent update-magnitude and node-dispersion inflation.
Round 2 is released as a frozen-checkpoint layer-6 residual-strength audit over
coefficients `0.50, 0.75, 1.00, 1.25, 1.50`. Round 3 remains conditional.

Round 2 was submitted once to Kunshan as job `122270044` from immutable source
commit `7b1a4ea` and minimal source archive SHA-256
`528883b21a25c37e09c6d22634062d9d0a666023ff5fbcd72486dea3453eb2ee`.
It was first observed pending for priority. No other job for this account was
queued or running at submission time.

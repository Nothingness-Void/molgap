# Engineering feasibility observations, 2026-09-30

Disposition: CPU_LABEL_INTERFACE_VERIFIED_ONLY; scientific question remains
ACTIVE. No training, model forward/backward, PCQM row, development role,
checkpoint inference, Kaggle submission or protected-role access occurred.

The synthetic CPU fixture suite verifies 200 finite normalized descriptors and
512 path fingerprint bits against the installed upstream libraries for seven
valid SMILES spanning aromatic, charged, disconnected, chiral and small inputs.
Invalid/missing SMILES, descriptor failure and nonfinite descriptor results
retain explicit source identity and reason codes. No full-cache coverage claim
is made; no extrapolation of fixture timing to 100K is supported.

The reusable objective configuration and existing optimizer/checkpoint interfaces
were extended after the user asked to parameterize the wheel. See
`docs/operations/GPTRANS_OBJECTIVE_PARAMETERS.md` for implementation coverage
and the exact remaining experiment-adapter work. The frozen initial
`pair_contract_draft.md` remains the pre-check plan; its stated loss-adapter
gap is addressed by this later implementation, not by rewriting its snapshot.

Default auxiliary weights are zero. The candidate JSON uses width32 and
0.1/0.1 auxiliary weights. Scalar/tensor fixtures check exact default L1 and
gradient, separate Gap/total metrics, weighted auxiliary gradients, bad target
rejection, configuration identity, checkpoint rejection and export-key filtering.
These do not establish full-model equivalence. The remote CUDA test is retained
but skipped locally; actual T4 throughput and inference equivalence are unknown.

Verification: 15 label tests and 20 objective/metadata/optimizer-step tests
passed. Five existing non-model GPTrans V4 checks passed; the existing full
model backward test was deselected under the local execution boundary. One
new remote GPU test was skipped. RML validation passed with 62 trajectories;
this count includes the new feasibility record, not a trained candidate.

Corrected static head size: 288->32->712, 32,744 parameters, approximately
0.624% of GPTrans-T. MAC count and parameter count are not wall-time measures.
Ordinary forward contains no auxiliary call; actual exported equivalence remains
an explicit GPU check. Head/optimizer/EMA memory is not counted as zero.

Next bounded action: finish the authenticated train-only label-cache adapter
and per-arm Spec binding, then run the frozen remote GPU engineering preflight
before deciding whether a 100K dual-arm screen is worthwhile. The active
geometry run is independent and untouched. This preparation is not training-ready
or replay-ready and makes no model-promotion claim.

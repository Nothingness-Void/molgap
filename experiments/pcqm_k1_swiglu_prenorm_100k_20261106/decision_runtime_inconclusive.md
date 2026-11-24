# Terminal decision: INCONCLUSIVE — 2026-11-06 JST

This is the authoritative terminal disposition. Complete training and endpoint
inspection passed; strict comparison qualification failed. Close INCONCLUSIVE
through the existing explicitly non-strict terminal path,with the failure
preserved. No adoption,scale-up,baseline retraining or successor is authorized.

Kaggle3 job143019649/version1 completed40passes,31240updates and3998720
presentations. The best/final Live checkpoint is pass40; aligned finite50000-row
prediction MAE is0.1420404841130972eV. Retained original K1 endpoint is
0.1412944608205557eV,paired endpoint delta+0.746023meV with2000-bootstrap95percent
interval[-0.326416,+1.783182]meV. The observed numerical endpoint does not meet
the planned3meV improvement gate. This NOT_STRICT comparison does not establish
a causal architecture effect,significant harm,or training-seed uncertainty.

Strict accept_run failed on SCIENTIFIC_CONTRACT_MISMATCH. The unplanned mismatch
is runtime_certificate_scope: candidate
`c75011519274a52cd40b1610d8b967276fc8fcad5887ab4ca2b68969533ba539`
versus reference
`a2c521194604d5a98617fb7480271aabca910407a2e30ef494758e44aa769a37`.
Architecture identity differs as prospectively declared; runtime scope does not.
Both observed records use Torch2.11.0,T4,FP32,native AdamW. However,the scope
fingerprints the whole runtime environment,and the full payload was not retained.
There is no evidence identifying the specific difference or proving it harmless.
Prelaunch copied the reference's expected scope; that was a plan,not an observation.
Native repeat/resume certificates separately accepted each model,not cross-runtime
equivalence. Do not replace scopes,declare a retrospective intervention,modify
the frozen gate or treat omitted --strict as strict acceptance.

The failed strict report remains at
`runs/pcqm-k1-swiglu-prenorm-100k-20261106-s42/prospective/strict_qualification/458d7b18e893a6484de2/swiglu_prenorm/comparison_readiness.json`.
The earlier decision.md was a numerical-gate draft written before this strict
result; it was never finalized. It is preserved with that failed staging's
hash bindings and is superseded by this decision,not an accepted negative result.

Descriptive attribution remains in attribution.md: stable complete training,
early endpoint advantage fading by the end,and no supported unique underfit,
overfit or exposure-shortage cause. It cannot close the causal evidence gap.
The prior unnormalized run remains separately INCONCLUSIVE. No additional model
execution occurred during acceptance; only retained artifacts were inspected.

Measured process4926.980339015s,allocation5234.597692252s,
10469.195384504allocated T4-device-seconds=2.908109829T4-device-hours,including
the idle second device. Queue/provisioning-before-Python is unknown. The observed
role audit found no protected-role file opens in its recorded Python scope.

Preserve exact source/Spec/recipe,selected model,predictions,full trace,resume,
native proofs,cost and role evidence. Record strict/replay exclusions honestly
in RML; this non-strict closure does not release a strict replay-ready claim.

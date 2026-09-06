# Audit Preparation (2026-09-06)

Local source review found that the official-full EdgeState configuration
defaulted to dropout 0.05 while the geometry factory fixed dropout 0.10. The
actual checkpoint setting required scheduled verification. The historical
warm-start used FP16, whereas the accepted 100K screen used FP32.

These are potential confounders, not established causes of the observed MAE
regression. The prior single-batch initialization test was insufficient to
exclude every input/evaluation discrepancy. The authorized P0 audit was
prepared to resolve these uncertainties before any scratch successor.

No new accuracy outcome was available at preparation. Historical negative
evidence remains in `../pcqm_geometry_warmstart/eight_epoch_result.json`.

## Completed P0 Audit (2026-09-06)

IMS job 1448851 completed the full 73,545-row replay. Source and mapped
candidate FP32 MAEs were 0.09969422899247901 and 0.0996942296343565 eV;
the largest paired difference was 2.86102294921875e-6 eV. Candidate FP16 MAE
was 0.09963855845687164 eV, reproducing the retained source performance.
This supports preservation of the initial evaluation function on the complete
accepted validation population.

The actual source dropout was 0.05; the candidate used 0.10. Four fixed
training-batch FP32 numerical probes had finite gradients and updates. Three
FP16 probes overflowed at the default initial GradScaler scale. Such initial
overflows may be resolved by scaler backoff; these probes do not prove
persistent FP16 failure or explain the previous training regression. Train-
mode dropout also prevents interpreting gradient differences as pure rounding
error. The audit supports a fixed FP32 scratch control, not another tuning
round on the official validation set.

The local paired-forward test caught a NameError in the later no-op atom-block
hook (`layer` was undefined in the pure Triangle loop). Enumerating the loop
repaired that interface without changing the accepted geometry mechanism.
The original architecture/evidence commits were verified as e083bee, bfd8279,
and ba320fa. No new server-side discovery module was selected.

Downloaded audit parts and recomputed arithmetic are summarized by
`audit_acceptance.json`; launch identities are recorded in `launch.json`.

## Superseded Scratch Proposal (2026-09-06)

The paired scratch preflight job 1448871 stopped at the initial-function
comparison, before throughput measurement or training. Its log did not print
the measured difference, so no causal diagnosis was inferred from that
assertion. Neither arm started and no accuracy result was produced.

The user then supplied the newer, three-seed-accepted GraphState9 handoff.
Desktop verified that evidence on the server branch and superseded the
two-arm proposal with `../pcqm_graph_state_full/`. No repeat of this older
preflight or training was submitted. P0 audit evidence was retained unchanged.

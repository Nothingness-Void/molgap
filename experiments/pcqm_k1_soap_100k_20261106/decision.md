# SOAP100K: positive below the frozen gate — 2026-11-07 JST

Scientific outcome: SOAP `POSITIVE_BELOW_GATE`; same-job Muon reference `CLOSED`.
Do not promote this fixed SOAP recipe to500K. Its selected improvement is real
on the retained development rows but smaller than the preregistered3meV gate.
Runtime/replay qualification belongs to the existing terminal closure report;
this scientific decision does not substitute for that verification.

| Live development metric | Muon | SOAP |
|---|---:|---:|
| Selected MAE,eV, recomputed from retained predictions | 0.132520667662 | 0.131572498874 |
| Selected pass / optimizer step | 40 / 31240 | 37 / 28897 |
| Final MAE,eV, trace | 0.132520660758 | 0.131823241711 |
| Completed passes / optimizer calls | 40 / 31240 | 40 / 31240 |
| Sample presentations | 3998720 | 3998720 |
| Measured process wall,seconds | 11480.861596 | 11477.607365 |

SOAP-minus-Muon selected delta is -0.948169meV. The2000-resample paired-row95%
interval is[-1.656169,-0.256079]meV over50000 aligned finite rows. Final delta
is -0.697419meV; last-five matched-pass mean is -1.167342meV. All directional
criteria favor SOAP, but the selected delta fails the3meV material criterion.
The row interval describes this retained cohort,not seed-to-seed variation.
Small trace/recomputed differences are reduction precision,not new evaluations.

Process-wall ratio is0.9997166: no meaningful training-speed advantage is measured.
Shared allocation was11826.746272 wall seconds and23653.492545 allocated T4
device-seconds, counted once. Queue/provisioning before Python remain unknown.
SOAP's extra optimizer-state storage is an actual cost even with similar runtime.

## Attribution

This fixed SOAP recipe starts behind Muon: mean deltas at passes1–10 and11–20
are+3.624292 and+1.673865meV. The final10-pass mean reverses to-0.705154meV;
20of40 matched evaluations favor SOAP. The observed shape is late modest
recovery,not faster initial optimization or an early gain that vanishes.
Both completed their frozen horizon; an execution/exposure shortfall does not
explain this gate miss. SOAP also has higher normalized online train error at
the end. Online train averages use changing weights and training mode; they
cannot establish underfitting or a fixed-weight generalization gap.

Post-hoc median residual shifts reduce the selected delta to-0.318347meV,
versus-0.948169raw; calibration contributes to the observed improvement but
does not explain all of it. This same-role fitted shift is descriptive only.
The diagnostic equal-weight two-model blend scores0.128274618eV, better than
either single model. That is retained-prediction complementarity evidence,
not an accepted standalone model,an independent validation,or an automatic
two-model500K authorization. The fitted0.53 blend weight is not adopted.

No isolated covariance mechanism, learning-rate mismatch, or remaining-exposure
benefit can be identified from this one fixed recipe/seed. The author algorithm
changes moments/basis behavior as declared,including one fewer hidden-matrix
update; no extra exposure was added retrospectively. Retain the result and
avoid an unregistered LR/frequency grid or SOAP-Muon hybrid.

## Evidence and closure

Producer143051786/version1; source/Spec/package identities remain in STATUS.md.
Both native workers and shared execution report reached complete. Standard
strict retrieval and diagnosis mechanically verified40-pass traces, counters,
model/checkpoint and aligned50000-row predictions. Observed data-open provenance
feeds the existing protected-role audit; official validation and test roles
are not released by this decision. Per-arm native proof and replay results,
role audit and canonical terminal records are produced by accept_run, not here.

Diagnostic: `D:/molgap-artifacts/runs/pcqm-k1-soap-100k-20261106-s42/v1/reports/diagnostic_fdc1d7c421c016bdd1ec3dd8d2ba56be016b854b3cb2bb0300655b2bf5f3d489.json`.
Artifacts: `D:/molgap-artifacts/retrieved/pcqm-k1-soap-100k-20261106-s42/v1/version1`.
No new training,test-role access or baseline substitution was performed.

# EC-informed gate100K: negative under contract — 2026-11-07 JST

Scientific outcome: expansion_gate `NEGATIVE_UNDER_CONTRACT`; same-job
degree_gate reference `CLOSED`. Close this exact early-layer EC residual recipe.
It does not outperform the matched degree-guided gate. No automatic500K,
new seed,or layer/strength search follows. Native proof,strict comparison and
replay qualification belong to the existing terminal closure report.

| Live development metric | Degree guide | EC guide |
|---|---:|---:|
| Selected MAE,eV, recomputed retained rows | 0.131312467455 | 0.131868945331 |
| Selected pass / optimizer step | 40 / 31240 | 36 / 28116 |
| Final MAE,eV, trace | 0.131312474608 | 0.131969302893 |
| Completed passes / optimizer calls | 40 / 31240 | 40 / 31240 |
| Sample presentations | 3998720 | 3998720 |
| Measured process wall,seconds | 13139.132112 | 13183.035968 |

EC-minus-degree selected delta is+0.556478meV;2000-resample paired-row95%
interval[-0.023800,+1.144367]meV on50000 aligned finite development rows.
This does not establish statistically resolved harm; it clearly fails the
frozen improvement<=-3meV criterion. Final delta is+0.656828meV and the
last-five matched-pass mean is+0.473607meV,also opposite to the frozen gate.
Row uncertainty is conditional on this cohort,not training-seed variation.

Process wall ratio is1.0033415,EC+43.903856seconds. The shared allocation ledger
records13511.651167wall seconds and27023.302335allocated T4-device-seconds,
counted once. Queue/provisioning before Python remain unknown. Both arms
compute identical topology statistics; this isolates guidance within this
recipe and does not measure cost against an unmodified K1 worker.

## Attribution

The preregistered CPU screen established informative EC variation and low
topology cost; the real-topology gradient probe established parameter
connectivity. Complete40-pass traces and equal exposure contradict a failed
launch,missing training,or an inactive third-layer path as the gate-miss
explanation. Informativeness/connectivity were necessary checks,not proof
that EC guidance would improve chemical Gap prediction.

Only11of40 matched evaluations favor EC. Mean deltas at passes1–10,11–20,
and the final10 are+1.387827,+0.879638,and+0.382617meV. EC gradually reduces
its deficit but does not obtain a favorable terminal or last-five endpoint.
The final online train MAEs are nearly equal:0.076285versus0.076222eV.
These are in-pass averages with changing weights,not fixed-weight training
evaluations; they do not diagnose underfitting,overfitting or an exposure
bottleneck. No claim that all expansion-aware methods fail follows.

Post-hoc median-shift correction leaves a+0.405540meV deficit,so calibration
alone does not reverse the ranking. Equal-weight retained-prediction blending
scores0.129116487eV; this descriptive same-role complementarity is not an
accepted ensemble or authorization to scale two models. No fitted blend is
adopted. Historical Muon scores are contextual: this experiment has no fresh
unmodified Muon arm,so the degree arm's lower point MAE does not establish
strict causal improvement over original K1/Muon.

The tested change was bounded residual guidance at layers2/3,not literal
EC-Gate paper reproduction. The result supports retaining the cheaper-to-reason
degree guidance over this EC variant within their matched pair,not adopting
either addon into the current recommendation. Exact causal failure mode is
insufficient_evidence; no new training is added to fill that attribution gap.

## Evidence and closure

Producer143052037/version1; source/Spec/package remain in STATUS.md.
Strict selective retrieval and diagnosis mechanically verified complete
trace/exposure,selected/checkpoint state and aligned50000-row finite predictions.
Observed data-open provenance feeds the existing role audit. Official validation
and test roles remain sealed; the acceptance owner verifies per-arm native
proof,replay,cost and role records rather than this prose assigning them.

Diagnostic: `D:/molgap-artifacts/runs/pcqm-k1-ec-gate-100k-20261106-s42/v1/reports/diagnostic_76214434a23ec90281c45cf22fff7d1d209823a89104b0f72765ba5e12667fad.json`.
Artifacts: `D:/molgap-artifacts/retrieved/pcqm-k1-ec-gate-100k-20261106-s42/v1/version1`.
No shared trainer/verifier was changed and no successor was submitted.

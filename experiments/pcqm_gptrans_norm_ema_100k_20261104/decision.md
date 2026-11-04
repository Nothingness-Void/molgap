# Decision: normalization exemption rejected at 100K

2026-11-04 JST. Kaggle1 job142547457/version1; run
pcqm-gptrans-norm-ema-100k-20261104-s42, attempt v1.

uniform_wd_multi_ema closes as CLOSED (same-job reference).
norm_no_decay_multi_ema closes as NEGATIVE_UNDER_CONTRACT.
No adoption, further WD grid, retry or500K release follows.

Primary EMA0.999 development MAE: uniform0.139440192079eV(pass42),
LayerNorm-affine zero decay0.141208115811eV(pass47).
Candidate-minus-reference:+1.767924meV; paired-row95%CI
[+0.849500,+2.673038]meV,2000resamples,seed42,50000aligned rows.
The frozen3meV improvement gate and negative upper CI are not met.

Predeclared descriptive EMA0.9999-minus-0.999 comparisons regress:
uniform+3.926531meV,CI[+3.549389,+4.320383]; exempt+0.388426meV,
CI[+0.156353,+0.616049]. Both auxiliary endpoints select pass60:
uniform0.143366722912eV; exempt0.141596541905eV.
Exemption improves the auxiliary comparison by1.770181meV, but neither
auxiliary beats the primary reference. This does not replace the primary gate.

Both arms finish60passes/46860updates/5998080sample presentations.
Four named predictors belong to two live training trajectories. Row intervals
condition on one training seed and reused selection role, not seed variance.
See [attribution](analysis.md) and the shared terminal records for qualification.

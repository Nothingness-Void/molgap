# Path + corrected EMA: no supported additive gain

On 2026-10-03 controller A claimed terminal event
`evt-95a1e9f6425ae35097541ffe` for Kaggle3 ID136793757 version1. Saved artifacts
passed independent no-inference acceptance. The supplied reference, frozen
initialization/source, fixed data, 60 observations, checkpoint chunks, runtime
qualification, observed roles and measured allocation cost were verified.

| Selected endpoint | Internal-development MAE (eV) | Best epoch (zero-based) |
|---|---:|---:|
| Frozen G1 + EMA0.999 | 0.14423262914597987 | 58 |
| G1 + chemical path mean + EMA0.999 | 0.14501232706010342 | 59 |

Candidate-minus-reference was +0.000779697914 eV. The paired 50,000-row bootstrap
95% interval [-0.000156646727, +0.001666097854] crossed zero. The prospective
>0.003 eV material-gain gate failed: no supported additivity, not stable harm or
universal uselessness of paths. No seed, longer schedule, path variant, ensemble,
500K inference/training or full run was released.

## Attribution

All 60 live training MAE, live development MAE, LR, step and presentation
observations exactly matched the earlier G1+path EMA0.9999 arm. Correcting EMA
improved that path endpoint, but the equally corrected G1 remained better.
Separate G1/G2 gains over the weaker original control did not imply additivity.

Against corrected G1, candidate EMA deltas at epochs9/19/29/39/49/59 were
-0.000273/+0.000855/+0.000122/+0.000767/+0.000270/+0.000770 eV. Terminal online
training/live development errors were also worse by +0.000809/+0.000702 eV.
Final ten-epoch EMA improvement was only 0.0000604 eV. This did not support a
lag-only rescue or budget extension, but did not prove an asymptotic limit.

The candidate won 49.222% of individual rows, not just a tiny specialist subset.
It helped high-reference-error quintiles and hurt easy ones; these label-derived
post-hoc groups do not establish an inference-time router or authorize fusion.

## Evidence and resources

The actual candidate and frozen G1 reference appeared in the rebuilt Replay
pool with `capability=complete` and the same comparability key. The observed
comparison was STRICT_CAUSAL for the declared path intervention. The immutable
RML adapter retained INCONCLUSIVE rather than fabricating a negative policy
truth from a nonsignificant regression. Acceptance and promotion are separate.

Both models retained 5,246,817 parameters, FP32/no TF32, BS128, seed42,
46,860 steps and 5,998,080 presentations. Wall time was 11,227.362 seconds
(3.119 hours); two allocated T4s cost 6.237423 device-hours, including the
explicitly justified idle device. The optional empty preflight text log was
preserved; all required artifacts passed. No protected official validation,
test-dev or challenge role was read; no model ran locally.

Authority: [acceptance](acceptance.json), [curve/replay analysis](analysis.json),
[strict comparison](../degree_path_bond_mean_ema999/results/comparison_readiness.json)
and [closure](../degree_path_bond_mean_ema999/results/closure.json).
The independently authorized final-readout dual was not cancelled, altered or
promoted by this result.

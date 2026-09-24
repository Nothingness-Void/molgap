# Frozen 100K → 500K PairToken scale diagnostic

Decision date: 2026-09-24. This separate, prospective `NO_TRAIN` question
closed with accepted inference job `122911152`. The two original 100K
checkpoints were SHA-256 bound and reproduced their saved 50,000-row 100K
development predictions (maximum differences `8.58e-6` and `1.43e-6 eV`).
The fixed 100K and 500K graph manifests, 100K target transform, 40 atomic
5,000-row chunks, source indices, targets, finite outputs, and all protected
role flags passed independent no-inference acceptance. Full results and
bootstrap details are in `results/analysis.json`; scheduler, cost and artifact
identity are in `results/job_metadata.json` and `results/terminal.json`.

| Trained on | Evaluated on | K1 MAE | PairToken MAE | PairToken gain |
|---|---|---:|---:|---:|
| fixed 100K | original 100K development | 0.141374 | 0.138330 | +0.003044 eV |
| fixed 100K, weights frozen | disjoint 500K development | 0.141253 | 0.140837 | +0.000416 eV |
| matched60-v4 500K | same 500K development | 0.103605 | 0.104304 | -0.000699 eV |

The frozen pair's change between disjoint development roles is `-0.002627
eV`; an independent-row bootstrap gives 95% CI `[-0.003805, -0.001441]`.
Thus the original 100K development advantage is largely *not portable* to
the later 500K molecules even before further training. On those same 500K
development rows, the additional contrast between the frozen-100K pair and
the separately 500K-trained pair is `-0.001115 eV` (paired row-bootstrap 95%
CI `[-0.002059, -0.000158]`). These two contrasts sum arithmetically to the
observed `-0.003742 eV` gain change; roughly 70% appears in the role change
and 30% in the subsequent training-contract comparison. **These percentages
are descriptive, not identified causal shares.** The 100K development role
selected both checkpoints, and 500K training also changed data exposure,
schedule length, target normalization and selected checkpoint.

The role populations differ: mean atom count `13.94 → 14.62`, mean conjugated
bond fraction `0.502 → 0.453`, and mean Gap label `5.372 → 5.664 eV`. Yet
single-descriptor standardization attributes only `-0.000427 eV` to the atom
count marginal shift and `-0.000029 eV` to the conjugation marginal shift;
most of the frozen-pair attenuation occurs *within* their coarse bins. This
rules out a simple explanation that the sample merely contains more large or
less conjugated molecules. It does not identify which unmeasured chemical
feature, joint interaction, target shift, or development-selection effect
causes the within-bin change. On 500K rows, the frozen 100K pair still helps
the highest-conjugation bin by `+0.002633 eV`, but the 500K-trained pair loses
there by `-0.001017 eV`; this is a diagnostic subgroup, not a deployable gate.

The saved matched60-v4 learning curves corroborate an additional late
generalization deficit. Across epochs 40–59 PairToken's training MAE was
lower than K1 by `0.001667 eV` on average, while its development MAE was
higher by `0.001126 eV`; the relative train–development gap therefore grew
to about `0.002793 eV`. The accepted frozen-network intervention already
showed the relation branch is active, so “the new module was never used” is
inconsistent with evidence. Extra capacity is only 22,848 parameters
(`+0.62%`) and sample presentations actually grew from about 4.00M to
30.00M, so “fewer total presentations for a larger model” is not a supported
explanation for this 100K→500K pair. Larger-capacity fitting, K1 catching up
on relation information, and schedule/selection effects remain hypotheses,
not resolved causes.

Conclusion: there are **two observed contributors to lost relative gain**:
poor portability of the original selected advantage to later development
molecules, and a further 500K-training-contract generalization reversal on
the same rows. Their exact mechanisms are not identified. PairToken does not
qualify for another seed, full-scale promotion, an inference-time router, or
protected-role access. K1 and the desktop-owned full-scale line are unchanged.
No new training is released by this diagnostic.

Preflight job `122910464` passed. An initial full launch, `122910907`, failed
in three seconds from a Bash `set -u` empty-array wrapper error before model
execution; wrapper fix `5727f906` was syntax-checked, then the sole full retry
`122911152` completed. All three scheduler attempts and their native allocated
time are recorded, with no invented CPU-hour measurement. The diagnostic is
V5 endpoint evidence, not a strict causal training pair or a replay-ready
training trajectory.

# Attribution: norm decay and multi-EMA 100K

The primary normalization intervention regresses. Longer EMA also regresses
on both exact live trajectories. Retain uniform WD0.05/EMA0.999 as the reference
recipe. This one-seed100K screen establishes neither a universal optimum nor
a500K transfer conclusion.

## Endpoints and provenance

| Training arm | EMA | Selected pass | Development MAE,eV |
|---|---|---:|---:|
|uniform_wd_multi_ema|0.999|42|0.139440192079|
|uniform_wd_multi_ema|0.9999|60|0.143366722912|
|norm_no_decay_multi_ema|0.999|47|0.141208115811|
|norm_no_decay_multi_ema|0.9999|60|0.141596541905|

Package14de351c195d59c8c6296363f52b70212c3a23227782824fe3c001d88c6315f3;
spec e363494b2e6888b5d6ccc78b032aa664448c95537ae2e84136dab460c2f93df7;
remote source cec17977c38551d4f4d70c8ee82cd0d4a437e7a0.
The shared verifier owns source/runtime/row/hash validation, not these labels alone.

Artifacts: D:/molgap-artifacts/retrieved/pcqm-gptrans-norm-ema-100k-20261104-s42/v1/version1/.
Shared diagnosis: D:/molgap-artifacts/runs/pcqm-gptrans-norm-ema-100k-20261104-s42/v1/reports/diagnostic_e2d1d5522381ce763d18a5a0213f5fa23ae399a52aa9b0834821377602832b76.json.
Saved-prediction helper output: D:/molgap-artifacts/timings/norm-ema-100k-analysis.json.
Its generic ensemble/fit/n_inits fields are incidental: four predictors are
not four independent initializations. Only declared paired contrasts and scores
are used for this decision. All four contrasts use2000resamples,seed42.

## Curves and discriminators

|Pass|Uniform EMA0.999|Exempt EMA0.999|Uniform EMA0.9999|Exempt EMA0.9999|
|---:|---:|---:|---:|---:|
|10|0.173343|0.174411|0.334896|0.318317|
|20|0.150373|0.151935|0.292001|0.204206|
|30|0.142075|0.145088|0.232093|0.166341|
|40|0.139627|0.142357|0.181975|0.150085|
|50|0.139727|0.141231|0.154396|0.143658|
|60|0.139867|0.141309|0.143367|0.141597|

Exempt primary trails uniform primary at every listed matched exposure.
Selected-to-final degradation is0.101268meV(exempt) versus0.426696(uniform).
This is not simply a large late collapse. The shared diagnosis flags neither
early-lag/late-recovery nor late fade, and global bias shift is not dominant.
Zero decay does not rescue this recipe; universal superiority of norm decay
is not established by one seed.

Both0.9999 curves lag substantially and improve to the boundary. The approximate
10000-update EMA window spans12.8passes here, versus1000updates/1.28passes
for0.999. Lag while live weights change is consistent with these curves.
The difference between the two EMA penalties is-3.538105meV: the effect depends
on the training trajectory. The evidence cannot isolate initialization effects,
averaging nonlinear weight paths, or weight-scale effects. Continued improvement
of a lagging EMA does not prove more training would beat the short-EMA endpoint.

Final online live normalized-Gap train MAE is0.055250929(uniform) and
0.056693444(exempt). These are not fixed-weight eV development metrics;
do not subtract them to infer overfitting or underfitting. Exposure shortage,
capacity ceiling, and why100K improvements fail at500K remain insufficient_evidence.

The shared post-hoc equal blend of primary predictions scores0.134351400964eV,
versus best single0.139440192079eV. This shows complementary errors but cannot
identify a normalization-exemption advantage versus an equal-cost two-seed
ensemble. It grants no training, teacher, adoption or promotion.

## Resources and roles

Each arm finishes60passes/46860steps/5998080presentations. Process wall:
9940.313606s(uniform),9993.409795s(exempt),ratio1.00534; this is not isolated
kernel-speed measurement. Python allocation ledger:10370.134962wall seconds,
20740.269924T4-device seconds(5.761186device hours),including allocated idle.
Queue and pre-Python provisioning remain unknown.

Fixed100K train and50000selection-used development rows are used. The hash-bound
data-open audit records protected roles untouched within its stated Python-main-
process observation scope; see role_audit_v1. Strict retrieval fetches only
manifest-required artifacts, native proof dependencies and execution costs.

Stop this WD/EMA search under this recipe. No additional grid or scale-up is
justified. New work requires a distinct evidence-supported question.

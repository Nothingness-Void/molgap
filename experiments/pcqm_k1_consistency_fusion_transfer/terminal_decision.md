# Fixed K1 equal-blend transfer decision — 2026-10-04

Outcome: **NO_TRAIN**. Completed positive retained-checkpoint inference
diagnostic; fixed equal blending passes the frozen nomination rule against
both constituents. No model adoption, training replay claim, new training,
distillation, scale-up or official-role release.

On source_idx [500000,550000), Gap MAE is mean2 0.14400446114063262 eV,
consistency2 0.14070332780838013 eV, fixed equal blend 0.13789134101867676 eV.
Fusion gain versus consistency is 2.811986789703369 meV; paired-row 95% interval
[2.4006143468022345,3.2208507683873178] meV. Versus mean2 the gain is
6.1131201219558715 meV; interval [5.728896095395088,6.508566447794437].
Both satisfy >=1 meV with a positive lower bound. Rows are development for
other historical experiments, outside both retained arms' original training
and checkpoint selection; this is not a sealed test. One seed and exploratory
row bootstrap do not quantify retraining variance.

The selected live epoch-37 checkpoints strictly load into source-identical K1.
Reconstruction on 2048 old development rows has max delta 1.90735e-6 eV
(mean2) and 2.38419e-6 eV (consistency), below 1e-4 eV tolerance. New predictions
are aligned and finite, the mixture has no fitted weight, and geometric fields
are removed before pure-2D inference. All official roles remain untouched.

Native measured assigned RTX5060 allocation:35.2542018 device seconds including
qualification and both common-cohort passes. Common-cohort pass wall times
16.1087876s +13.5354761s=29.6442637s; synchronized forward times
10.7922984s +11.4879808s=22.2802792s. These sequential single observations are
not a replicated hardware benchmark. The dual pass costs approximately twice
a single pass; GPU allocated peak is143996416 bytes, not total device memory.

Attribution: the prior equal-blend gain transfers to this separate cohort;
complementary prediction errors are actionable without another module.
The two individual checkpoints differ only in the accepted training objective,
but this result does not establish a learned-feature or optimization cause.
No evidence here distinguishes underfitting, exposure shortage, or overfitting.
The original subthreshold training outcomes and replay exclusions remain intact.

Next justified research question, separately authorized and prospective:
can a single K1 student retain this fixed teacher's measured improvement while
restoring single-model inference cost? Teacher quality is supported; successful
distillation is not established. Do not repeat baseline training for comparison.

Git routing: accepted diagnostic and reusable inference implementation to
molgap-desktop under BRANCHES' non-promotion route. Operational full-scale
EdgeState reference remains unchanged.

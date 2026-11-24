# Decision — 2026-11-08 JST

NO_TRAIN: the prospectively fixed equal blend is positive on the consumed500K
development cohort. Recommend planning a separately authorized full-scale
two-expert validation, with accelerated original K1 as the complementary expert
to GPTrans. Do not launch training or consume official/test roles from this
diagnostic. This is not full-scale qualification or training replay readiness.

The equal blend MAE is0.096338391495eV versus GPTrans0.099987377815 and
K1 0.101863694448. Equal-minus-GPTrans is-3.648986meV with paired95% interval
[-3.992140,-3.299123]meV (2000 row resamples, seed42,50000 aligned rows).
Both the prospective point3meV/negative-upper check and the existing helper's
stricter upper<=-3meV check pass. Against the existing equal two-GPTrans blend
(0.097524865608), improvement is1.186474meV, paired95%[0.848935,1.525876]meV.
That incremental gain is below3meV, and is reported separately rather than
claimed to pass a3meV margin against the two-model reference.

Predeclared K1 weights0.25 and0.75 yield0.097009799492 and0.097991639140eV.
All three fixed weights improve on the GPTrans single, without fitting weights.
No secondary weight replaces the primary. Full-cohort signed residual
correlation is0.850298; absolute-error correlation0.809933. The primary wins
54.498% of rows versus GPTrans. These support useful error complementarity.

The benefit is not universal:>=8eV has1339rows (2.678%) and worsens5.202348meV
versus GPTrans,10.501426meV versus the two-GPTrans blend. The other three
physical-Gap bins and all target quartiles improve against GPTrans. Retain this
tail weakness in any future full-validation plan; true target bins are analysis,
not an inference-time router or a license to tune on the same labels.

The existing diagnostic's older three-expert mean was0.096301eV, descriptive
context only. This new two-expert result is close in average score with one
fewer model; no paired equality or serving-time claim is made for that scalar.

All inputs match accepted prediction manifests. Their int64 IDs and float32
targets are exactly equal, [500000,550000), Gap eV. LIVE K1 and EMA GPTrans are
deliberately heterogeneous selected recipes; this is fixed-prediction arithmetic,
not causal attribution of an architecture or optimizer. Development was used
for endpoint selection and this diagnostic. Row bootstrap does not estimate
training-seed variability, independent test performance or full-scale transfer.

Actual arithmetic wall3.790398s/processCPU4.343750s, four CPU threads. No model,
checkpoint, optimizer, accelerator, training or inference was used. Existing
selected predictions remain externally retained by hash; only compact results
are added. The completed diagnostic uses NO_TRAIN RML closure with observed
development use, not a fabricated training trace. See [analysis](analysis_zh.md),
[result](results/result.json), [protocol](protocol.md), and [inputs](inputs.json).

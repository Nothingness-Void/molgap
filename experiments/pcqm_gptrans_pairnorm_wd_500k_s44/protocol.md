# GPTrans PairNorm500K weight-decay discriminator, initialization44

Desktop prospective question, authorized2026-11-03. Companion initialization is43.

## Evidence and hypothesis

Read ../pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_acceptance/attribution.md
and ../pcqm_gptrans_pairnorm_init_noise_500k/decision.md. Lower peak LR was negative;
late stochastic TRAIN decline and EMA-development regression do not identify overfitting.
The observed init-only spread is1.332meV. K1 WD negatives do not establish the answer
for this different family and from-scratch full-horizon contract. No GPTrans WD
intervention was found in desktop RML. Hypothesis: stronger decoupled shrinkage changes
the generalization trajectory. Low WD is a new alternative, not the accepted baseline.

## Controlled change

wd_low=0.01 versus wd_high=0.10; only weight_decay differs within this physical job.
Both start from retained initialization44, sampler/dropout seed42, GPTrans-T/2
BondLocal+fresh-pair-update-norm, LR1e-3, four-pass warmup/cosine60 to1e-6,
EMA0.999, best-development-EMA, physical BS128/drop-last, FP32/noTF32.
Fixed500K train[0,500000); internal50K development[500000,550000), used for selection.
60 passes=234360 updates=29998080 presentations per arm. No teacher, pretraining,
geometry or changed data. Historical WD0.05 at matching initialization is retained
context; no strict causal claim against a cross-session reference without qualification.
Official validation/test-dev/test-challenge remain sealed.

## Reading and decision

Primary delta=wd_high minus wd_low selected EMA MAE. Material nomination requires
<=-0.003eV and a negative upper paired-row95%CI (2000 resamples, seed42) in EACH
initialization pair. Report the two effects separately and their mean; two initializations
do not establish a population variance estimate. Smaller gains stay below gate.
Compare complete curves at passes10/20/30/40/50/60 and selected-to-final drift;
TRAIN stochastic objective and EMA MAE are not a fixed-weight generalization gap.
If directions reverse or neither passes, do not nominate stronger WD. Low beating high
is evidence of the direction of this contrast only, not proof of gain over WD0.05.
Both pairs must complete before a reproducibility conclusion; partials stay truthful.

## Cost, logging and closure

One T4x2 session per pair, ceiling42900seconds, cleanup reserve180seconds.
Two-slot total ceiling85800seconds/23.833334 quota-hours,47.666668 T4-device-hours.
Expected10-12hours per slot is an estimate, not assurance of completion.
Shared epoch logging only; shared atomic checkpoint and two retained resume chunks.
No automatic retry, successor, continuation or full promotion. Luna owns monitoring.
Use submit_workflow.py, strict minimal retrieval, draft-closure and accept_run.py.
Require per-arm model/prediction/trace/resume/native proof, observed role use and
measured cost, then four distinct complete replay entries without exclusions.
Only necessary manifests and proof dependencies are retrieved to D:.

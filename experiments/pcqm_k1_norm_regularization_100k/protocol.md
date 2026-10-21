# Frozen regularization protocol

User authorized2026-10-22 JST Kaggle1 two T4x2 slots/four new candidates.
BN pair: bn_wd1e4 vs bn_wd1e3. LN pair: ln_wd1e4 vs ln_wd1e3.
All use LR2e-4,warmup8/cosine160,minLR1e-6,160epochs,124960updates,
15994880presentations,batch128/drop-last,seed42,FP32/noTF32,AdamW,L1,
clip1,dropout0.05,live best-development selection and EMA0.999 tracking only.
Same transported full base state and fixed100K/next50K as accepted predecessor.
Only weight decay differs within each job. Cross-job norm and historicalWD1e5
contrasts remain exploratory; no seed variance or strict causal claim.
No historical baseline is retrained; moderateWD is a new candidate comparator.
Primary point gain threshold is0.003eV versus same-job moderateWD; report
aligned50000-row paired bootstrap, distinguish row uncertainty from seed variance.
Retain every epoch train/dev MAE,usedLR,steps/exposure,checkpoint hash,wall.
Predeclare40/80/120/160 summaries; no early-stop from a partial curve.
Check whether strongerWD raises train error but helps dev, harms both, or
changes bias/tails; online train/eval dev differences do not prove overfitting.

Per pair max15T4-device hours,7.5hour Python wall ceiling via existing packaged
authority watchdog; total30T4-device hours. Expected12-13T4-device hours per
pair. Platform quota units remain separate. All-arm native preflight barrier,
atomic best/last,resume_keep_count2; retain exact source/config/initialization.
On cap/interruption retain partial evidence; no retry/scale-up is released.
Only train/development; official_validation,test_dev,test_challenge sealed.
Minimal manifest-bound final acceptance/nativeproof downloads to D: after size
review; no environment/cache/epochchunk bulk retrieval. Each arm receives
independent terminal RML; strict replay has its own qualification gate.
No model adoption or automatic successor. After exact submissions and Git push,
shut down desktop as requested; no monitor or server handoff.

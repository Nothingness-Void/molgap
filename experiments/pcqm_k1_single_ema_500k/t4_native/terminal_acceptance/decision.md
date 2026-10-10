# Bounded T4 terminal acceptance

2026-10-11 Asia/Tokyo. CPU saved-artifact inspection only; no training,
inference, accelerator access, new role consumption or remote operations.

## Disposition

Exact kernel `nothingnessvoid/molgap-k1-single-ema-500k-t4-20261010`,
version1, kernelID137983743. Parent-retained SDK listing reports COMPLETE;
worker and checkpoint report STOP_FOR_COST. These describe different layers.
Mechanical disposition: PASS_RETAINED_PARTIAL. Scientific disposition:
INCONCLUSIVE. No adoption, complete60 endpoint, strict60 replay, full release,
early-stop policy, or successor authorization follows from this acceptance.

[Machine report](acceptance.json) owns the inspected hashes, exposure, saved
predictions, uncertainty, native costs and limitations. Frozen worker source is
`a858c4c474aad731fb058ce9cf64c25bfe5e20ef`; acceptance code is subsequent local
infrastructure, not a change to that source. [Protocol](../protocol.md) and
[parent contract](../../protocol.md) retain scientific authority.

## Matched comparison

Matched prefix:18 complete epochs,70,308 steps and8,999,424 presentations per
arm. Reference actually retained19 complete epochs,74,214 steps/9,499,392
presentations. EMA retained18 complete epochs plus2,912/3,906 batches,
73,220 steps/9,372,160 presentations. The unmatched tail is excluded.

| Comparison | Reference clean MAE | EMA clean MAE | Reference-minus-EMA | Paired-row95% gain interval |
|---|---:|---:|---:|---:|
| Best within matched18 prefix | 0.11723301 | 0.11130745 | 0.00592556 | [0.00540671,0.00642884] |
| Matched-last18 endpoint | 0.12076076 | 0.11130745 | 0.00945333 | [0.00903146,0.00987741] |

Units:eV. Best-prefix selection uses reference zero-based epoch15 and EMA17;
matched-last uses epoch17 for both. The former compares two selected checkpoints,
not identical checkpoint exposure. Bootstrap10,000/seed42 measures paired-row
uncertainty only, not training-seed variance or selection uncertainty. Internal
50K development was repeatedly selection-consumed, not an independent holdout.
The full60 material gate was not evaluated.

Raw predictions are secondary only: best-prefix reference0.11310193 versus
EMA0.11379638; matched-last reference0.11871282 versus EMA0.11379638.
They do not replace the frozen clean primary selection.

## Native cost and custody

Observed allocation wall32,289.88923883438s. Two physical Tesla T4s allocated,
one visible/used device0, one idle allocation:17.93882735490799 allocated
T4-hours versus8.969413677453995 visible-device allocation hours. Neither is
GPU busy time or provider billing. Provider billing, CPU-hours and queue-hours
remain unknown. Sum of recorded formal step intervals is13,028.010332s reference
and13,619.541929s EMA; complete-round intervals also contain loading, calibration,
evaluation and checkpoint overhead. No full60 cost extrapolation is accepted.

Large checkpoints/source/raw trace remain in parent-managed ignored custody;
do not stage them. The report binds each inspected worker file by exact SHA256,
and lists remote-manifest files not locally retrieved. In particular matched-last
reference `epoch_17.pt` is
`bef9c85aa1336107879965575cc3163824e2a149c9d65c6c3b294346c55d8b69`.
The source SHA sidecar absent from retrieval was instead checked against the
frozen payload using the pre-submission local copy; its exact custody is reported.

Checkpoint inspection verifies finite live/EMA/optimizer tensors, RNG contents,
sampler coordinates, all retained step rows, epoch LR/order/loss aggregates and
resume counters. No local replay on the different CPU runtime occurred.
Saved clean/raw/selected predictions align50K source indices and targets.
Calibration reports and selected BN-buffer hashes match; frozen source binds
first16,384 training members, features only. No graph shard or target-role reload
was performed. CPU FP32 reduction tolerance is5e-7 relative/1e-8 absolute;
cosine LR tolerance is1e-14 relative/1e-18 absolute across Windows/Linux libm.
Saved values remain unchanged and CPU-recomputed values are retained separately.

[Attribution](attribution.md) must accompany this disposition before another
module. [RML blockers](../rml/terminal_closure_blockers.md) own the remaining
formal-finalization gap. Parent owns integration, final tests and platform work.

## Local verification

`D:/文档/molgap/.venv/Scripts/python.exe`, owner `src` on PYTHONPATH,
CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1. Focused synthetic suite:
20 tests passed. Real saved-artifact CPU acceptance and10,000-draw paired-row
bootstraps completed successfully. Canonical trace export re-run verified
idempotence against the accepted raw trace. Tests cover corrupted prediction
order/targets/finiteness/MAE, paired alignment, trace counters/LR/order/loss,
resume contents and canonical trace tampering. These are infrastructure checks,
not training, inference or scientific qualification on the local CPU.

# Top-20 Architecture Transfer Screen Decision

Status: seed-42 remote short screen complete; candidate rejected at the QM9 gate.

## Predeclared decision rule

The old architecture is the promoted QM9 precision fusion from
`experiments/qm9_architecture/results/summary.json`. Its reference is
`0.0708138843 eV` average MAE and `0.084272936 eV` Gap MAE on the fixed
ETKDG-successful protocol.

Each replacement candidate must, on the same requested split and successful
row intersection:

1. reach at least `0.003 eV` lower average MAE and `0.002 eV` lower Gap MAE;
2. repeat the improvement for encoder seeds 42, 43, and 44, or have a
   three-seed mean improvement with no target regression larger than `0.002 eV`;
3. remain within the remote resource budget and produce atomic checkpoints,
   independent metrics, and an embedding payload.

The first remote job for each materially different candidate is a bounded
30-epoch, seed-42 screen. It is evidence for resource/accuracy triage only. If
one clears both metric margins, the three-seed confirmation is submitted. If it
does not, that candidate is not promoted and no PCQM training is launched from
it.

## Current implementation

`src/molgap/tgt_lite.py` implements the first isolated pair/triplet screen.
That seed-42 screen completed at `0.0879193991 eV` average MAE and
`0.1031505316 eV` Gap MAE, so it failed the predeclared gate. A first fused
topology/geometry implementation then failed in the legacy PyG GINE branch
before metrics. Its GINE-free v2 replacement exposed a mixed 2D/explicit-H
batching error before metrics.

The v3 correction is implemented in `src/molgap/tgt_hybrid_v2.py` and
`src/molgap/topology_attention.py`: topology and explicit-H geometry now carry
independent node/edge batch metadata, while retaining the transferable global
attention, pair-channel, and triplet-interaction ideas. A remote synthetic
forward with deliberately mismatched topology and geometry node counts passed
on the IMS environment. The resulting bounded seed-42 QM9 screen was submitted
as IMS job `1309959.ccpbs1` using
`platforms/ims/top20_qm9/train_tgt_hybrid_v3.pbs`. It completed 30 epochs and
produced independent test metrics. The candidate returned average MAE
`0.0930039063 eV` and Gap MAE `0.1095121875 eV`, versus the old precision-
fusion reference `0.0708138843 eV` and `0.0842729360 eV`. It therefore failed
both promotion margins; no seed-43/44 confirmation or PCQM training submission
is authorized from this candidate.
At 11:22 JST it subsequently obtained `ccg004`; epoch 0 completed with
intermediate validation MAE `0.26265 eV` and no runtime error. This intermediate
value is not the final QM9 gate result.
At 11:32 JST, epoch 2 had reached intermediate validation MAE `0.20968 eV`;
the job remained in progress and no final metrics had been written.
At 11:41 JST, epoch 3 had reached intermediate validation MAE `0.19035 eV`;
the job remained in progress and no final metrics had been written.
At 11:43 JST, epoch 4 had reached intermediate validation MAE `0.17713 eV`;
the job remained in progress and no final metrics had been written.
At 11:56 JST, epoch 7 had reached intermediate validation MAE `0.14882 eV`;
the job remained in progress and no final metrics had been written.
At 12:00 JST, epoch 8 had reached intermediate validation MAE `0.14361 eV`;
the job remained in progress and no final metrics had been written.
At 12:09 JST, epochs 9 and 10 had reached intermediate validation MAE
`0.13606 eV` and `0.13366 eV`; the job remained in progress and no final
metrics had been written.
At 12:14 JST, epoch 11 had reached intermediate validation MAE `0.12928 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 12:24 JST, epochs 12 and 13 had reached intermediate validation MAE
`0.12611 eV` and `0.12023 eV`; the checkpoint timestamp advanced and the job
remained in progress without final metrics.
At 12:28 JST, epoch 14 had reached intermediate validation MAE `0.11694 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 12:33 JST, epoch 15 had reached intermediate validation MAE `0.11431 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 12:38 JST, epoch 16 had reached intermediate validation MAE `0.10925 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 12:43 JST, epoch 17 had reached intermediate validation MAE `0.10706 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 12:47 JST, epoch 18 had reached intermediate validation MAE `0.10580 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 12:52 JST, epoch 19 had reached intermediate validation MAE `0.10168 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 12:56 JST, epoch 20 had reached intermediate validation MAE `0.10008 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:01 JST, epoch 21 had reached intermediate validation MAE `0.09970 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:06 JST, epoch 22 had reached intermediate validation MAE `0.09785 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:10 JST, epoch 23 had reached intermediate validation MAE `0.09525 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:15 JST, epoch 24 had reached intermediate validation MAE `0.09480 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:20 JST, epoch 25 had reached intermediate validation MAE `0.09370 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:24 JST, epoch 26 had reached intermediate validation MAE `0.09334 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:31 JST, epoch 27 had reached intermediate validation MAE `0.09261 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:34 JST, epoch 28 had reached intermediate validation MAE `0.09234 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:39 JST, epoch 29 had reached intermediate validation MAE `0.09233 eV`;
the checkpoint timestamp advanced and the job remained in progress without
final metrics.
At 13:41 JST, the job reached terminal completion and wrote final metrics.
On the fixed successful ETKDG test intersection, HOMO, LUMO, Gap, and average
MAE were `0.0812518746`, `0.0882476419`, `0.1095121875`, and `0.0930039063 eV`.
The candidate failed the predeclared replacement gate and was not promoted.

The Route B continuation has been prepared but not submitted. The accepted
source cache exposes separate GPS 2D topology and primary ETKDG explicit-H
views, so `build_pcqm_tgt_cache.py` now records explicit per-view batch counts,
`src/molgap/pcqm_tgt_hybrid_v3.py` consumes those fields, and
`platforms/ims/top20_qm9/train_pcqm_tgt_hybrid_v3.pbs` is the matching training
adapter. The older `train_pcqm_tgt_lite.pbs` is intentionally excluded because
it would train the rejected isolated candidate rather than the v3 hybrid.

## Reopened pure-2D candidate

The user explicitly reauthorized unlimited Route C attempts. The next candidate
is `edge_global_2d`, a compact GPTrans-T/EGT-inspired encoder that keeps a
learnable state on each 2D bond edge, updates it from endpoint nodes, and uses
the edge state as a bias for global node attention. It uses no conformer and no
optional GINE/torch-cluster CUDA path. Remote synthetic forward and remote
syntax checks passed. IMS job `1311296.ccpbs1` was submitted for a bounded
30,000/3,000/3,000 topology-only preflight; this is not yet the full QM9 gate.

The first preflight attempt reached `ccg001` but exited before training because
the remote copy of the thin experiment wrapper still had a stale argparse
candidate list and rejected `edge_global_2d`. Its log is retained at
`logs/edge_global_2d_preflight_seed42.log`; no metrics were produced. The
wrapper was corrected and uploaded to the remote experiment tree, and remote
static/import checks confirmed that the candidate is now registered. The
failure log was copied to
`logs/edge_global_2d_preflight_seed42_entrypoint_failed.log`; a new isolated
retry, `1311369.ccpbs1`, was submitted with separate output paths and still
requires the same preflight gate.

Retry `1311369.ccpbs1` reached `ccg010` and entered the forward pass, then
failed before metrics because mixed precision produced a Half dense edge-bias
tensor while the indexed edge-bias values were Float. The traceback is
retained at `logs/edge_global_2d_preflight_retry_seed42_dtype_failed.log`.
The edge-bias write now casts values to the destination dtype; local static
checks and remote PBS syntax checks pass. Isolated retry2 job
`1311419.ccpbs1` was submitted with separate output paths.

At 16:38 JST, retry2 completed on `ccg011` and wrote final metrics. The best
checkpoint was epoch 18. On the 3,000-row test split, HOMO, LUMO, Gap, and
average MAE were `0.1186921597`, `0.1275041848`, `0.1668345779`, and
`0.1376769692 eV`, respectively. The preflight validation average was
`0.1367870569 eV`. This is far above the replacement gates of `0.0678138843`
average and `0.0822729360` Gap, so the candidate failed decisively and was not
promoted to the full QM9 gate. No seed-43/44 or Route B job was submitted.

The result separates implementation stability from representation quality:
the pure-2D edge-state/global-attention path ran stably on one A100 with
`4,688,803` parameters and no conformer construction, but its topology-only
representation lacks enough orbital-sensitive information for this target. The
next Route C attempt should retain the pure-2D deployment constraint while
restoring stronger multi-scale/local-global interaction capacity, rather than
widening this exact edge-bias stack.

## Pair/triplet pure-2D candidate

The next candidate is `pair_triplet_2d`. It keeps a persistent state for every
node pair, initializes that state from bond features and one-to-three-step 2D
walk indicators, updates it with a low-rank triplet contraction, and preserves
both bond-local message passing and global attention. This is the compact
pure-2D subset of the TGT/GEM-2/GPTrans ideas identified in the top-20 audit;
it uses no conformer and no optional geometry or GINE CUDA path. Local
`py_compile`/diff checks and remote import/PBS syntax checks passed. IMS job
`1311867.ccpbs1` was submitted for a bounded 30,000/3,000/3,000 seed-42
preflight with isolated output paths. It must pass the preflight gate before a
full QM9 run is considered.

At 16:50 JST, the RCCS scheduler reported `1311867.ccpbs1` still in normal
`Queue` state. The H queue had zero free GPUs and approximately 39 waiting GPU
jobs, so this is an external resource wait rather than a candidate failure or
an application error. The job remains the sole submitted attempt for this
candidate.

At 17:26 JST, `1311867.ccpbs1` finally ran and failed before epoch 0. The
remote traceback showed that the dense pair feature tensor had width 7 while
the first pair projection expected width 8. The construction contains four
bond channels plus three path channels, so the correct width is 7. The complete
failure log is retained at
`logs/pair_triplet_2d_preflight_seed42_shape_failed.log`; no metrics or model
promotion evidence was produced.

The source was corrected to use `edge_dim + path_steps`, passed local static
checks, and passed remote Python/PBS syntax checks. A new isolated retry,
`1312240.ccpbs1`, was submitted with separate result/model directories. No
full QM9, seed-43/44, or Route B job is authorized by this preflight alone.

At 18:13 JST, `1312240.ccpbs1` completed all 20 epochs on `ccg004` without
runtime failure and wrote its metrics and checkpoint artifacts. The best
epoch was 19. On the 3,000-row test split, HOMO, LUMO, Gap, and average MAE
were `0.1025283188`, `0.1186410636`, `0.1492583901`, and `0.1234759241 eV`,
respectively. The preflight validation average was `0.1241790056 eV`.
These values miss the replacement gates of `0.0678138843` average and
`0.0822729360` Gap by `0.0556620398` and `0.0669854541 eV`, respectively.
The candidate therefore failed on representation quality rather than
execution stability; no seed-43/44, full-QM9, or Route B job was submitted.
The 3.20M-parameter model used no conformers, and its persistent pair state
and low-rank triplet update did not compensate for the missing orbital and
distance information in the pure-2D topology input.

## Rich pair/triplet pure-2D candidate

The user-authorized follow-up candidate `pair_triplet_2d_rich` adds the
transferable EGT/TGT/GEM-2 ingredients that were absent from the failed
compact screen: symmetric atom-pair descriptors, reachability and log path
counts through five 2D walk orders, and value propagation from the persistent
pair state back into node updates. It uses 256 node channels, 96 pair
channels, ten layers, rank-16 triplet mixing, and batch 48; it remains
conformer-free and avoids optional geometry CUDA kernels. Local Python
compilation, remote Python compilation, and remote PBS syntax checks passed.

IMS job `1312364.ccpbs1` was submitted as one isolated 30,000/3,000/3,000
seed-42 preflight with separate result and model directories. At 18:24 JST it
was still in H `Queue` state awaiting a GPU, with no log or metrics yet. Its
bounded exit condition is the predeclared average/Gap gate; seed-43/44 and
Route B remain unauthorized until that gate and the complete seed-42 gate are
both passed.

At 19:31 JST, `1312364.ccpbs1` completed all 20 epochs on `ccg014` and wrote
independent checkpoint, model, embedding, log, and metrics artifacts. The best
epoch was 19. On the 3,000-row test split, HOMO, LUMO, Gap, and average MAE
were `0.0987659171`, `0.1143094152`, `0.1428524852`, and `0.1186426207 eV`.
The candidate missed the replacement gates by `0.0508287364 eV` average and
`0.0605795492 eV` Gap. It was therefore rejected for representation quality;
no seed-43/44 or Route B job was submitted from it.

The next user-authorized Route C candidate, `tgt_egt_hybrid`, was prepared with
a portable GINE-free GPS-like local/global topology branch and an ETKDG
EGT/TGT pair-state branch. Local and remote static checks passed, and a remote
small-shape forward produced a finite `[2, 3]` output. Its isolated 30k/3k/3k
seed-42 preflight is the only subsequent GPU submission authorized at this
stage; it must clear the same average/Gap gate before a full QM9 run.

At 19:35 JST, after the rich preflight result was retained, the candidate was
submitted as isolated IMS job `1312784.ccpbs1` using
`platforms/ims/top20_qm9/train_tgt_egt_hybrid_preflight.pbs`. The scheduler
placed it in H `Queue` for one GPU; no log or metrics were present at the
submission observation.

At 19:45 JST, a read-only poll still showed H `Queue` with a GPU resource
reason. No node, log, epoch, or metrics artifact was present; the job was not
re-submitted or otherwise changed.

At 19:47 JST, the same read-only poll still showed H `Queue` with no remote
artifact or log change.

At 19:52:37 JST, the job entered `Run` on node `ccg002`. The log file existed
but was empty at the first running-state poll; no epoch or metrics artifact was
yet present.

At 19:53:16 JST, the log contained only the known optional `torch-cluster`
import warning. The job remained in `Run`; no training epoch or result artifact
was present, and no corrective resubmission was made.

At 19:54:57 JST, the scheduler still reported `Run` on `ccg002`; the log was
unchanged at 314 bytes and no output artifact had appeared. The running job was
left untouched.

At 19:56:24 JST, the scheduler still reported `Run` on `ccg002`; the log and
output directories remained unchanged. No intervention was made.

At 19:56:52 JST, the log advanced to `ETKDG 2000/36000 success=1859
elapsed=158s`, confirming that the job was actively constructing the immutable
ETKDG graph cache. No model epoch or metrics artifact was available yet.

At 19:59:40 JST, the cache advanced to `ETKDG 4000/36000 success=3725
elapsed=316s`; the job remained on `ccg002` in `Run`, with no model epoch or
metrics artifact yet.

At 20:01:38 JST, the cache advanced to `ETKDG 6000/36000 success=5624
elapsed=443s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:04:20 JST, the cache advanced to `ETKDG 8000/36000 success=7473
elapsed=606s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:07:13 JST, the cache advanced to `ETKDG 10000/36000 success=9321
elapsed=776s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:09:43 JST, the cache advanced to `ETKDG 12000/36000 success=11188
elapsed=924s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:12:16 JST, the cache advanced to `ETKDG 14000/36000 success=13047
elapsed=1079s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:14:18 JST, the cache advanced to `ETKDG 16000/36000 success=14936
elapsed=1206s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:17:12 JST, the cache advanced to `ETKDG 18000/36000 success=16791
elapsed=1366s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:20:07 JST, the cache advanced to `ETKDG 20000/36000 success=18647
elapsed=1538s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:22:40 JST, the cache advanced to `ETKDG 22000/36000 success=20490
elapsed=1703s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:26:24 JST, the cache advanced to `ETKDG 24000/36000 success=22354
elapsed=1858s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:28:06 JST, the cache advanced to `ETKDG 26000/36000 success=24190
elapsed=2023s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:31:11 JST, the cache advanced to `ETKDG 28000/36000 success=26021
elapsed=2201s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:33:56 JST, the cache advanced to `ETKDG 30000/36000 success=27877
elapsed=2372s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:36:23 JST, the cache advanced to `ETKDG 32000/36000 success=29768
elapsed=2517s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:39:04 JST, the cache advanced to `ETKDG 34000/36000 success=31636
elapsed=2661s`; the job remained in `Run` on `ccg002`, with no model epoch or
metrics artifact yet.

At 20:41:36 JST, the immutable cache completed at `ETKDG 36000/36000
success=33498 elapsed=2819s`. The log then entered training and emitted only the
known `torch-scatter` max-reduction acceleration warning; the job remained in
`Run` on `ccg002`, with no completed epoch or metrics artifact yet.

At 20:44:11 JST, epoch 0 completed on `ccg002` with train loss `0.47958` and
validation MAE `0.34132 eV` in `164.7s`; the remote checkpoint was written and
the job remained in `Run`. This is still only the bounded preflight, so no gate
decision is made from the first epoch.

At 20:47:28 JST, epoch 1 completed with train loss `0.28305` and validation MAE
`0.24079 eV` in `157.7s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. The validation error is decreasing, but the preflight gate
still depends on the final test metrics.

At 20:50:14 JST, epoch 2 completed with train loss `0.24345` and validation MAE
`0.23497 eV` in `157.5s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Training remains stable with a gradual validation decrease.

At 20:52:04 JST, epoch 3 completed with train loss `0.21811` and validation MAE
`0.20584 eV` in `158.0s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease without runtime errors.

At 20:55:29 JST, epoch 4 completed with train loss `0.20134` and validation MAE
`0.20140 eV` in `163.3s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease.

At 20:58:30 JST, epoch 5 completed with train loss `0.18767` and validation MAE
`0.17656 eV` in `163.7s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to improve substantially.

At 21:01:32 JST, epoch 6 completed with train loss `0.17731` and validation MAE
`0.16985 eV` in `166.1s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease.

At 21:03:19 JST, epoch 7 completed with train loss `0.16563` and validation MAE
`0.16089 eV` in `159.8s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease without runtime errors.

At 21:06:19 JST, epoch 8 completed with train loss `0.15572` and validation MAE
`0.15996 eV` in `159.4s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation remains stable with a slight further decrease.

At 21:08:52 JST, epoch 9 completed with train loss `0.14596` and validation MAE
`0.14671 eV` in `160.9s`; the atomic checkpoint write completed and the job
remained in `Run` on `ccg002`. Validation continues to decrease.

At 21:11:15 JST, epoch 10 completed with train loss `0.13817` and validation MAE
`0.14124 eV` in `163.1s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease.

At 21:14:08 JST, epoch 11 completed with train loss `0.12907` and validation MAE
`0.13918 eV` in `163.7s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease without runtime errors.

At 21:16:38 JST, epoch 12 completed with train loss `0.12078` and validation MAE
`0.13039 eV` in `163.2s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease.

At 21:19:36 JST, epoch 13 completed with train loss `0.11353` and validation MAE
`0.12640 eV` in `163.0s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease.

At 21:22:11 JST, epoch 14 completed with train loss `0.10673` and validation MAE
`0.12003 eV` in `163.7s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease.

At 21:24:48 JST, epoch 15 completed with train loss `0.10098` and validation MAE
`0.11762 eV` in `163.4s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. Validation continues to decrease.

At 21:27:27 JST, epoch 16 completed with train loss `0.09531` and validation MAE
`0.11534 eV` in `163.0s`; the checkpoint was updated and the job remained in
`Run` on `ccg002`. No final `metrics.json` was present.

At 21:30:42 JST, epoch 17 completed with train loss `0.09120` and validation MAE
`0.11279 eV` in `161.5s`; the job remained in `Run` on `ccg002` and no final
`metrics.json` was present.

At 21:33:19 JST, epoch 18 completed with train loss `0.08857` and validation MAE
`0.11263 eV` in `163.4s`; the job remained in `Run` on `ccg002` and no final
`metrics.json` was present.

At 21:35:54 JST, epoch 19 completed with train loss `0.08696` and validation MAE
`0.11207 eV` in `163.5s`, completing the final training epoch. The job remained
in `Run` on `ccg002` while final evaluation was pending; no `metrics.json` was
present.

At 21:36:54 JST, the preflight wrote final test metrics: HOMO `0.0984003`, LUMO
`0.1058973`, Gap `0.1358602`, and average `0.1133859 eV`. It missed the
replacement gates by `0.0455720 eV` average and `0.0535873 eV` Gap. The candidate
was rejected; no seed-43/44 or Route B job was submitted, and all remote logs
and artifacts remain retained.

At 21:38:32 JST, a read-only scheduler check found no remaining record for
`1312784.ccpbs1`; the final metrics and independent output files were present,
so the job was recorded as a normally completed failed-gate preflight.

After that failed-gate diagnosis, the compact `tgt_egt_compact` candidate was
uploaded with local and remote static checks passing. It preserves the measured
`tgt_lite` ETKDG pair/triplet path, adds explicit EGT heavy-atom bond channels,
and initializes the pair-to-node value residual at zero. It was submitted once
as isolated IMS job `1313997.ccpbs1` at 21:49:15 JST using
`train_tgt_egt_compact_preflight.pbs`. The initial scheduler observation was H
`Queue` with no node, log, checkpoint, or metrics; no other GPU job overlapped
the submission.

At 22:02:24 JST, the scheduler showed `1313997.ccpbs1` in H `Run` on `ccg015`;
the log had just been created and no cache progress, checkpoint, or metrics had
yet appeared.

At 22:04:16 JST, `1313997.ccpbs1` had exited before epoch 0. The retained
traceback identified an interface-only defect: the shared forward path passed
`topology_x`, but the compact encoder's `encode()` signature did not accept it.
No model, checkpoint, or metrics were produced. The signature was fixed,
revalidated remotely, and a retry with separate output and log paths was
submitted as `1314176.ccpbs1` at 22:07:18 JST; its initial state was H `Queue`.

At 22:26:29 JST, the retry moved to H `Run` on `ccg013`; no epoch log was yet
visible. The retry remains isolated from the retained first-attempt log.

At 22:29:13 JST, retry2 completed epoch 0 with train loss `0.59567` and
validation MAE `0.45560 eV` in `37.9s`; an atomic checkpoint was present and no
runtime exception had occurred.

At 22:30:19 JST, epochs 1 and 2 completed with train/validation losses
`0.42686/0.35342 eV` and `0.33355/0.29241 eV` in `32.8s` and `32.6s`. The
checkpoint continued to refresh and the run remained on `ccg013`.

At 22:31:21 JST, epochs 3 and 4 completed with train/validation losses
`0.28678/0.25365 eV` and `0.25971/0.23128 eV`, each in `32.6s`; the run remained
stable and the checkpoint advanced.

At 22:33:05 JST, epochs 5, 6, and 7 completed with train/validation losses
`0.23865/0.21489`, `0.22170/0.20223`, and `0.20935/0.19752 eV`, each in
`32.6s`; the checkpoint continued to advance on `ccg013`.

At 22:34:06 JST, epochs 8 and 9 completed with train/validation losses
`0.19808/0.18372` and `0.18777/0.18126 eV`, each in `32.6s`; no final metrics
had been written.

At 22:35:09 JST, epochs 10 and 11 completed with train/validation losses
`0.17836/0.17124` and `0.16962/0.16123 eV`, each in `32.6s`; the checkpoint
continued to refresh.

At 22:36:09 JST, epochs 12 and 13 completed with train/validation losses
`0.16205/0.15960` and `0.15608/0.15466 eV`, each in `32.7s`; the run remained
stable on `ccg013`.

At 22:37:12 JST, epochs 14 and 15 completed with train/validation losses
`0.14984/0.15089` and `0.14461/0.14732 eV`, each in `32.6–32.7s`; the
checkpoint continued to refresh.

At 22:38:11 JST, epoch 16 completed with train loss `0.14016` and validation
MAE `0.14533 eV` in `32.7s`; three training epochs remained and no final
metrics had been written.

At 22:39:10 JST, epochs 17 and 18 completed with train/validation losses
`0.13620/0.14274` and `0.13450/0.14083 eV`, each in `32.7s`; only final epoch
19 and evaluation remained.

At 22:40:20 JST, epoch 19 completed with train loss `0.13333` and validation MAE
`0.14079 eV` in `32.9s`, and final test metrics were written: HOMO `0.1265981`,
LUMO `0.1419699`, Gap `0.1766737`, and average `0.1484139 eV`. The corrected
compact retry missed the replacement gates by `0.0806000 eV` average and
`0.0944008 eV` Gap. It was rejected; the job exited normally and all retry2
artifacts remain retained. No seed-43/44 or Route B job was submitted.

The failure diagnosis identified a nonzero random initialization path from the
new bond columns into the pair projection, so the compact model was not truly
initialized as the measured `tgt_lite` baseline. A stabilized variant now
zero-initializes those columns while retaining the zero pair-value residual.
After local and remote static checks, isolated retry3 job `1314400.ccpbs1` was
submitted at 22:46:36 JST with independent output and log paths. Its initial
state was H `Queue`; no seed-43/44 or Route B job was submitted.

At 23:09:44 JST, `1314400.ccpbs1` had completed normally on `ccg011` after all
20 epochs. Its final validation MAE was `0.1441357 eV`; fixed-test MAEs were
HOMO `0.1293535`, LUMO `0.1439315`, Gap `0.1822294`, and average `0.1518381
eV`. It missed the replacement gates by `0.0840243 eV` average and `0.0999564
eV` Gap. The stable retry therefore remains rejected; its model, checkpoint,
metrics, and log are retained, and no seed-43/44 or Route B job is authorized.

The retained failures separate the next hypothesis from the compact geometry
branch. `tgt_egt_hybrid` was the best completed 30k preflight (`0.1133859 eV`
average), while `pair_triplet_2d_rich` was the best completed pure-2D preflight
(`0.1186426 eV` average). The next bounded candidate `tgt_egt_rich` composes
those two measured ingredients: multi-scale 2D walk/pair descriptors and
low-rank triplet updates in the heavy-atom view, fused with the ETKDG
pair-state/bond-channel geometry view. The new source passed local static
compilation, remote compilation, PBS syntax, a remote parameter instantiation
(`13,182,884` parameters), and a remote two-molecule finite synthetic forward
after correcting the local-to-global topology-edge conversion. It has not been
submitted until the isolated PBS output path is recorded.

The isolated preflight was then submitted once as IMS job `1314542.ccpbs1`
using `train_tgt_egt_rich_preflight_retry4.pbs`. At 23:25:01 JST it was still
in H `Queue` with no log, checkpoint, or metrics yet; no other GPU job was
active for the account. This remains a seed-42 preflight only.

Retry4 later reached `ccg009` and completed epoch 0 with a finite checkpoint,
but its logged training loss was `nan` and validation MAE was `0.60973 eV`.
At the next training step it exited with a retained CUDA device-side index
assert in `_GeometryPairEncoder._dense_bonds`. Remote inspection confirmed
the epoch-0 model, best state, and optimizer tensors were finite. The failure
was diagnosed as an unguarded out-of-range topology edge in a mixed-size batch,
with AMP NaN as a separate stability signal; no gate decision was made from
this crashed attempt.

The fix adds bounds filtering to geometry bond densification and gives the
rich candidate a lower `2e-4` learning rate with AMP disabled, while keeping
retry4's independent artifacts unchanged. Local and remote static checks plus
a remote finite synthetic forward passed. The isolated retry5 was submitted
once as job `1314619.ccpbs1`; at 23:53:17 JST it was in H `Queue` with no
artifacts yet. It remains seed-42 preflight only; seed 43/44 and Route B are
not authorized.

At 01:34 JST on 2026-08-23, retry5 had completed all 20 epochs on `ccg013` and
written its atomic checkpoint, model, embedding payload, log, and metrics. The
training trace was finite throughout after the bounds/FP32 fix; the best
validation average was `0.1179307 eV` at epoch 18. On the fixed 2,794-row ETKDG
test intersection, HOMO, LUMO, Gap, and average MAE were `0.1023617`,
`0.1176421`, `0.1445250`, and `0.1215096 eV`. The result missed the replacement
gates by `0.0536957 eV` average and `0.0622520 eV` Gap, so the candidate was
rejected for representation quality; no seed-43/44 or Route B job was
submitted.

A read-only payload comparison on the common 2,827-row validation and 2,794-row
test intersections showed that a validation-selected scalar blend of the best
ETKDG hybrid and the rich pure-2D encoder reached only `0.107131 eV` average
and `0.128158 eV` Gap on test. The validation error correlation was about
`0.75`, so the pure-2D branch supplies limited complementarity rather than the
large independent signal required by the gate. This closes the rich-fusion
hypothesis as tested; the retry4 CUDA failure and retry5 failed-gate artifacts
remain retained for diagnosis.

The next bounded hypothesis is `tgt_egt_hybrid_plus`: it retains the measured
`tgt_egt_hybrid` ETKDG encoder and its prediction head as an identity path,
adds a smaller conformer-free `pair_triplet_2d_rich` expert, and learns only a
residual correction from the concatenated embeddings. The base branch is
initialized from the retained `1312784.ccpbs1` model; the residual output is
zero-initialized, so the warm start is behaviorally the measured hybrid before
the 2D expert contributes. This directly tests the payload evidence that
prediction-level blending was better than the rejected convex embedding gate.
The source passed local compilation, remote compilation, parameter
instantiation (`13,666,323` parameters), a remote mismatched-node-count finite
forward, warm-start key matching, and PBS syntax validation. Its isolated
30,000/3,000/3,000 seed-42 preflight uses one ETKDG view, FP32, and a bounded
20-epoch exit; it has no authority to submit seed 43/44 or Route B unless both
preflight gates pass.

After the no-overlap scheduler check, the isolated preflight was submitted once
as IMS job `1315094.ccpbs1` at 01:53 JST using
`train_tgt_egt_hybrid_plus_preflight.pbs`. The first read-only scheduler poll
at 01:54 JST showed H `Run` on `ccg004`; only the known optional
`torch-cluster` warning was present and no epoch/checkpoint/metrics artifact had
yet appeared. This remains seed-42 preflight only.

At 04:14 JST on 2026-08-23, `1315094.ccpbs1` completed all 20 epochs on
`ccg004` and wrote its checkpoint, model, payload, log, and metrics. The best
validation average was `0.1066186 eV` at epoch 18. Fixed-test MAEs were HOMO
`0.0919820`, LUMO `0.1016053`, Gap `0.1254396`, and average `0.1063423 eV`.
This improved on the measured hybrid preflight but missed the replacement gates
by `0.0385284 eV` average and `0.0431666 eV` Gap. The candidate was rejected;
all artifacts remain retained and no seed-43/44 or Route B job was submitted.

The frozen follow-up `tgt_egt_hybrid_frozen` keeps the warm-started hybrid
identity path fixed and trains only the conformer-free rich pure-2D residual
path. Local and remote static checks, frozen parameter counts, and warm-start
key matching passed. After `1315094.ccpbs1` completed and its artifacts were
verified, isolated job `1315582.ccpbs1` was submitted at 04:14 JST with
`train_tgt_egt_hybrid_frozen_preflight.pbs`; the first poll showed H `Run` on
`ccg004`. At 04:37 JST it had completed epoch 4 with best validation average
`0.11197 eV`; its checkpoint and log were present, with final metrics not yet
written. This remains seed-42 preflight only.

While `1315582.ccpbs1` was active, a third isolated hypothesis was statically
prepared but not submitted: `tgt_egt_hybrid_warmblend`. It warm-starts the
measured ETKDG hybrid and the already-trained rich pure-2D expert, initializes
target-specific prediction blending at `0.567`, and adds a zero-initialized
residual using both embeddings and both predictions. The remote model has
`21,220,829` parameters; local/remote compilation, PBS syntax, head shape, and
both checkpoint load checks passed. Its output paths remain isolated and it has
no submission authority while `1315582.ccpbs1` is active.

At 05:05 JST, `1315582.ccpbs1` completed after the patience stop and wrote its
checkpoint, model, payload, log, and metrics. The best validation average was
`0.1119734 eV` at epoch 2. Fixed-test MAEs were HOMO `0.0984198`, LUMO
`0.1054147`, Gap `0.1355124`, and average `0.1131156 eV`; the candidate
missed the replacement gates by `0.0453017 eV` average and `0.0532394 eV` Gap.
It was rejected and all artifacts remain retained.

After a second no-overlap check, the warmblend hypothesis was submitted once
as `1315750.ccpbs1` at 05:05 JST. The first PBS copy contained an incorrect
underscore shim path and exited before epoch 0 with the retained GLIBC/
`torch-cluster` import traceback; no model, checkpoint, payload, or metrics
were produced. The PBS path was corrected and a separate retry2 script with
isolated outputs was statically checked. A read-only scheduler check confirmed
the startup failure was retained and no GPU job overlapped, then retry2
`1315767.ccpbs1` was submitted at 05:10 JST. This remains seed-42 preflight
only; no seed-43/44 or Route B job was submitted.

At 08:04 JST, `1315767.ccpbs1` had completed all 20 epochs on `ccg007` and
wrote its atomic checkpoint, model, payload, log, and metrics. The best
validation average was `0.1005727 eV` at epoch 19. On the fixed 2,794-row
ETKDG test intersection, HOMO, LUMO, Gap, and average MAE were `0.0859962`,
`0.0949967`, `0.1211954`, and `0.1007294 eV`. The result missed the
replacement gates by `0.0329155 eV` average and `0.0389224 eV` Gap. It was
rejected; all retry2 artifacts remain retained and no seed-43/44 or Route B
job was authorized from this preflight.

After a read-only no-overlap check found no active MolGap QM9 GPU job, the
frozen warmblend preflight was submitted once as `1316416.ccpbs1` at 08:06 JST
with isolated output paths. This variant freezes both warm-started encoders
and trains only its `107,090`-parameter blend/residual head within the
`21,220,829`-parameter model. The first poll showed H `Run` on `ccg007`; it
remains a 30k/3k/3k seed-42 preflight and carries no seed-43/44 or Route B
submission authority.

At 09:24 JST, `1316416.ccpbs1` had completed all 20 epochs on `ccg007` and
wrote its checkpoint, model, log, and metrics. The warm-start validation
average was `0.10614 eV`; the best validation average after head-only training
was `0.1047327 eV` at epoch 13. Fixed-test MAEs were HOMO `0.0902122`, LUMO
`0.1005120`, Gap `0.1259164`, and average `0.1055469 eV`. It missed the
replacement gates by `0.0377330 eV` average and `0.0436435 eV` Gap. The
head-only frozen warmblend was rejected; its logs, checkpoint, model, and
metrics remain retained, and no seed-43/44 or Route B job was submitted.

## Retained-payload calibrator

The retained 30k/3k/3k payloads showed that a prediction-level stack could be
tested without another graph-encoder run, provided that the calibrator used
only the train split of the retained seed-42 payloads. The new
`payload_stack_calibrator` therefore concatenates the warmblend embedding with
predictions from six retained experts, standardizes using train rows only, and
adds a zero-initialized 128-channel correction to the warmblend prediction.
The implementation writes atomic checkpoint, model, payload, and metrics
artifacts. It does not use the overlapping 100k payloads, which were excluded
from this experiment because they would leak the 30k preflight validation/test
rows.

Local compilation, `git diff --check`, PBS shell syntax, remote compilation,
remote import-only instantiation, and all six retained-payload existence checks
passed on 2026-08-23. After `1316416.ccpbs1` had completed and a read-only
no-overlap check found no active MolGap QM9 GPU job, the isolated seed-42
preflight was submitted once as `1316923.ccpbs1`. The first observation placed
it on `ccg004` in H `Run` at 09:48 JST with an empty startup log and no metrics
yet. This remains a preflight-only experiment; seed-43/44 and Route B are not
authorized by this result until the predeclared gates are cleared.

At 09:50 JST, `1316923.ccpbs1` had completed all 40 epochs on `ccg004` and
wrote its atomic checkpoint, model, payload, log, and metrics. The best
validation average was `0.0981376 eV` at epoch 6. On the fixed retained test
intersection, HOMO, LUMO, Gap, and average MAE were `0.0852661`, `0.0902548`,
`0.1177084`, and `0.0977431 eV`. The calibrator missed the replacement gates
by `0.0299292 eV` average and `0.0354355 eV` Gap, so it was rejected; all
artifacts remain retained and no seed-43/44 or Route B job was submitted.

The follow-up `payload_multistack_calibrator` retained every row from the
complete pure-2D payloads and added explicit masks for missing ETKDG views. Its
11-source input had 3,134 features and used 30,000/3,000/3,000 union rows while
reporting the 27,877/2,827/2,794 ETKDG common intersection for the gate. Remote
alignment, target equality, compilation, and isolated-output checks passed.
Job `1316955.ccpbs1` completed 40 epochs on `ccg004`; the best common
validation average was `0.0980712 eV` at epoch 1. Common-test HOMO, LUMO, Gap,
and average MAE were `0.0851697`, `0.0900337`, `0.1171055`, and `0.0974363 eV`.
The result missed the replacement gates by `0.0296224 eV` average and
`0.0348326 eV` Gap, so it was rejected. All artifacts remain retained and no
seed-43/44 or Route B job was submitted.

## GPS precision late fusion preflight

The payload calibrators did not recover the information gap, so the next
bounded hypothesis returns to late fusion: train GPS9 and GPS11-160 on the
pure-2D topology view independently, train one ETKDG SchNet view independently,
then train a 256-channel gated fusion head. This is a new 30k/3k/3k preflight
with isolated encoder and fusion outputs; it uses one conformer view rather
than a multi-conformer expansion. Local PBS syntax and repository checks,
remote PBS syntax and `qm9_screen` compilation, and the no-overlap scheduler
check passed on 2026-08-23.

After `1316955.ccpbs1` completed and failed the multistack gate, the late-fusion
pipeline was submitted once as `1316994.ccpbs1`. It completed on `ccg001` with
GPS9 test average `0.1206445 eV`, GPS11-160 `0.1214834 eV`, and one-view SchNet
`0.1368549 eV`. The 256-channel gated late-fusion head reached common-test
HOMO `0.0921368`, LUMO `0.0954937`, Gap `0.1264117`, and average `0.1046807
eV`, improving the GPS9 baseline by `0.0131631 eV` but missing the replacement
gates by `0.0368668 eV` average and `0.0441388 eV` Gap. All stage and fusion
artifacts remain retained; no seed-43/44 or Route B job was submitted.

To isolate the remaining conformer-noise hypothesis without adding a third
view, a two-view follow-up was prepared. It reuses the completed GPS9/GPS11-160
pure-2D payload, builds one additional ETKDG seed-43 cache in resumable shards,
trains a single SchNet on the paired seed-42/43 views, averages its two output
views, and retrains the same 256-channel gated head. Local and remote static
checks passed, and the seed-42 payload plus raw-QM9 input checks confirmed the
pipeline would not overwrite prior outputs.

After `1316994.ccpbs1` completed, the isolated two-view preflight was submitted
once as `1317128.ccpbs1` at 10:30 JST. Its first observation placed it on
`ccg005` in H `Run`; the seed-43 ETKDG construction had begun and only the
known `torch-cluster` warning was visible. This remains a seed-42 preflight
only; no seed-43/44 encoder confirmation or Route B submission is authorized.

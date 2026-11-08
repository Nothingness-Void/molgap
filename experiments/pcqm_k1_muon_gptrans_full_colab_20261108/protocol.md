# K1 batched Muon and GPTrans full-train preparation

2026-11-08 JST. Desktop-owned Track B. The user requests two frozen full
training packages and a Colab notebook that can run them without writing more
training code. Local preparation/data staging is authorized now. The user
starts the A100 session through the delivered notebook; this task does not
start cloud training or official evaluation.

## Evidence and question

The completed saved-prediction diagnostic at immutable owner3a3a95337873edb2ecfffd42df1af397dc8c8895,
experiments/pcqm_k1_muon_gptrans_fusion_500k_20261108/, used fixed equal weights.
It reports0.096338391495eV, improving on GPTrans alone by3.648986meV
(paired95%[3.299123,3.992140]) and the existing two-GPTrans equal blend by
1.186474meV ([0.848935,1.525876]). The >=8eV tail worsens, so retain that
diagnostic observation when a separately authorized full-evaluation plan is made.

Question: do the two accepted500K recipes yield useful full-training experts
for a fixed50:50 fusion? The old full K1/GPTrans fusion used different recipes
and20M presentations and closed without promotion. It is not this experiment's
new full result. Do not reopen current_read, add a teacher, pretraining, extra
seed, new addon or reference model.

## Fixed full data and normalization

Reuse the accepted full official-train cache:3,378,606 rows [0,3378606), OGB
atom9/bond3/RWSE16, Gap eV, topology only. Canonical manifest SHA256
7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d;
topology aggregate be18a9964754c8428b5c62433136bd0bafb6c75016811c79e223aec30f3dbe98.
Its68topology files total10,681,142,468bytes. Copy those files only, preserving
their exact hashes and relative paths. Do not export geometry, official-valid,
test, raw archives, other stores or remote output directories.

The accepted full K1 completion manifest at
experiments/pcqm_k1_gptrans_full_fusion/results/accepted_k1_gptrans_fusion_r3/k1/completion_manifest.json
binds that same full manifest/row count and provides full-train target mean
5.689459009120812eV and sample-std1.1621397733722936eV. Reuse those exact
observed constants; do not use500K constants or recompute from development.
The existing loader checks actual full input and statistics before GPU work.
Canonical JSON bytes reproduce the already accepted7d358a67 identity without
changing any original manifest or shard evidence.

There is no internal development role in this full view. All official-train
members, including earlier screen-development rows, now belong to TRAIN. Final
weights are fixed by budget rather than selected on the screen-development set.
Official validation/test/shadow roles remain outside this launch.

## Two independently frozen single-device recipes

Shared: fresh seed42 initialization from retained original state; physical
BS128/drop-last, one supervised forward, FP32/noTF32, Torch2.11.0,
torch-pass-randperm, normalized Gap L1, clip1, deterministic algorithms,
checkpoint128, finite check50, exact resumable chunks, standard epoch logging.
Keep accepted model shape and module modes unchanged.

- k1_muon: neural_atom_k1/3 reference; k1-muon-batched-adamw-fp32-v1;
  LR4e-4, WD1e-5, cosine_epoch60 to1e-6, no warmup; EMA.999 tracked;
  fixed final-live output. No Clean-BN or read side path.
- gptrans_pairnorm: gptrans_t/2 with registered
  gptrans_degree_bond_local_pair_update_norm/1, mode
  degree_bond_local_pair_update_norm_ema999; LR1e-3, WD.05,
  four-pass warmup and cosine through60 to1e-6, EMA.999;
  fixed final-ema output. No precision, architecture or optimizer search.

Both complete60 full passes:26,395 updates/pass,1,583,700updates and
202,713,600sample presentations. Each pass drops46rows under the unchanged
sampler contract. This intentionally preserves the60-pass recipe horizon on
the full data, not the old20M-presentation full contract. No early-stop rule is
inferred from the previous500K curves. Display TRAIN metrics, progress and ETA;
do not fabricate a development MAE where no development role exists.

## Colab execution and durability

One A100, two separate single-arm Specs/packages, sequential execution. Reuse
prepare-family-training and experiment_training_worker preflight/train. Native
qualification must verify exact full data, runtime, model, optimizer, RNG and
actual serialized resume state before formal updates. No Kaggle/T4 launcher,
local-host receipt lock or historical full trainer is substituted.

A notebook session has a10h entry-wall cap and300s cleanup reserve. The
platform adapter sends cooperative SIGTERM before that cap, leaving the shared
runner's PAUSED checkpoint/trace/roles/cost state on Drive. Restart only the
same package and state; keep the original full horizon/cosine, sample counters,
optimizer, scheduler, EMA, RNG and accumulated metrics. The user's subsequent
notebook sessions may continue that finite plan; there is no autonomous cloud
session creation, monitor bridge, successor question or extra training.

Input topology uses local SSD after exact hash checks. All unique training
outputs and checkpoints go directly to Drive. Existing AllocationLedger records
physical device windows, including idle assigned time, once per actual window.
Keep A100 seconds, process wall, queue and pre-entry provisioning distinct;
unmeasured A100 throughput/cost stays unknown. T4 speed is not an A100 forecast.
Release the Colab runtime at normal completion, controlled pause or cell error.

## Acceptance boundary

Final-only output retains native proof, final selected weights, checkpoint,
complete canonical exposure/trace, source/data bindings and observed train-role
use. Use existing family CPU inspection and V5/RML closure. Do not create empty
development predictions or claim a full benchmark score from training loss.
Official full scoring/fusion comparison is a later separately frozen evaluation
with its actual role authority. Existing EdgeState remains the operational
full-scale reference until that decision. This preparation is not acceptance.

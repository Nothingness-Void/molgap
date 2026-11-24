# Current-layer read under batched Muon: 500K protocol

2026-11-07 JST. Desktop-owned Track B, Kaggle1 `nothingnessvoid`. Compare
original K1 (`muon`) with `current_read`, using the same newly registered
`k1-muon-batched-adamw-fp32-v1` in both arms. Registration is present through reviewed infrastructure commit
`dd9dbb42`, cherry-picked as `a99bccd3`; the actual builder generates its policy. This pair isolates the read intervention under one
shared execution identity, not the effect of batched versus unbatched Muon.

## Evidence and question

Read-only RML `lookup --mechanism current` and `lookup --mechanism muon` against
desktop and `codex/exp/k1-read-calibration-100k-20261107` preceded this plan.
The canonical calibration at commit3659ef15, directory
`experiments/pcqm_k1_read_calibration_100k_20261107/`, reports strict causal
comparison, native qualification and two qualified replay entries. Original
Muon selected MAE0.132520667662eV versus current0.130144503669eV: gain2.376164meV,
candidate-minus-reference paired95% interval[-3.122626,-1.679513]meV, final
gain2.259254meV and last-five gain2.791223meV. It is POSITIVE_BELOW_GATE:
the frozen3meV threshold failed. The user's new explicit500K authority releases
this exploration without relabeling that100K outcome or claiming a passed gate.

The history toggle's canonical decision at
`experiments/pcqm_k1_history_read_100k_20261106/decision.md` on its owner reports
only0.103962meV selected improvement and an interval spanning zero. Do not stack
history. Copy only the unchanged `k1_history_read.py` from calibration commit
3659ef15 (SHA2562e712058fe694652c38ac230df036098264c1bf40b036d8fba7adebe3d08d80c)
and its `k1_current_read/1` family3 catalog contract; no whole-branch merge.

Question: does the simpler current-only side path retain a material benefit at
500K after a matched60-pass schedule? Competing explanations are exposure/cohort
contraction and an effect specific to this seed or execution. The cheapest
authorized scale discriminator is this one matched pair, not a recipe grid.

The local synthetic RTX5060 shared FamilyKernel short-training observation (~1.85x) is not a T4
end-to-end speed measurement or bit-exactness claim. Batched arithmetic needs
its own actual-runtime qualification; prior unbatched Muon proof is contextual.

## Fixed scientific conditions and bindings

Reuse the accepted fixed500K dataset consumed by
`codex/exp/k1-muon-500k-20261106` at afa66f98, directory
`experiments/pcqm_k1_muon_500k_20261106/`: TRAIN[0,500000),
development[500000,550000), pure2D/direct PCQM Gap, OGB atom9/bond3/RWSE16.
Dataset manifest SHA256630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751.
TRAIN-only mean5.413434982299805/std1.17672860622406eV.
Development source-index SHA2564299750a50b2a3bd4f03f47f946e7ef15598aaffa13bbbdb416d433a001f37b2;
target SHA2565a8ef02a608f73dad257269d253d5c507b4e3aa480ce4c452b79e4384095b9a2.

Generate a local metadata-only projection of that accepted data declaration:
preserve dataset/feature/split/row/target/transform identities, replacing only
the consumed control-recipe binding as required. Do not re-export data or
rewrite a data hash to describe a changed recipe. Retain an explicit same-run
500K target manifest; do not inherit the100K calibration target manifest.

Both fresh seed42 arms share original state
8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd.
Declare full `neural_atom_k1/3` models: reference addons=[] and candidate
addons=[k1_current_read/1], whose fixed seed42/history=false contract mounts
the existing side path at layers3,6,9. No pretrained initialization or history.
Use one self-contained control recipe generated through the registered builder;
do not override the reference optimizer through shared input training fields.

One supervised forward; physical BS128/drop-last, torch-pass-randperm,
FP32/noTF32, Torch2.11, normalized L1, clip1, LR4e-4/WD1e-5, no warmup,
cosine60passes to1e-6. Complete234360updates/29998080presentations per arm.
Select best internal-development Live; track EMA.999 separately. No Clean-BN
selection, extra view, batch/precision change or optimizer reset. Batched Muon
keeps the registered FP32 NS5, momentum/Nesterov and parameter-partition contract
in both arms; its generated policy is the execution authority.

## Gate, attribution and roles

Nomination requires selected current-minus-reference MAE<=-0.003eV,
2000-resample paired95% upper<0, final delta<0 and last-five matched-pass mean
delta<0, plus strict/native/replay acceptance of both completed arms. Report
complete exposure-aligned traces, selected/final states and finite aligned50K
predictions. Row uncertainty is not training-seed variance. One seed is a bounded
exploration; no claim of optimizer-family tuning or automatic full/adoption.
Persist terminal attribution before another unrelated route.

Only TRAIN and the fixed internal-development role are authorized. Official
validation, shadow and tests remain sealed. Observed provenance records actual
use; authorization alone is not an observed role ledger.

## Execution, durability and cost

One arm per T4 in the same T4x2 session. Standard epoch logs, existing atomic
checkpoint cadence128 and resumable chunk retention. Set pause_after_step234360
directly; do not impose20/40-pass pauses. Scientific horizon stays60 throughout.

Each physical session: entry-wall<=9h, cleanup reserve180s. At most two physical
sessions, cumulative entry-wall<=18h and allocated T4-device time<=36h. These
are ceilings, not measured costs or a promise to finish in one session. Native
T4 duration is unknown. If wall-limited, resume only the exact retained state,
including optimizer/scheduler/RNG/EMA/counters, within the remaining total budget;
do not reset a9h allowance independently of the cumulative cap. Count the shared
allocation once, including any assigned idle device, and keep queue/wall/device
units separate. Unexpected numerical/identity/role errors stop for diagnosis.

Before publication, pin source/config/input/cache and CPU-cache acceptance,
prospective trajectories, actual account quota and native execution checks.
Retain independently retrievable selected/resume states, exact receipts and
job/source versions on D:. Reconcile unknown remote effects in their original
journal before any successor. Existing submit/retrieve/resume/accept/closure
owners apply; no new polling or trainer. The original unbatched500K job is
unrelated and must continue unchanged. No protected role or successor is released.

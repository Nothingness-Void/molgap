# Parameter-free deep local-update control: frozen single-mechanism protocol

On 2026-10-07 the user authorized continuing the bottleneck-led research with
one seed42 fixed100K screen on Kaggle2. No extra seed, grid, baseline rerun,
500K training, protected role, teacher, geometry or automatic successor is
authorized. The primary control is the accepted `degree_bond_local_ema999`
endpoint and its native60-epoch trace, not the smaller corrected G1 model.

## Hypothesis and cheapest falsifier

The [accepted frozen audit](../pcqm_gptrans_bottleneck_audit/decision_terminal.md)
found connected local branches and retained frozen-weight improvement on later
fixed500K development molecules. The 100K early advantage nevertheless narrowed
as deep local update/input RMS increased. These are correlated observations;
they do not establish an optimization or generalization cause. Existing degree,
capacity, decay, path and final-readout experiments are not reopened.

Test whether preventing unusually large deep local updates preserves useful
local information without extra learned capacity. Alternatives: amplitude may
not be causal; bounded updates may remove useful signal; additional fitting
freedom or baseline catch-up may explain late erosion. This one screen changes
the decision to retain or close exactly this local-control hypothesis.

## One changed mechanism

Reuse all 5,871,201 local-bond parameters, every frozen seed42 initial tensor,
all12 core blocks and local branches. Layers1–3 remain byte-equivalent in
computation. Before each core block4–12, for each molecule independently:

`Delta_bounded = Delta * min(1, 0.25 * ||H_real||_2 / max(||Delta_real||_2, 1e-12))`.

Only real atoms enter both norms. Virtual/padded updates are zero. Equal counts
make this exactly the update/input RMS ratio. Norms and clamping are FP32;
gradients flow through them, without stop-gradient. Zero input permits zero
update; zero-initialized branches stay exact identity with finite gradients.
Training and inference use the same operation. No learnable gain, new RNG,
attention, normalization, readout, target, optimizer or schedule is added.
Cap0.25 is a prospectively chosen falsifier, not a fitted optimum. It is above
the audited deep-layer average while constraining outliers; no cap sweep.

[DeepNet](https://arxiv.org/html/2203.00555), sections3–4, motivates separating
residual magnitude from gradient magnitude. Its actual mechanism is scaled
Post-LN plus architecture-dependent initialization. This experiment does
neither and is not a DeepNorm reproduction or proof of its theory on GPTrans.

## Unchanged contract and evidence

Accepted fixed PCQM100K train[0,100000), internal development[100000,150000);
OGB9/3, SPD20, direct normalized Gap L1; FP32/noTF32, physicalBS128, drop_last,
seed42, AdamW foreach=false single group lr0.001/wd0.05, gradient clip1,
warmup4/cosine60/minlr1e-6, EMA0.999 every step, minimum EMA development MAE,
46,860 optimizer steps and5,998,080 sample presentations. Reuse the portable
train-only target transform and identical initial tensor SHA
`a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b`.
Scientific-purpose `architecture_comparison`; only architecture identity varies.

Shared release verifies actual local-reference artifact pointers/hashes and
enrollment. Remote optimizer-inclusive deterministic preflight precedes
training; actual runtime/software remain provenance. Each epoch atomically
retains full state/RNG/cursor, live and EMA metrics, step/presentation counters,
native trace and checkpoint identity; every10 epochs an independent checkpoint
chunk. Aligned terminal predictions/targets/rows, per-role usage and full
allocation cost bind saved-artifact acceptance and RML closure. Unknown CPU,
queue and billing measurements remain explicitly missing, never estimates.

First scheduled training batch each epoch records per-molecule raw/bounded
ratios and scales at all9 controlled layers; final-batch local output gradient
norms are separately labeled. This tests control engagement/connection, not
causality or complete training-population frequencies. No hot-loop scalar sync.

## Budget and terminal decision

Request T4; one model isolated to one visible device. The second allocated T4
is idle, not an unauthorized experiment, and both devices count in cost.
Estimate4 wall hours/8 allocated T4 hours; preflight estimated training cap6
hours; physical deadline7 wall hours/14 allocated T4 hours. Infrastructure
failure preserves evidence, reconciles before any retry and never changes the
scientific recipe. No automatic continuation or successor.

Recompute paired endpoint MAE/bootstrap against the accepted uncapped local
reference; compare same-step live/EMA curves and measured cost. Gain must reach
exceed the frozen material policy0.003eV, with a wholly favorable paired-row
95% interval, to support further consideration. This is a
policy gate, not measured seed variability. Below-gate improvement is recorded,
not promoted. A nonengaged cap or disconnected branch weakens interpretation;
it does not justify a cap grid. Reused development selects only; no untouched
holdout or scale/full claim. A full training result requires the real strict
evidence and candidate/reference Replay admission; failures remain incomplete.

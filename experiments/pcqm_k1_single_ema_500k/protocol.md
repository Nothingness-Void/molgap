# Bounded single-forward K1 EMA 500K contract

Frozen question date: 2026-10-10 Asia/Tokyo. Owner: desktop. The user requested
one Colab A100 500K efficient screen and explicitly selected four hours TOTAL,
including both arms, accelerator setup, preflight, calibration and evaluation.
No additional credit purchase, retry, continuation or successor is authorized.

## Question and arms

Can stepwise parameter EMA recover development accuracy in the efficient
single-forward K1 recipe, at matched 500K exposure and clean normalization?
This is not a repeat of the closed 100K single/mean2 noninferiority test.

Reference: one stochastic forward, normalized Gap L1, live parameters.
Candidate: identical optimization plus parameter EMA0.999 after every update.
Both start from the same pinned random K1 state, not pretrained weights.
Both receive the same declared training-member-only, Dropout-off BN calibration
before their primary development measurements. Only EMA differs between arms.
Keep original/raw predictions as secondary diagnostic outputs; never choose
between raw/clean/live/EMA using whichever wins on development.

OGB atom9/bond3/RWSE16, K1 node192/edge64/slot64, nine local blocks, one active
slot at layers3/6/9, unchanged3,658,817 parameters. Pure2D strips cached geometry
before model input and never generates coordinates. No teacher or consistency
penalty. Historical pretrained/double-forward/T4 endpoints are context only.

## Fixed data and optimization

Accepted fixed500K cache manifest SHA256:
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Private mirror: `nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1`.
Training source_idx0:500000; internal development500000:550000,50K rows.
Both are official-train-derived. Development is repeatedly selection-consumed,
NOT independent holdout. Official valid/test-dev/test-challenge/common/OOD/P8-hard
are prohibited. BN calibration uses fixed first16,384 training members with
features only; no labels determine the buffer estimate.

Seed42, FP32/noTF32, deterministic algorithms, physicalBS128, no accumulation,
drop_last32, AdamW lr4e-4/wd1e-5/clip1/foreachFalse/fusedFalse.
Full60-epoch cosine horizon, eta_min1e-6, never compress the schedule to fit
the four-hour screen. Each full epoch is3,906 updates and499,968 presentations.
Full endpoint would be234,360 updates/29,998,080 presentations PER ARM.
Training order is the historical500K randperm seed42+epoch, fixed for both arms.
Target normalization is all500K train-only mean/unbiased std.

## Budget and decision

One A100, arms advanced serially in alternating complete-epoch rounds with
independent RNG/resume state. Compare only equal completed exposure. Reserve
time for atomic checkpointing and release; watchdog stops work before14,400s.
Stage CPU installation/data validation before requesting A100 where possible.
If accelerator setup is required afterwards, count it against the same budget.
Record hardware and full observable allocation interval; unknown billing tails
remain unknown. Never equate worker time, GPU busy time and Colab CCU billing.

No performance-based early stopping is authorized. Preflight failure means
NO_TRAIN; budget cutoff means STOP_FOR_COST. Partial curves provide diagnostics,
not final500K ranking or STRICT_CAUSAL endpoint claims. If both complete60
epochs, primary promotion nomination requires reference-minus-candidate
gain>=0.003eV and positive paired-row95% interval, plus applicable V5 acceptance;
one seed does not measure seed uncertainty. No full-data launch or adoption follows.

## Durability and qualification

CPU stage verifies manifest/shard hashes, declared sizes and roles. Before
training: actual A100 identity, finite first/last training-batch forward/backward,
repeatable optimizer updates, atomic RNG/optimizer/model/EMA resume and initial
tensor SHA roundtrip. Both arms must qualify before either starts formal training.
Preflight exposure is separate from formal optimizer/sample counters.

Freeze source/config/initialization and prospective records before execution.
Write last/best checkpoints, aligned predictions, traces, exact optimizer/sample
coordinates, native cost and role observations atomically to private Drive.
Save calibrated state inside its temporary context; restore uncalibrated live
state and RNG so diagnostics do not perturb subsequent optimization. Interrupted
work preserves last durable state and never pretends to finish a planned endpoint.

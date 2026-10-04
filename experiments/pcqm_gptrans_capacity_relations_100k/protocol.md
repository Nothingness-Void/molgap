# Local relations and retained-route scale study

Frozen on 2026-10-05, server-owned, with the four-physical-experiment authority
and [common budget/recipe constraints](../pcqm_gptrans_capacity_nodes_100k/protocol.md).
Kaggle2 only. This notebook isolates two different questions on two T4 devices.

## Arm1: real-bond local stream (100K)

Keep corrected G1+EMA0.999, all original GPA blocks and the frozen recipe. Add
one 64-channel true-directed-bond sum message before each GPA block, reading
source/target normalized node state and the existing persistent pair state
at the REAL bond position only. Output projection starts at zero. Do not add
all-pair message MLP, new graph cache, RWSE, geometric data, or path preprocessing.

This changes information flow, not only the final readout. The
[readout failures](../pcqm_gptrans_readout_100k/gpu/results/decision.md) do not
test this interior message path. Related K1 local adapters did fail: the
alternative explanation is redundant local information/overfitting, not that
any local module is guaranteed to help. [GPS++](https://arxiv.org/pdf/2302.02947)
motivates a chemical local bypass but its much larger budget/3D experiments
are not evidence for this particular pure2D GPTrans addon.

Match the accepted G1+EMA999 reference under an architecture intervention,
no baseline rerun, strict release/terminal checks and actual Replay admission.
The parameter cap, exposure, gate and stopping rules are linked above.

## Arm2: same-live 500K EMA study

Reuse the accepted byte-identical fixed500K asset, train[0,500000) and
internal-development[500000,550000), NOT official validation. One seed42 live
G1 encoder/AdamW stream maintains EMA0.9999 and EMA0.999. Both views evaluate
the SAME live trajectory and independently select the best of60 frozen
observation rungs. Train from accepted G1 random initialization, no warm start.

Freeze 46,860 updates /5,998,080 presentations, BS128, FP32/no TF32, the same
LR sequence as the100K recipe (781 updates/rung, warmup4rungs/cosine60rungs),
normalized L1 using the SAME accepted train100K-subset transform, wd0.05.
Sampler continuously traverses seed42+cycle permutations of500K, dropping
32 rows per full cycle; it does NOT reset sampling at observation rungs.
This is approximately12 passes, NOT60passes and not full-scale qualification.

The [accepted execution profile](../pcqm_gptrans_scale_qualification/results/interpretation.md)
qualifies runtime feasibility only. No accepted target-scale matched EMA
reference bundle exists BEFORE this study. Therefore prelaunch is explicitly
`transfer_study / PRELAUNCH_NONCAUSAL_PLANNED`, with the100K reference as
context ONLY. Never fabricate a500K comparator, grant STRICT_CAUSAL from
the100K bundle, or claim an accepted Replay pair before both500K views are
actually observed and independently accepted. Raw dual-view traces, predictions,
checkpoint/filter/RNG state, indices, role events and costs must be retained
so subsequent paired/Replay qualification can be independently evaluated;
incomplete qualification remains incomplete, not forced into the pool.

Scientific question: does the EMA999 selection benefit remain directional
after genuine500K optimization at equal optimizer exposure? This isolates
two weight filters within this run but does not identify architecture scale
causality, long-budget convergence, seed variation or full performance.
Use aligned saved predictions, paired row bootstrap and the0.003eV material
gate for interpretation only; record shared live trace equality, curves and
native cost. The repeated development role is selection, not sealed holdout.

The full one-live/two-filter state checkpoints atomically at every rung;
independent ten-rung chunks preserve continuation. No schedule restart.
Minimum15% VRAM reserve; measured qualification must fit6optimizer hours.
Maximum notebook7wall hours /14allocated T4hours; retain partial evidence on
budget expiry rather than changing batch, precision or scientific recipe.

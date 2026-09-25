# PCQM scale-transfer reassessment, 2026-09-25

Question: do newly accepted 500K mechanism endpoints and full continuation
traces support the older RML diagnosis of 100K-to-500K gain erosion and
500K-to-full rank reversal?

This is an offline, source-only postmortem. Use existing RML indexes to locate
canonical contracts, traces, decisions, role records, and measured cost. Do
not load molecular labels, prediction tensors, checkpoints, official-validation
payloads, or any protected role. No training, inference, or remote scheduler
action is authorized.

Round 1: build an exact comparability matrix for 100K, 500K and full. For
each pair, record dataset/split, optimizer steps, sample presentations,
weight semantics, selection rule, and evaluation role. Only same-scale,
same-role, matched-budget pairs can yield causal mechanism gain.

Round 2: align accepted matched-500K GPTrans reference/candidate traces by
optimizer step. Calculate development gain and the difference between online
train metrics at fixed checkpoints. Online train MAE is not a fixed-subset
train evaluation; do not call their difference a generalization gap. Distinguish
observed peak, terminal, and selection-point gains.

Round 3: reconcile accepted full training and continuation budgets. Compare
source and continuation endpoints only within each architecture. Quote the
accepted official-validation rank solely from its existing decision; do not
recompute or subdivide that role. Test whether a convergence plateau is
established under each contract.

Round 4: backtest the earlier STQS/projection claims against subsequently
accepted 500K outcomes. State what hypotheses survive, what is falsified, and
what remains unidentified. Recommend the smallest future matched run only if
it would change a resource decision. Do not activate an early-stop threshold
from these retrospective cases.

Preserve exact source paths and SHA256 for the machine-readable traces used.
Missing facts remain unknown. This audit cannot establish seed variance or
true full-scale architecture ranking under equalized compute.

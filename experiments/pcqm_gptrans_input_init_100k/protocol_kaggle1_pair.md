# Kaggle1 same-allocation input-initialization pair

This amendment answers the same 15-table input-initialization question as
`protocol.md`. The user requested two arms on the available Kaggle1 T4x2
allocation after the P100 and single-T4 planning routes remained unsubmitted.
Its paired comparator and native resource gate supersede the historical
cross-runtime comparison and idle-second-GPU plan in `protocol_t4.md`.

Run the unmodified frozen GPTrans-T seed-42 reference on T4 device 0 and the
`Normal(0, 0.02)` input-embedding candidate on T4 device 1, concurrently in one
kernel. Both load the same SHA-pinned initial-state artifact and accepted
Kaggle1 100K graph mirror. Only the candidate changes the 15 input embedding
tables; all other source, row, target, optimizer, batch, precision, exposure,
EMA and checkpoint semantics remain those of `protocol.md`. This fresh control
is decision-relevant because the historical reference was trained on a P100:
it permits a same-allocation paired effect estimate, not a baseline
bookkeeping replacement. Preserve the old accepted reference as context.

Before either arm trains, both isolated single-T4 processes must pass the V4
finite-step, repeatability, and memory preflight. The candidate must also pass
the alternating same-device baseline/candidate training-step and inference
latency gates: each median ratio at most 1.05. Require at most six estimated
T4 training device-hours **per arm**. An inconclusive or failed preflight stops
both training arms. Record measured device-hours separately from wall and queue
time; parallel execution does not make aggregate device-hours equal wall time.

Each arm trains once for 60 epochs, with 46,860 optimizer steps and 5,998,080
presented examples, then retains its selected and resumable checkpoints,
source-aligned 50K development predictions, trace, cost and role evidence.
The candidate is a 100K shortlist only if its MAE is at least 0.003 eV below
the **accepted same-job reference**, and the paired row-bootstrap 95% upper
bound of candidate-minus-reference error is below zero. The historical
0.1536272043 eV candidate threshold is contextual for this amended comparison.
A failed arm blocks a same-job causal claim; an infrastructure failure is not a
scientific negative. Row bootstrap does not estimate seed variation. No second
seed, 500K, protected evaluation, full-scale training or promotion is released.

The prospective two-arm Spec declares `same_run_replay`; terminal evidence must
bind both arms to the same platform job, attempt, source and package. Replay
qualification and scientific acceptance remain separate decisions.

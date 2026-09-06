# Frozen GraphState Full-Data Contract (2026-09-06)

## Selection and provenance

The user supplied the accepted GraphState three-seed winner while the older
scratch preflight was being diagnosed. The desktop verified its handoff on
`origin/molgap-server` at 3b8975d and its confirmation source at
9068ddb82e6bdf16b841570abbff023b90c07f07. Exactly the three local/global classes
are extracted into `src/molgap/pcqm_graph_state.py`; AST equality to the pinned
source, count 3,665,809, no atom attention, and batched gradients are tested.
Its parent geometry encoder and GPS base remain the existing reusable code.
The old two-arm full-data plan is superseded before training. No server branch
is overwritten or merged wholesale.

## Inputs and gate

Reuse the accepted full ETKDGv3/MMFF94s cache with acceptance SHA-256
650b1fbd14771888ba1c9342cdcba240b420bd91b0dced974ae654a4b65ab9d2.
There are 3,378,606 official-train and 73,545 official-validation rows.
Every consumed shard is checked against its immutable hash and count.
No official test role is opened. The completed P0 audit has independently
verified source-function preservation over all accepted validation rows.

A100 preflight uses only official-train graphs. It checks the exact model
count, finite FP32 loss/gradients, 64 optimizer steps, at least 15% GPU memory
reserve, and 12-epoch runtime projected with 20% overhead within 43,200 s.
Only an accepted preflight releases one full-data training job. A rejected
gate must not silently reduce epochs, change precision or change the model.

## Training

- one randomly initialized GraphState candidate, seed 42;
- node 192, EdgeState64, wedge16, GraphState64, rank32, nine local blocks;
- graph communication at 3/6/9, dropout 0.10, scalar direct Gap head;
- FP32, TF32 off; batch 192; no DataLoader subprocesses;
- AdamW 1.6e-4, weight decay 1e-6, matching the accepted screen's optimizer
  priors; gradient clipping 1; fixed cosine schedule of 12 epochs to 1e-6;
- no pretrained parameters, distillation, residual target, or fusion;
- atomic initialization, shard-boundary last state, per-epoch logs, complete
  model/optimizer/scheduler/RNG and cursor for explicit resume;
- a per-process 11.5-hour stop and cumulative 12-hour cap preserve the last
  completed shard. No automatic schedule restart or horizon extension.

Unlike the superseded paired proposal, training does not repeatedly select
on official validation. The final epoch is frozen, then the official-valid
role is evaluated once with per-shard predictions and hashes. The legacy
`best.pt` filename in the shared runner denotes the fixed final checkpoint
here; metrics explicitly identify `selection=fixed_final_epoch`.
There is no early stopping or validation-derived hyperparameter adjustment.

The historical original-full EdgeState score is a deployment reference with
a different training horizon. This one-model run cannot establish a matched
full-scale architectural causal effect, and 12 epochs do not guarantee
convergence. No leaderboard submission or production promotion is authorized.

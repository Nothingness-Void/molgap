# other_architecture (GINE, recurrent graph state, determinism audit, held-out budget)

## Quick answer
Recent closed (of 4):
TB-xian-determinism-audit-20260915 | NEGATIVE_UNDER_CONTRACT | xian-determinism-audit | bounded-synthetic-full-model-replay-with | null (see experiments/pcqm_xian_determinism/decision.md) | -
TB-recurrent-graph-state-100k-seed42 | NEGATIVE_UNDER_CONTRACT | recurrent-graph-state | shared-recurrent-graph-state-through-the | null (see git://285e1dc/experiments/pcqm_gap_architecture/results/recurrent_graph_state_seed42/decision.md) | -
TB-pcqm-h1-heldout-budget | NEGATIVE_UNDER_CONTRACT | pcqm-h1-heldout-budget | reserve-six-percent-of-a-fixed | +5.83 | endpoint_regression
Open (ACTIVE) 0: -
Reopen: New preregistered design with explicit authority, for example a randomly drawn fit block; no automat / materially new pure-2D information-flow question / new runtime or bounded diagnostic contract
src (src/molgap/): gine.py, pcqm_gap_architecture.py
Experiments: pcqm_gap_architecture, pcqm_gine_expert, pcqm_h1_heldout_budget, pcqm_xian_determinism

## Closed (4)
TB-xian-determinism-audit-20260915 | NEGATIVE_UNDER_CONTRACT | xian-determinism-audit | bounded-synthetic-full-model-replay-with | null (see experiments/pcqm_xian_determinism/decision.md) | -
TB-recurrent-graph-state-100k-seed42 | NEGATIVE_UNDER_CONTRACT | recurrent-graph-state | shared-recurrent-graph-state-through-the | null (see git://285e1dc/experiments/pcqm_gap_architecture/results/recurrent_graph_state_seed42/decision.md) | -
TB-pcqm-h1-heldout-budget | NEGATIVE_UNDER_CONTRACT | pcqm-h1-heldout-budget | reserve-six-percent-of-a-fixed | +5.83 | endpoint_regression
TB-gine-1m-gap-specialist-v7 | POSITIVE_UNDER_CONTRACT | gine-gap-specialist | task-specific-gine-gap-specialist | null | -

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
gine-gap-specialist | gine-1m-gap-specialist-v7 | POSITIVE_UNDER_CONTRACT | null | - | task-specific-gine-gap-specialist
pcqm-h1-heldout-budget | pcqm-h1-heldout-budget | NEGATIVE_UNDER_CONTRACT | +5.83 | endpoint_regression | reserve-six-percent-of-a-fixed
recurrent-graph-state | 100k-seed42 | NEGATIVE_UNDER_CONTRACT | null (see git://285e1dc/experiments/pcqm_gap_architecture/results/recurrent_graph_state_seed42/decision.md) | - | shared-recurrent-graph-state-through-the
xian-determinism-audit | 20260915 | NEGATIVE_UNDER_CONTRACT | null (see experiments/pcqm_xian_determinism/decision.md) | - | bounded-synthetic-full-model-replay-with

## Reopen conditions
New preregistered design with explicit authority, for example a randomly drawn fit block; no automatic scale-up, no protected-role access, no promotion from this 100K pilot.
materially new pure-2D information-flow question
new runtime or bounded diagnostic contract

## src
src/molgap/gine.py
src/molgap/pcqm_gap_architecture.py

## Experiments
- experiments/pcqm_gap_architecture: scripts accept_pcqm100k_cache.py, accept_pcqm100k_local_operator_search.py, accept_pcqm100k_novel_seed42.py, accept_pcqm100k_recurrent_graph_state.py, accept_pcqm100k_seed42.py
- experiments/pcqm_gine_expert: scripts accept_pcqm_expert.py, continue_pcqm_gine_local.py, scale_pcqm_gine_local.py
- experiments/pcqm_h1_heldout_budget: scripts prepare.py, score.py
  - trace experiments/pcqm_h1_heldout_budget/worker_rml/rml_finalized/trace.json
- experiments/pcqm_xian_determinism: scripts accept_terminal.py

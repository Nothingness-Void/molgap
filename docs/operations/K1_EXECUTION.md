# K1 Execution Selection

This page owns selection instructions, not an old contract or submission release.
For new compatible single-forward `neural_atom_k1/2` declarations, select
`k1_initialized_layout/1`. The registered family trainer owns execution:
CPU dense-layout reuse and scoped TRAIN safety-fill removal, native AdamW,
unchanged model/loss/batch/precision/schedule. Deterministic algorithms stay on;
the fill setting is restored after each step and clean evaluation stays common.
Do not copy the trainer or combine this addon with another unsupported addon.

## Reusable entry

```python
from molgap.experiment_execution import (
    build_family_recipe,
    RECOMMENDED_K1_EXECUTION_ADDON,
    RECOMMENDED_K1_EXECUTION_TORCH_VERSION,
)

addon = RECOMMENDED_K1_EXECUTION_ADDON
recipe = build_family_recipe(
    ("neural_atom_k1", "2"), addon=addon,
    source_idx_sha256="<frozen-source-index-sha256>",
    target_sha256="<frozen-target-sha256>",
    execution_torch_version=RECOMMENDED_K1_EXECUTION_TORCH_VERSION,
)
```

Freeze the recipe and declare the same addon/version/config/source SHA in the
Spec; use [workflow](EXPERIMENT_WORKFLOW.md) and [CLI](EXPERIMENT_CLI.md).
The runtime pin is explicit, not an instruction to install/downgrade Torch.
An unqualified runtime fails closed. Existing native all-arm qualification and
exact reference/resume checks are mandatory before formal training. Different
software/hardware/input contracts need their applicable qualification. This
screen interface is not permission or an implementation for500K/full training.

`addon=None` deliberately remains the original reference path. Frozen recipes,
existing jobs and checkpoints are not rewritten. Reference fallback for a new
task requires its declared contract; a failed preflight never silently changes
policy or launches a retry. This execution selection changes neither production
models/registry nor scientific or compute release gates.

## Dated selection authority

2026-10-12 JST: the user explicitly selected the latest initialized-layout
implementation after reviewing the native paired100K/40epoch result.
This is engineering selection despite a failed nomination gate, not a claim
that the frozen >=10% speed gate passed. Retained result: TRAIN time decreased
7.31237%, full epoch pipeline time5.83105%, with byte-equal50000 predictions,
selected/final model and final optimizer/RNG state. That one run does not prove
all graphs, scales, runtimes or hardware behave identically. Fused AdamW is not
the selected policy.

Evidence owner: `codex/exp/k1-fused-layout-t4-100k`, native kernelID138206744,
version1, executable source `cd615aa4716a43f984de8223ab50b137d4b45eab`.
Follow its [result inspection](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/initialized_layout_100k/release_002/terminal_acceptance/decision.md),
[detailed review](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/initialized_layout_100k/release_002/terminal_acceptance/detailed_review.md)
and [selection record](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/initialized_layout_100k/adoption.md).
Formal terminal RML closure stays on that owner and remains pending; selection
does not fabricate missing role/cost/comparison-readiness evidence or replay-ready
status. Only reviewed reusable execution/transport helpers are integrated here,
not the full experiment branch or server work.

## Integration verification

Reusable source was selected from owner `d601daa1` onto desktop `2d1202ee`;
no frozen experiment/source package was rewritten. Local CPU regressions first
reported416 passes,1 host-limited skip and1 missing branch-local test template.
That test now reuses existing declaration fixtures; targeted rerun2 passes and
the complete initialized-execution/Spec test files135 passes. These batches
overlap and are not a single summed test count. No GPU training/inference ran.
Desktop RML frozen check passes in the primary checkout; an isolated checkout
without existing ignored historical checkpoints cannot perform that check.
This does not qualify pending terminal records or whole-repository portability.

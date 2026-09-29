# Structural residual follow-up — 2026-09-29

## Scope

This was a read-only derivative of already accepted internal-development
prediction artifacts. No model was trained or inferred, no new role was opened,
and no new RML training trajectory is claimed. The frozen cross-scale terminal
and 100K aligned prediction matrix remain the source authorities. The exact
inputs, payload hashes, fixed structure groups and numeric reductions are in
[the machine report](results/structural_residual_audit.json); the calculation
is reproducible with `python -m
experiments.pcqm_k1_cross_scale_frozen.analyze_structural_residuals` using the
project venv. The script verifies the matrix hash, every cross-scale chunk
against the accepted terminal, both new-family payload hashes, row IDs,
targets and K1 prediction reproduction before calculating effects.

Positive gain below means K1's absolute error minus the candidate's error.
All numbers are descriptive on repeatedly used internal development roles.
These structure groups were selected for diagnosis, not as a prospective
router or independent validation.

## What the retained predictions establish

| Candidate | Original 100K development gain | Frozen 500K development gain | Evidence boundary |
|---|---:|---:|---|
| MetaGIN2D | -0.022814 eV | unavailable | Negative across every tested atom-count and conjugation stratum; not a specific missed subgroup |
| One-shot motif K1 | -0.000342 eV | unavailable | Original-role subgroup signs vary; no cross-role portability claim |
| PairToken | +0.003044 eV | +0.000416 eV | Same frozen 100K weights; most selected gain disappears with the role change |

The group pattern is not a stable blueprint for another relation module:

- PairToken's gain for at most 12 atoms was `+0.005429 eV` on 12,981 original
  rows but `-0.002196 eV` on 7,455 later rows. For conjugated-bond fraction at
  most 0.25 it was `+0.005409 → -0.003098 eV`. These are changes in model
  advantage within inference-visible groups, not just group frequencies.
- The high-conjugation group above 0.75 retained a smaller frozen gain
  (`+0.004255 → +0.002352 eV`), but the separately accepted matched-500K
  training result for that group was `-0.001017 eV`. It is not a stable
  cross-training architecture advantage. A joint high-atom/high-conjugation
  cell even changed from `-0.000898` to `+0.006803 eV` under frozen weights;
  the apparent rule depends on the inspected population.
- MetaGIN2D was negative even in the high-conjugation group (`-0.032936 eV`)
  and the at-least-16-atom group (`-0.022764 eV`). Its 100K terminal already
  showed continued training fit without comparable development gain. This
  supports a broad failure of this *adapted screen*, not a proof about the
  paper's full-scale model.
- Motif K1 helped the original high-conjugation group by `+0.002812 eV` but
  worsened the at-least-16-atom group by `-0.001295 eV`; its whole-role effect
  was below the original gate with an interval spanning zero. No frozen500K
  predictions exist for this exact checkpoint. Treat the subgroup signal as
  unconfirmed and do not infer that motif exchange scales.

## Decision

The audit did **not** identify a structure-defined deficit that is both
portable across roles and retained after matched 500K training. It does not
release a Router, a MetaGIN/motif retry, a path/slot micro-variant, a new GPU
screen, an extra seed, or protected-role access. The earlier router audit also
found poor inference-visible identification of individual expert winners.

The remaining high-value question is whether a different information flow can
retain benefit under matched exposure, but the present saved endpoints cannot
select that mechanism causally. Any later proposal needs a distinct
prospective hypothesis, an unconsumed or appropriately qualified role,
calibrated training-run uncertainty, native cost and a one-mechanism design.
The separate GPTrans author-alignment P0/G1/G2 plan already owns paper-parity
questions and must not be duplicated by this K1 follow-up.

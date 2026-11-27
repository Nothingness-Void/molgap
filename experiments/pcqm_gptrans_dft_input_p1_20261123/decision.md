# GPT-T with DFT distances as a training-time input (P1): below the gate

2026-11-28, Track B, 3D-supervised training with pure-2D inference (separate
contract from strict 2D). IMS run `gptdftp1-chain-20261123` (jobs 2146337,
2161254, 2187128) finished COMPLETE: six 500K items, both pairs. Rules:
[protocol.md](protocol.md), "Gates". Closure built with `rml legacy-terminal`
and `terminal-pipeline` (no hand-written glue).

## Verdict

Family P1: **POSITIVE_BELOW_GATE**, failure mode `below_gate`. Gates (i), (iii)
and (iv) pass. Gate (ii) does not: P1 beats the pure-2D control by a small,
consistent margin, but less than the 2.0 meV line. Nothing is adopted. No tail
score, official valid or test read is released. P2 and P6 are not authorised.

Arm notation: P1 = POSITIVE_BELOW_GATE. Control arms (s42, s43) = CLOSED.
Zero-input arm (ablation) = CLOSED, no candidate conclusion.

## Development MAE (rows 500,000..549,999, final EMA, meV)

| Item | 2D | 3D-only | 2D+3D |
|---|---:|---:|---:|
| ctl s42 | 100.713 | - | - |
| P1 s42 | 99.680 | 74.816 | 74.006 |
| zeros s42 | 102.456 | 105.515 | 102.453 |
| ctl s43 | 100.593 | - | - |
| P1 s43 | 99.367 | 74.038 | 74.098 |
| zeros s43 | 101.731 | 105.821 | 101.738 |

## Gates

Paired 2,000-draw bootstrap over the 50,000 dev rows (seed 20261123). It
measures row noise only, not init or seed variance. Numbers are copied from
`evidence/dev_analysis_s42.json` and `evidence/dev_analysis_s43.json`.

| Gate | Requirement | s42 | s43 | Result |
|---|---|---:|---:|---|
| (i) P1 3D-only minus P1 2D | <= -10 meV, both pairs | -24.86 | -25.33 | pass |
| (ii) P1 2D minus control 2D | mean of pairs <= -2.0 meV, negative in both | -1.03 | -1.226 | **not met** (mean -1.13) |
| (iii) P1 2D minus zeros 2D | <= -1.0 meV, both pairs | -2.78 | -2.36 | pass |
| (iv) P1 over control: median step time / total wall clock | <= 1.5x / <= 2.0x | 1.032 / 1.040 | 1.058 / 1.073 | pass |

95% intervals, gate (ii): s42 [-1.593, -0.464], s43 [-1.726, -0.704]. Neither
pair interval crosses zero. The two-pair mean (-1.13) has no bootstrap entry,
so it is described here only; the closure facts carry the single s43 pair.

Gate (iv), zeros over control (descriptive): step 1.054 / 0.981, wall 1.059 /
1.008 (s42 / s43). Total item wall, all six items: 176,213.6 s (48.95 h).

Zeros arm against control: +1.75 meV (s42), +1.14 meV (s43), worse than the
control. Reading: modality dropout plus masking costs about 1.4 meV by
itself, the geometry signal returns about 2.6 meV, net about -1.1 meV.

## What the numbers do and do not say

- "3D mode beats 2D mode by about 25 meV" is diagnostic evidence for gate (i)
  only: the model can use true distances when given them. Inference is pure
  2D. It is not a model score and goes into no ranking.
- The record does not separate "the 2D path is undertrained" from "the 32-channel
  pair state has too little capacity". Neither is chosen here.
- There is no 100K companion, so no reading of "scale erosion" is made.
- Two pairs differ in init and seed; row bootstrap is not seed noise.

## Role use (structural evidence, Opus ruling)

- `official_validation`, `test_dev`, `test_challenge`: **untouched**. The
  training code bound at the commit below has no code path that opens official
  valid or test. Both offline score files record `protected_roles_read` all
  false. The recipes set `development_rows: 0`, so the job reads only the fixed
  500K train store (rows 0..499,999). The `run_provenance.json` of four items
  (ctl s42, ctl s43, P1 s42, zeros s42) records `tail_holdout_v1_read`,
  `official_valid_read` and `test_read` all false. The s43 P1 and zeros items
  have no retrieved provenance file; their evidence is the s43 score file, the
  final serial summary and the same commit and spec hash in the retrieval record.
- `tail_holdout_v1`: **untouched**. Tail rows (>= 3,250,000) are outside the
  training shards and were not scored.
- `dev_500k` (rows 500,000..549,999): **consumed**, offline scoring only
  (labels, DFT coordinates as scoring input, MAE). Events follow the
  dft-distance-aux precedent and carry `selection_used: true`: the pair-42 dev
  scores gated the resume of the s43 jobs under the protocol's early-stop rules
  (a go/no-go use, not best-epoch selection; the readout is the final EMA).

## Binding and custody

- Training commit `cc822c1d0cf3a82377698852a181669634fae213` (named by
  `serial_summary.json`, the retrieval records and `run_provenance.json`). The
  trajectory froze `7ae12db48...` at planning; `src/` and `platforms/` differ
  between those commits (build commits). File hashes: `evidence/source_binding.json`.
- Spec sha256 `50c254287e4f9289bbdd1b94a4ca7040f73f052a302474abba26ee8a3018b09a`;
  source archive sha256 `4d4688cde634e0aa00a70c01e6083bcea2a56c49eb37893b1e508f605ea3f7cb`.
- Retrieved checkpoints stay out of the repo, at
  `D:/molgap-artifacts/ims_retrieved_20261125/p1/<item>/` (pair 42; last
  checkpoints too) and `D:/molgap-artifacts/ims_retrieved_20261128/p1_s43/training/<item>/`
  (all six). `selected_model.pt` sha256:

| Item | sha256 |
|---|---|
| ctl-500k-s42 | 35b4a1c7cb4322a9cd06111247ee3fc189ef6c09655d476cbef10e9ad5313292 |
| p1-500k-s42 | 1a4ee0c38f1a1892211549826cbf9e13f8fba50009f1a95ee6f83ed48c59d068 |
| zer-500k-s42 | 7dd7d024a2b4857e54ba0792a6c6704325b7191bad02e69bc8c726c72af88da8 |
| ctl-500k-s43 | cc49f8280f1766cf35d008f9f7cbd6f22d7b0ab73a500915995a84b97ba9037a |
| p1-500k-s43 | bce79c77c17be77866dcaabc95c099a0da3b5c004c12b21bf424e6f20d80cb96 |
| zer-500k-s43 | 7607f0c98c3e5d83df1b05f77ea769fd194a91613cae1343e6544b1db6b24dbd |

- Score files: `D:/molgap-artifacts/ims_retrieved_20261125/p1_score/dev_analysis.json`
  and `D:/molgap-artifacts/ims_retrieved_20261128/p1_s43_score/dev_analysis.json`
  (copies in `evidence/`). Scorer: `score_p1.py` (hash in `source_binding.json`).
- Cost: one measured A100 wall figure (sum of per-item wall over attempts); CPU
  and queue hours are not measured.

## Open problems

- 2D-path undertraining versus pair-state capacity: not separated.
- No 100K companion: transfer across scale not assessed.
- 3D-only mode keeps topology through degree encoders and BondLocal messages
  (protocol caveat).

## Not authorised

P2, P6, a 100K companion, a tail score, an official-valid read, a test read,
and any adoption. Reopen only with new explicit user authority.

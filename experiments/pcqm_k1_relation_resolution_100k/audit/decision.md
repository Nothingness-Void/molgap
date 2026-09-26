# Frozen-checkpoint relation portability audit — September 27, 2026

## Decision

The three exact relation-resolution interventions were closed without 500K
training, extra seeds, full training or a successor. Their favorable but
below-gate original-development training records were preserved unchanged;
this separate NO_TRAIN terminal added negative portability evidence, not a
retroactive rewrite of training outcomes.

The Kaggle2 audit `kaseichou/molgap-k1-relation-audit-s42:v1` completed and
passed saved-tensor acceptance. The original predictions reproduced within
7.63e-6 eV for the reused K1 reference and 1.91e-6 eV for every newly inferred
candidate. Checkpoints, frozen source, transforms, target/row alignment and
all atomic chunk hashes matched. The accepted K1 prediction chunks were reused
unchanged, without another reference inference or reference training.

## Evidence

| Candidate | Original-role gain vs K1 | Frozen500K-role gain vs K1 | 95% paired row CI, candidate minus K1 on frozen500K |
|---|---:|---:|---|
| Receiver-pair | +0.001866 eV | -0.002190 eV | [+0.001397, +0.003025] eV |
| Triplet-aggregate | +0.001516 eV | -0.001942 eV | [+0.001138, +0.002815] eV |
| RRWP-pair | +0.002449 eV | -0.003576 eV | [+0.002714, +0.004420] eV |

K1's fixed500K internal-dev MAE was 0.1412533075 eV. Receiver-pair,
triplet-aggregate and RRWP-pair reached 0.1434436589, 0.1431951821 and
0.1448292434 eV respectively. Exact saved-tensor statistics and bootstrap
configuration are in [portability analysis](results/portability_analysis.json).
Float64 post-hoc reductions can differ from the remote float32 scalar by
approximately 1e-8 eV without changing predictions or any conclusion.

The added RRWP mechanism did not establish an incremental original-role win
over Receiver-pair: its paired interval crossed zero. On fixed500K dev it
was worse by 0.001386 eV, with a favorable-to-reference interval
[+0.000605, +0.002168] eV. The triplet-vs-receiver interval crossed zero on
both roles; there was no supported incremental triplet benefit.

## Attribution and limits

1. **No training-scale change occurred.** All candidate weights remained their
   selected 100K checkpoints. Thus additional 500K optimizer steps, a changed
   batch, learning rate, width or parameter exposure cannot explain this
   particular reversal. This does not identify the cause of every previous
   500K/full-training reversal.
2. **Execution parity was reproduced.** Wrong checkpoints, normalization, row
   alignment and ordinary reproduction drift were checked rather than assumed.
3. **The loss was uneven across ordered rows.** Receiver improved five of ten
   5K blocks; RRWP and triplet each improved six. Nevertheless, the later
   blocks dominated the mean loss. RRWP's final block regressed about
   0.02196 eV against K1. These blocks were a post-hoc descriptive partition,
   not independent random validation splits or a structure-based router.
4. **Information richness was not sufficient for portability.** Receiver-local
   nonlinear pair messages, walk-relative conditioning and vector pair-to-pair
   aggregation all produced an original-role positive but an audit-role
   negative. Selection optimism and differing molecular coverage remained
   plausible; neither was uniquely established without structure-linked
   residual analysis. The result did not reject GRIT or TGT as whole models.
5. **Statistical limits remained.** Row intervals were post-hoc, unadjusted for
   multiple comparisons and not estimates of seed variance. Original dev and
   this audit role were reused research roles, not sealed final tests. No
   official validation, test-dev or test-challenge was accessed.

The inference notebook occupied 380.467 wall seconds and 760.935 summed
allocated-device seconds (two T4s allocated, one used), below its 5,400-second
allocation cap. The full allocation was recorded; official account billing
was not inferred. CPU and queue costs remained unmeasured.

## Durable boundary

The three training trajectories remained replay-ready. The audit had its own
prospective plan, accepted endpoint evidence, observed role events, native
allocation cost and terminal decision. It was NO_TRAIN, with no optimizer
trace or training-prefix replay claim. The controller released zero successor
jobs. A future proposal would need a new mechanism-specific justification and
prospective authority; another width/depth/gate variant was not authorized.

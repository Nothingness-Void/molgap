# Final atom/bond readout dual — terminal decision

On 2026-10-03, saved-artifact acceptance independently accepted Kaggle3 kernel
`nvoid912/molgap-gptrans-g1-readout-dual-s42`, physical ID `136806829`, version 1.
Both independent T4 arms completed the frozen 60-epoch, FP32/no-TF32, BS128,
seed-42 contract: 46,860 optimizer steps and 5,998,080 sample presentations.
The immutable G1 degree-scaled, EMA-0.999 reference was reused, not retrained.

## Endpoint evidence

The numbers below were recomputed from aligned, saved EMA predictions on the
same 50,000 internal-development rows. Positive delta means worse than reference.

| Readout | Selected MAE (eV) | Candidate minus reference (eV) | Paired-row 95% interval (eV) | Best zero-based epoch |
|---|---:|---:|---|---:|
| Frozen virtual-node / virtual-self-pair reference | 0.1442326291 | — | — | See reference acceptance |
| Real-atom mean; original virtual-self-pair retained | 0.1448040565 | +0.0005714273 | [-0.0003376671, +0.0014825154] | 50 |
| Real directed-bond pair mean; original virtual-node retained | 0.1469739705 | +0.0027413413 | [+0.0018064035, +0.0036890557] | 59 |

Both models retained 5,246,817 parameters. Neither passed the prospectively
frozen material-gain gate. The atom-mean difference was inconclusive at the
paired-row level; the bond-mean difference was adverse at that level. These
intervals do not measure training-seed variation. The RML terminal adapter
preserved its declared `INCONCLUSIVE` non-promotion label for both arms; it
did not manufacture a new policy truth from an adverse point estimate.

## Bounded attribution

At epoch 59, reference live-training MAE was 0.09714278 eV, compared with
0.09360021 for atom mean and 0.09234526 for bond mean. Both replacement heads
therefore fitted training data more closely without a development improvement.
Their live-development endpoints were also worse, so the observed regression
was not solely an EMA selection artifact. This supports a fitting/generalization
trade-off, not a failed execution or evidence of insufficient total exposure.

Atom-mean EMA improved by only 0.00003041 eV from epoch 50 to 59; its selected
checkpoint was epoch 50. Bond-mean EMA improved by 0.00044248 eV over that span,
but still ended 0.00273201 eV behind the reference's same-epoch EMA. No unrun
extension was extrapolated into a win.

Access to more final states did not itself produce more useful prediction
information. A uniform atom mean discards selective graph summarization; a
bond-only pair mean discards non-bonded/virtual pair summaries at readout.
Those are plausible explanations, not isolated causal proofs of information
loss: this experiment changed the readout selection/aggregation mechanism,
not each proposed explanatory component independently. It closes these two
specific mean replacements, not every alternative readout or the whole
GPTrans family. Target-derived residual quintiles were descriptive only;
they did not justify routing, fusion, or additional experts.

## RML and resource closure

Both arms passed observed `STRICT_CAUSAL` comparison readiness, terminal
artifact/role/cost validation, and RML finalization. Each candidate was then
located in the rebuilt Replay pool with `capability=complete` and the actual
complete reference trajectory `TC-gptrans-g1-ema999-100k-s42` under the same
comparability key. The verification snapshot, hashes, matched curve samples,
and group members are recorded in [analysis.json](analysis.json); this was
actual admission, not merely a planned eligibility assertion.

The physical notebook consumed 11,510.057 seconds (3.19724 wall hours), or
6.39448 allocated T4-device hours, inside the 5-wall-hour / 10-device-hour cap.
Both allocated devices were used. Per-arm cost attribution divides this one
physical allocation; the notebook was not billed twice in this analysis.

Decision: close both replacements without promotion, seed confirmation,
scale-up, fusion, or a successor submission. Completion event
`evt-fdd45e634c82044f7f785123` was claimed by server A for this interpretation.
The already-paused existing Luna heartbeat required no new monitor. No local
training/model inference or protected-role access was performed.

## Evidence routes

- [Independent acceptance](acceptance.json), [analysis and actual Replay pair verification](analysis.json).
- [Atom-mean comparison](../degree_node_mean_readout_ema999/results/comparison_readiness.json),
  [closure](../degree_node_mean_readout_ema999/results/closure.json).
- [Bond-mean comparison](../degree_bond_mean_readout_ema999/results/comparison_readiness.json),
  [closure](../degree_bond_mean_readout_ema999/results/closure.json).
- [Prospective protocol](../../protocol.md),
  [immutable reference](../../reference/reference_bundle.json).

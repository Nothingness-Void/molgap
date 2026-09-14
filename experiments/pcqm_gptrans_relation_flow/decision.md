# Evidence review — 2026-09-14

The user reopened two Kaggle2 architecture rounds after the GPTrans reference.
This decision released the first of those two rounds, not a full-data repeat.

## Observations and limits

- K1 simplification combination gained 0.0021624118 eV against its own reference,
  below the 0.003 material/run-variation floor. Its gain over each parent was
  smaller still. This did not justify more slot-count/gate micro-variants.
- The adapted GPTrans core scored 0.1566272043 at 100K and 0.1039478481 in the
  accepted desktop 500K study. K1's separate 500K score was 0.1048590988.
  The latter difference was contextual, not a matched architectural effect.
- The GPTrans reference's selected epoch was its last (59), with train MAE
  about 0.103756. The 100K score used EMA; EMA decay 0.9999 has approximately
  6,931-step half-life: 8.87 100K epochs versus 1.77 500K epochs. EMA lag,
  exposure, dev-role differences and architecture all remain possible causes.
  Neither overfitting nor dataset-size sensitivity was proven in isolation.
- Code inspection found unnormalized persistent pair accumulation and raw
  attention logits entering the pair update. Adding a constant to each valid-key
  logit row does not alter softmax attention, but does alter that update.
  These are testable architectural hypotheses, not established implementation bugs.
- The closed-route indexes and evidence audit contained no pair-channel pre-norm
  or valid-key logit-centering experiment. Earlier relation-slot, shortest-path,
  geometric and directed-bond losses do not test these mechanisms.

## Release

The first round isolated `pair_prenorm` and `centered_logits` on two T4 devices.
Both retain the reference's complete parameter tensors, seed/RNG, topology,
loss, optimizer, EMA and exposure. No new dataset/geometry is built. The frozen
GPTrans reference, not the differently trained K1 score, decides mechanism gain.

The second round was reserved for controller attribution after terminal
acceptance. A distinct follow-up may test how persistent pair content reaches
nodes (rather than only the newest relation update), if the first round's
evidence still supports that question. It must not automatically combine these
candidates, tune EMA, or repeat a closed mechanism. Shortlist status requires
0.003 eV gain and a favorable paired interval; it is not K1/full promotion.

## Sources

- [GPTrans paper](https://www.ijcai.org/proceedings/2023/0396.pdf) and
  [authors' implementation](https://github.com/czczup/GPTrans) establish the
  three propagation directions, not the effectiveness of these new variants.
- [Combined K1 evidence](../pcqm_k1_combined_simplification_100k/decision.md).
- [GPTrans 100K evidence](../pcqm_gptrans_t_100k_v4/decision.md).
- [GPTrans 500K evidence](../pcqm_gptrans_t_500k/results/decision.md).
- [Historical mechanism attribution](../pcqm_gap_architecture/results/architecture_failure_attribution_2026-09-08/decision.md).
- [Frozen release facts](results/audit.json).

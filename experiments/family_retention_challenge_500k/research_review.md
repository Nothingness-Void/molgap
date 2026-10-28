# Evidence and literature review — 2026-10-28

## Investment conclusion

Park additional PPGT training under the present short-run budget. Keep ESA's
implementation and accepted count-readout evidence as an efficiency reserve,
without nominating it for full training. Investigate one complete GRIT model
against a reusable strong reference before opening another plugin campaign.
These are resource-priority decisions, not claims that either family is
intrinsically incapable of competitive accuracy.

## What the completed experiments actually established

The [500K terminal record](../modern2d_scale500k/results/completed_v3/decision.md)
and [saved-curve diagnosis](../modern2d_scale500k/results/completed_v3/convergence_diagnosis.md)
are the numeric authorities:

| Completed configuration | Parameters | Sample presentations | Common-Dev selected MAE | Gain in the last 499,840 presentations |
|---|---:|---:|---:|---:|
| PPGT mean | 2,577,729 | 1,999,360 | 0.1714706253 eV | 0.0100262 eV |
| ESA count | 5,015,745 | 5,998,080 | 0.1376542035 eV | 0.0000232 eV |

Both improved substantially over their own accepted source100K predictions on
identical Common-Dev rows. Neither had a qualified strong Common-Dev reference.
Both terminal RML records therefore remain CONTEXT_ONLY/INCONCLUSIVE, not
Replay-Ready. Cross-family exposure and learning-rate differences prohibit an
architecture-only interpretation of their score difference.

PPGT's late decline contradicts an intrinsic-convergence-ceiling claim. At its
observed roughly751 seconds per99,968 presentations, six million presentations
would cost about12.5 hours at the same cadence, before fresh startup overhead.
This projection is descriptive, not a measured equal-hardware benchmark. It
is nevertheless a reason not to spend another short notebook merely extending
the same expensive learning curve.

ESA's final development curve was nearly flat while online training error
continued to fall. Both families ended at LR1e-6. The evidence does not separate
small update size from representation/generalization limitations. A larger-LR
rescue could answer that narrower question, but would not establish that ESA
is competitive with an incumbent. It is not the first use of the next budget.

The [ESA count experiment](../esa_count_persistence_100k/results/completed_v2/decision.md)
did establish an accepted within-family benefit. This should be preserved rather
than erased by the weaker overall family ranking. PPGT's
[conditioner experiment](../modern2d_structure_100k/results/decision.md) did not
establish a material, stable gain; reopening that same plugin is unwarranted.

## Why the stronger model numbers cannot simply be subtracted

The [server GPTrans local-transfer pair](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md)
tested 5,998,080 presentations. Its real-bond-local arm achieved approximately
0.1096073 eV against0.1152574 for its matched G1 control. This is a stronger
reason to retain explicit chemical-bond messages than another small100K lead.
It used the earlier internal development role, not Common-Dev.

The reviewed desktop evidence at
[`0945f0e5e`, BondLocal+PairNorm decision](https://github.com/Nothingness-Void/molgap/blob/0945f0e5e/experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_acceptance/decision.md)
reported approximately0.0999874 eV for LR1e-3. It used29,998,080 presentations
and the earlier development role. Its within-job contrast isolated learning
rate, not PairNorm versus the older backbone. It is not a6M Common-Dev comparator.
Desktop jobs remain independently owned; this review did not adopt them.

Thus raw0.09999-versus0.13765 comparisons mix exposure, role and recipe.
The proposed experiment deliberately establishes the missing common benchmark.

## What the papers add

### PPGT: the local model is a compact adaptation

The [paper's v4 methods and configuration tables](https://arxiv.org/html/2504.12588v4)
combine a structural stem with simplified attention and adaptive normalization.
The published small configuration has17.6M parameters and a long training
schedule, unlike the local2.58M short run. The local result cannot falsify the
published family. Conversely, those larger-budget paper results do not establish
that our compact adaptation is a good use of a nine-hour T4 screen.

### ESA: edge tokens alone do not guarantee efficient chemistry learning

The [ESA paper](https://arxiv.org/html/2402.10793) combines edge tokens,
incidence-masked and global set attention, and attention-based pooling. The
order of these operations matters. It does not imply that every compact
configuration will outperform a persistent node/pair backbone. Local count
readout helped, but local identity-residual changes did not give a consistent
gain. Representation and optimization remain hypotheses, not identified causes
of the entire performance gap.

### GRIT: test the complete mechanism, not another positional plugin

[GRIT, ICML2023](https://proceedings.mlr.press/v202/ma23c.html) jointly evolves
node and pair representations initialized from relative random-walk probabilities.
Relations affect attention weights and transmitted values; degree information is
reintroduced at each layer. Its ablations support treating these components as
a coupled design. Those detailed ablations are on ZINC, so their effect sizes
must not be transferred to PCQM. The PCQM model is much larger and trained
longer than the proposed compact screen. This is a new local family test, not
a claim to have invented GRIT or to reproduce its published score.

The [official layer](https://github.com/LiamMa/GRIT/blob/main/grit/layer/grit_layer.py)
uses nonlinear node/pair conditioning, relation-valued messages, separate node
and pair residual updates, and BatchNorm. The degree must come from the real
graph before completing all node pairs. Dropping the pair update, treating RRWP
as a static scalar bias, or computing degree after dense completion would test
a different mechanism. Prior failed RWSE/SPD/plugin experiments do not answer
this complete-family question.

The [official PCQM configuration](https://github.com/LiamMa/GRIT/blob/main/configs/GRIT/pcqm4mv2-GRIT-RRWP.yaml)
uses16 layers, width256,8 heads, RRWP16, LR2e-4 and weight decay0. The proposed
compact model and shared incumbent recipe differ. A negative result closes
only this compact model under that recipe and budget; it does not close GRIT
under all training regimes. Implementation must pin an exact author revision.

### Why not replace this with a nominally simple GINE

The [2025 benchmark critique's PCQM appendix](https://arxiv.org/html/2502.14546)
uses a20-layer, width512 GINE with one million BS512 steps. Its roughly512M
sample presentations are not a cheap alternative to a6M screen. Strong paper
results alone do not justify allocating another family trial without examining
its exposure and optimization requirements.

## Remaining uncertainty

This review supports a better-motivated candidate, not a predicted winner.
The strongest demonstrated transferable component remains the explicit
real-bond-local GPTrans stream. A complete GRIT challenge tests whether more
coherent relation processing can improve that resource-bounded frontier.
No geometry, teacher, auxiliary pretraining, new data split or ensemble is
needed to test that hypothesis.

# Independent nonlinear pair transition: terminal interpretation

On 2026-10-06, version1 physical kernel137191673 passed independent saved-artifact
acceptance. Exact source/configuration, initialization, all60 epochs,46,860
updates,5,998,080 presentations,50,000 aligned development predictions, atomic
checkpoint chunks, runtime certificate, untouched protected roles and native
allocation cost were verified. Numerical details and actual Replay-pair proof
are retained in [interpretation.json](interpretation.json); the mechanical
authority is [acceptance.json](acceptance.json).

## Endpoint: no supported promotion

| Endpoint | Parameters | Selected epoch, zero-based | Saved-prediction Gap MAE, eV |
|---|---:|---:|---:|
| Immutable corrected G1+EMA999 | 5,246,817 | Frozen reference selection | 0.1442326291 |
| Independent pair transitions | 5,263,841 | 56 | 0.1440094602 |

Improvement was0.0002231690 eV. Candidate-minus-reference paired-row95% interval
was[-0.0010585800,+0.0006106318] eV, spanning zero;50.01% of rows improved.
The prospectively frozen0.003 eV material nomination gate did not pass. No
threshold or checkpoint selection was changed after observing the result.
The native terminal outcome INCONCLUSIVE/no-promotion was retained. Row
bootstrap is not training-seed variability, and repeated development-role
selection does not provide independent confirmation.

## Mechanism audit: three effective branches, one disconnected branch

The returned nonlinear updates were not uniformly dormant. At the final
scheduled diagnostic batch, return/input RMS ratios before blocks3/6/9 were
approximately2.09/0.197/0.068. All three had learned nonzero return weights
and nonzero last-batch gradients. The first branch carried the strongest update;
that observation alone does not establish overfitting or a rank bottleneck.

The block12 branch was different: return RMS, return-weight norm and last-batch
gradient norm were exactly zero at all60 observations. Static inspection of
the frozen implementation explains this. The addon changes only real-atom
pair rows/columns, excluding the virtual node. GPA computes each query row
from its existing node inputs and that same pair row; its pair projections are
pointwise and its node FFN/normalizations do not mix query rows. In the final
block, virtual-node inputs and virtual-pair row are consequently unchanged by
the addon. Readout uses only node[:,0] and pair[:,:,0,0], and there is no later
block to carry the changed atom states into the virtual node.

The prospective claim that moving the real-pair transition before block12
would connect it to the graph token was therefore incorrect. This is an
implementation/design limitation, not evidence that four effective nonlinear
pair transitions were tested. Earlier placement or a different readout path
would change the architecture and require a separately frozen intervention;
neither was silently applied or resubmitted. Frozen source and records were
preserved unchanged.

The matched EMA gain fluctuated around zero: epoch9 +0.001615, epoch19
-0.000469, epoch29 +0.000141, epoch39 -0.000415 and epoch59 +0.000229 eV.
There was no sustained material advantage. Final live and EMA development
values were0.14401558 and0.14401299 eV, respectively; EMA lag was not a
supported explanation for this small endpoint difference.

Thus the exact released architecture did not support promotion. It did not
establish a general failure of nonlinear relation processing, full GRIT, or
other pair-state model families. Post-hoc target-derived error quintiles were
diagnostics only; they did not authorize a Router or deployable selection rule.

## RML, cost and disposition

The observed intervention was STRICT_CAUSAL under its frozen comparison
contract, with no readiness blockers. Candidate and immutable reference were
actually found in the rebuilt Replay pool with capability=complete, no
exclusions and the same comparability key. This describes evidence completeness,
not full coverage of the intended four-branch mechanism. The INCONCLUSIVE
outcome has no binary policy-truth label; Replay admission is not promotion
or a newly scored policy decision.

The job consumed9,887.519232 wall seconds (2.746533 hours), with a mean
epoch duration162.626071 seconds. Both allocated T4s were counted even though
one justified worker trained:5.493066 allocated T4 hours, within the frozen
maximum. Utilization was not inferred from allocation. All required outputs
were retained through hash-pinned selective retrieval, without downloading
unselected outputs or executing a model locally.

The bounded route closed without a successor, extra seed,500K training or
frozen-weight portability run. Its separate prospective NO_TRAIN audit was
not released. The existing Luna monitor paused after one terminal handoff.
Repository RML validation and frozen-derived checks passed. No production
model, desktop job or protected evaluation role was changed.

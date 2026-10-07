# K1 cost and accuracy attribution — 2026-10-07

Scope: interpretation of the accepted 500K pair using retained trace/objective
metadata, saved-prediction metrics and the frozen executable source. No model
execution, new training, protected-role access or change to the
[terminal decision](decision.md). [Exact paired metrics](analysis.json) and
[mechanical inspection](../submission_v5/terminal_inspection/inspection_report.json)
remain authoritative. Historical comparisons below are contextual whole-package
comparisons, not qualified intervention effects.

## Finding

The composed K1 pays for two stochastic passes through an unchanged small
encoder, while its 500K selected endpoint stays near the older plain K1 endpoint.
The GPTrans package has improved substantially, with its EMA-versus-live output difference accounting arithmetically for a large
observed part of its advantage over K1. This supports deprioritizing this K1
recipe as a standalone accuracy/cost leader. K1 retains demonstrated value as
the complementary constituent of the frozen equal blend.

## Observed cost

| Accepted recipe | Mean epoch elapsed | Sum of 60 epoch windows | Selected endpoint | Elapsed through selected epoch |
|---|---:|---:|---:|---:|
| K1 pretrained consistency |1296.377 s|21.6063 h|epoch 49, 0.104904085 eV|18.0695 h|
| GPTrans G1/bond-local/EMA999 |901.240 s|15.0207 h|epoch 50, 0.101887695 eV|12.2320 h|

K1 is 43.844% slower by this recorded metric, adding 6.5856 h over 60 epochs.
These are one-worker epoch elapsed windows, including loading/training,
development evaluation and prediction publication before the trace timestamp.
They exclude some post-timestamp checkpoint/manifest work, queue/bootstrap,
idle peer allocation and historical pretraining. They are not CUDA busy time or
complete billed GPU hours. The pair allocation ledger remains a separate owner.

The existing same-recipe 100K K1 trace at commit 51bc61fc averages 223.999 s per
epoch. Fivefold scaling gives 1119.995 s; the final K1-only 500K stage averages
1137.904 s. This is compatible with ordinary data scaling for the current
two-pass recipe. The fivefold estimate also scales a fixed 50K development phase,
so it is not an exact training-throughput model. Earlier paired-stage K1 means
are 1305–1320 s; CPU/loader contention and other environment effects remain
unmeasured alternatives. There is no evidence that the completed versions
spent hours installing dependencies instead of training.

## Why additional compute need not produce better MAE

1. **A second whole model pass is mandatory.**
   [The composed objective](../../../src/molgap/pcqm_composed_500k.py) calls the
   K1 encoder twice per batch and backpropagates through both graphs. The
   [owning loss](../../../src/molgap/k1_pretrained_combo.py) is
   `0.5*(L1(first,y)+L1(second,y))+0.1*mean((first-second)^2)`.
   This increases stochastic supervised computation and encourages agreement;
   it adds no parameters, inputs or independent relation channels. The retained
   pretraining weights themselves add no per-step computation.
2. **Few parameters do not imply little execution work.**
   [K1's encoder](../../../src/molgap/qm9_neural_atom.py) has 9 persistent-edge
   updates and 9 edge-conditioned ResGatedGraphConv blocks per pass. With two
   passes, both stacks execute twice. Its graph-wide mixer runs only at layers
   3/6/9, with one active 64-dimensional slot. The slot-channel count is not 64:
    64 is the latent width. For a fixed assignment, that mixer returns one shared
   slot vector with node-specific scalar weights. GPTrans instead retains and
   updates a dense node-pair state through 12 blocks. This is an observable
   representation difference, not proof that the slot bottleneck causes this
   endpoint gap.
3. **The added agreement term is small in scalar loss value.**
   Equal-length epochs give a 60 epoch aggregate
   `0.1*sum(disagreement)/sum(supervised_l1)=0.409698%`; the mean of per-epoch
   ratios is 0.441333%. The corresponding early/middle/tail ratio means are
   0.28936%/0.44414%/0.58333% for epochs 1–10/21–40/51–60.
   These are numerical contributions to the normalized composite objective,
   not a comparison of physical L1 and MSE units, gradient norms or importance.
   They neither prove that consistency is ineffective nor diagnose harmful
   regularization. They do establish that the extra full forward is paid
   regardless of the agreement term's small scalar value.
4. **Execution settings also differ from older fast K1 runs.**
   This contract uses strict deterministic FP32 and AdamW with
   `foreach=False,fused=False`; the earlier 40 epoch 500K runner used fused AdamW.
   Both current arms share the deterministic/unfused setting, so it does not
   alone explain their ordering. The500K trainer constructs a fresh training
   loader each epoch and fresh development loaders, limiting the benefit of
   `persistent_workers=True`; the old scale runner reused loaders across
   epochs. There is no phase/operator profile to assign seconds to these
   factors. Fused/foreach speed and deterministic overhead are generic
   [PyTorch optimizer](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
   and [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
   observations, not measurements of this pinned Torch 2.4.1 runtime.

The used 500K preflight verifies deterministic optimizer/resume behavior,
mechanism activity and memory reserve. It does not record a synchronized
single-pass/two-pass timing comparison or a latency gate. Consequently,
mechanical qualification did not establish cost efficiency. This is the exact
missing performance evidence; it does not require another training framework.

## Why GPTrans overtakes this package

The old accepted matched 500K results used the same development row interval,
500000:550000. Read their [comparison](../../pcqm_500k_v4_evidence/final_comparison.json)
and [contract](../../pcqm_500k_v4_evidence/protocol.md).

| Family | Old matched 500K endpoint | Current composed 500K endpoint | Descriptive endpoint change |
|---|---:|---:|---:|
| K1 |0.104859870 eV|0.104904077 eV|0.0442 meV worse|
| GPTrans |0.106867528 eV|0.101887690 eV|4.9798 meV better|

K1 already beat old GPTrans on500K. The new ordering therefore does not establish
that 500K intrinsically breaks K1. Its new pretraining/consistency package has
not shown a 500K point improvement over the old endpoint, while the new GPTrans
package has a substantial descriptive improvement. Pretraining, consistency,
local flow, optimizer, transform and EMA changes cannot be credited or blamed
individually from these historical endpoints. The old 40 epoch K1 score is a
different record and is not substituted for this matched 60 epoch result.

At the current selected GP checkpoint, live MAE is 0.103826756 eV and EMA MAE is
0.101887690 eV. The 1.939066 meV difference is 64.2844% of its 3.016387 meV selected
advantage over K1; the remaining K1-versus-that-live difference is 1.077321 meV.
This is an arithmetic decomposition of observed outputs. It is not a causal
EMA ablation, an independently live-selected model comparison, or proof that
adding EMA to K1 would recover 1.939 meV.

## Explanations retained or rejected

| Explanation | Evidence disposition |
|---|---|
| The contract did not finish / resume reset training | Contradicted: 60 continuous epochs, 234360 updates, 29998080 presentations; accepted exact state/resume lineage and uninterrupted schedule. |
| Numerical explosion or inactive objective | No supporting evidence: all retained scalar fields finite, supervised/combined training loss decreases each epoch; disagreement and its preflight gradient are nonzero. |
| A few more epochs of this schedule should rescue K1 | Unsupported: best epoch 49, no improvement during the final 11 epochs, final 0.821 meV above best. |
| Underfitting, overfitting, excessive consistency or ineffective pretraining is established | Insufficient evidence: no matched clean training evaluation or isolated 500K intervention. |
| The current K1 global representation limits further gains | Compatible with code structure; not identified causally. No new slot-width sweep is released. |
| K1 has no remaining value | Contradicted for the frozen ensemble: K1 beats GP on 47.614% of rows; equal blend 0.098859154 eV improves 3.028536 meV over GP. |

The training-mode supervised MAE and clean development MAE cannot diagnose fit
from their gap. The second train-mode pass also updates K1 BatchNorm buffers
again; this is verified code behavior but an untested contributor to clean
inference behavior. GPU busy time, operator timing, data wait and clipping
frequency remain unavailable.

## Decision consequence

Do not continue this 60 epoch K1 schedule to chase standalone GPTrans. Preserve
the accepted K1 output for the fixed-blend question. Before committing a larger
cost to this particular two-pass recipe, the minimum missing performance
discriminator is a bounded, prospective, training-only comparison of single
and double passes using identical frozen runtime/batches, with loader,
forward/backward, optimizer and evaluation timings separated. Reuse the
existing family profiling owners; any optimizer/precision change needs its
own equivalence contract. This report does not launch that diagnostic, a new
module, an EMA retry, or full training, and preserves the terminal routing.

Metadata arithmetic sources: verified `trace.json` and
`objective_components.json` under the ignored retained locator
`platforms/_records/kaggle/staging/pcqm_k1_gptrans_package_transfer_500k/v5-checked-artifacts/<arm>/`.
Their remote output/checksum bindings remain in the committed stage manifests
and artifact inventories beside each arm's terminal acceptance.

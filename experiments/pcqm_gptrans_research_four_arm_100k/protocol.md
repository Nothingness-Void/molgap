# Four-arm fixed100K campaign — frozen2026-10-14

The human explicitly authorized two slots, four arms total: one baseline and
three candidates. This superseded the unsubmitted local-parent/cycle two-arm
proposal, not its accepted CPU input derivation. No fifth arm, seed43/44,
500Ktraining, full training, external corpus or protected-role access was covered.

## Mechanisms and evidence limits

All four arms retained the same scratch-initial local+triplet aggregate's
5,880,961parent tensors, exact state SHA
`32df5f9a19d353a95407d6bd4d53a10a2165ada7bb02c32392dd15a85cb0fd9a`.
The accepted historical aggregate was stronger than the local-only model on
the100Kdevelopment role. The new shared execution reference was trained once
because its execution contract differed; historical results were context,
not a substitute for the new reference's future predictions.

| Slot / arm | Sole change | Added stored parameters | Primary support |
|---|---|---:|---|
|1 / reference|Unchanged aggregate structure|0|[Accepted local aggregate](../pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md)|
|1 / cycle|Zero-start common-chordless-cycle3-category pair embedding|96,64active|[DGT paper](https://www.nature.com/articles/s41467-026-75005-9) and [pinned encoder](https://github.com/zhangsy-ryan/DGT/blob/2f5a4c3ef2e376d60238f481c36a87eeb5a581e8/graphgps/encoder/topology_edge_encoder.py)|
|2 / carry|Existing64-channel directed-bond messages carried between layers with11zero-start tanh scalars|11|[Local500K evidence](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md), [bounded proposal](../pcqm_gptrans_bond_carry_100k/proposal.md), [DGT bond state code](https://github.com/zhangsy-ryan/DGT/blob/2f5a4c3ef2e376d60238f481c36a87eeb5a581e8/graphgps/layer/dgt_layer.py)|
|2 / correspondence|Clean-main-input atom/bond local correspondence loss, no inference branch|9280train-only|[LEMON sections4.3–4.4](https://arxiv.org/html/2501.08589v1)|

DGT's ring ablation supports topology encoding, not a guaranteed PCQM gain.
Carry is a narrow adaptation of persistent chemical state, not a reproduction
of DGT/DeMol or another nonbacktracking spatial-edge experiment. Dense pairs
already persist; only the otherwise discarded sparse message is carried.
LEMON's transfer evidence is classification, not direct Gap regression. Its
local-correspondence idea is adapted to existing node/pair states; no full
line-graph encoder, global contrastive pretraining, author code/weights or
external training corpus is imported. This arm deliberately has lower confidence.
Negative native product, hard-routing and corrupted-K1 auxiliary experiments
remain closed and are not relabeled as evidence for these exact changes.

## Exact objective delta

The first three arms minimize normalized GapL1. Correspondence adds0.01times
symmetric InfoNCE at temperature0.2 between projected endpoint-mean node
representations and symmetric real-bond pair representations. Both projections
have32channels. Topology-only canonical uniform ranks retain up to8undirected
bonds per molecule, at most1024per physical batch. No label controls selection.
Negatives include other selected bonds in the batch; symmetric/equivalent-bond
false negatives and interference with Gap fitting are explicit null explanations.
No coefficient, temperature, cap or width sweep was released.
Only GapMAE is reported as the train/development target metric. Auxiliary loss
is not silently substituted for it. Development/inference computes no auxiliary
loss and does not alter the direct-Gap readout. Main inputs remain uncorrupted.

## Matched contract, authority and acceptance

Fixed accepted100K assets: train[0,100000), internal-development[100000,150000).
No graph rebuilding. Pure2D OGB atom9/bond3/SPD20, frozen train-only mean
5.338300266634226/std1.2750902063230574. Seed42, FP32/noTF32, physicalBS128,
drop_last32,46,860updates/5,998,080presentations/60passes,4warmup/cosine1e-6,
AdamWlr.001/wd.05/clip1/foreachtrue/fusedfalse, EMA.999each step/minEMAdev.
Pinned Python3.12/torch2.7.1+cu126. All common fields match; correspondence's
objective delta must be declared separately in eventual comparison readiness.

Two private notebooks, two isolated workers each. Every worker has its own
model/RNG/optimizer/checkpoints. Both preflights must pass in each slot before
formal updates there. The shared native repeat/serialized resume check,
exact disabled-intervention parent forward, connected mechanism gradients,
>=15%memory headroom and conservative timing budget gate are mandatory.
Local tests use only synthetic kernels/tensors and frozen initial tensor
inspection, never a real molecule forward or local graph-cache loading.

The ceiling is8observed allocation wall hours/16allocatedT4hours per slot,
32allocatedT4hours combined, including environment bootstrap/preflight and
idle devices. Estimates are not quota certification. Unknown prior allocation
intervals block continuation. Each physical invocation stops at the next10pass
boundary; accepted saved work is published privately before continuing the
unchanged60pass scientific horizon. No stopped segment is scientific completion.

Both slots may be submitted prospectively under this explicit four-arm
authority. The reference is PLANNED_NOT_ACCEPTED at release. The second slot
does not manufacture an accepted reference, strict comparison or Replay claim.
After completion, independently accept/freeze reference outputs first, then
each candidate against those exact target/row-aligned predictions. Require
complete60pass live-train/live-dev/EMA-dev traces, acknowledged checkpoints,
source/data/receipt identity, native certificates and full allocation costs.
The0.001eV favorable paired-row95%interval filter is directional consideration,
not seed stability, automatic nomination or production promotion. Existing
terminal/V5/RML validators own all stronger claims; no global gate is relaxed.

The separate full500Kinternal-development50K NO_TRAIN audit is prospectively
packaged. It uses matching accepted models, unchanged transform, BS128/FP32,
original100Kprediction reproduction before opening500Kdevelopment, prediction
retention before explicit scoring and model-state immutability. Its launch
requires actual training acceptance, bound bytes and remaining combined budget.
Neither an old10Kpanel nor a trained500Kmodel substitutes for that action.

Unknown submit outcome must be reconciled before retry. Retain distinct exact
kernel/version/source identities, scientific failures and infrastructure faults.
Do not cancel or adopt another machine's jobs. No new monitor chat/cron is created.

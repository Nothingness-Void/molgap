# Bond-message carry — design only, not a released experiment

Recorded2026-10-13. No training recipe, RML trajectory, prelaunch qualification,
platform job or monitor binding is created by this document. The tested tensor
primitive is not yet an integrated encoder or executable training addon.

## Question and evidence

Can preserving the successful64-channel real-bond message across depth improve
the accepted local+triplet aggregate, without deleting nonbonded relations or
introducing another attention path?

- [Local-stream500K result](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md)
  supports useful real-bond information after expanded-data optimization.
- [Triplet result](../pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md)
  supports retaining its soft all-pair communication rather than replacing it.
- [Product/hard-route results](../pcqm_gptrans_triplet_routing_100k/gpu/results/interpretation.md)
  close those exact modifications. Carry must not be disguised as another
  hard chemical mask, multiplicative pair composition or query/key grid.
- [Directed-bond GraphState](../pcqm_gap_architecture/results/directed_bond_graphstate_seed42/decision.md)
  was only weakly positive with a runtime penalty. Archived QM9 R10 failed its
  joint-target gate while Gap moved slightly favorably. Both remain closed;
  neither provides a PCQM promotion or permission to repeat nonbacktracking flow.

The native `BondLocalBlock.message` creates64channels, projects them to nodes,
and discards that separate hidden state at each layer. The dense32-channel
pair state already persists: the claim is **not** that GPTrans has no relation
memory. The proposed change preserves a distinct sparse chemical-message
state alongside the unchanged dense/global state.

## One mechanism and the strong comparator

Keep the accepted aggregate's12layers, all local branches, triplet insertions,
all-valid-pair routing, zero-start return/readout and initial tensor values.
For directed real bond e and layer l:

```text
u_l(e) = existing GELU message hidden(node_source,node_target,pair)
s_1(e) = u_1(e)
s_l(e) = u_l(e) + tanh(a_l) * s_(l-1)(e), l = 2..12
node_target += existing output projection(s_l(e)), summed over incoming bonds
```

Initialize all11 a_l to zero. There is no layer1 coefficient, which would be
permanently disconnected. Reset memory to absent at every model forward; keep
one exact directed-edge order throughout that forward. No reverse-edge walk,
bond-to-bond neighborhood, new feature cache, graph token, auxiliary objective,
pretrained checkpoint, geometry or output fusion is added.

The complete integrated model would have5,880,972parameters if no other tensors
are introduced: accepted aggregate5,880,961 plus11scalars. This is algebraic,
not a measured encoder inventory. The primitive's11parameters were measured
in synthetic tests; the full constructor remains unqualified.
New work per layer is O(E*64); the parent still contains dense pair/triplet
work. A live sparse state requires E*64floats; autograd retains a depth-dependent
history. Few parameters do not establish low runtime or training-memory cost.

Primary reference must be the frozen **aggregate**, not weaker G1. Retaining
all parent tensors and obtaining exact parent behavior with every carry zero
is a prerequisite, not a guarantee that learned carry will improve training.
Rotating hidden channel bases across depth, correlated/redundant messages and
negative optimization effects are explicit alternative explanations.

## Cheapest falsifier and release prerequisites

1. Reuse the native family/addon and source/init/acceptance/RML owners. Integrate
   only this tensor mechanism; do not implement another trainer or submitter.
2. Verify encoder-level disabled-carry equivalence, exact parent tensor pins,
   padding/edge order, per-batch reset, all11coefficient gradient paths, optimizer
   repeatability and atomic checkpoint/resume. Synthetic primitive tests alone
   do not qualify those properties for the integrated encoder.
3. Qualify the full50K audit adapter/reference and label-after-prediction boundary
   before relying on future portability evidence. The old10Kpanel is not a
   substitute. Keep each frozen action's identities separate.
4. Then prospectively freeze one seed42100K screen, FP32/noTF32/BS128/drop-last,
   the aggregate reference's complete native optimizer/EMA/exposure contract,
   module diagnostics and native cost. No baseline rerun or extra seed.
5. A proposed0.001eV material filter plus favorable paired-row interval may
   nominate only a single-seed follow-up; freeze it before compute. A reused-role
   improvement or row bootstrap does not establish seed stability or scale gain.

After real optimizer-inclusive timing, a future release would need an explicit
wall/device cap; no duration is promised from scalar parameter count. If carry
does not materially improve the strong comparator, close it without a depth,
coefficient-initialization, width or learning-rate grid. Log coefficient/return
RMS and matched train/live/EMA curves; do not infer mechanism activation from
destructive ablation or a final scalar alone.

## Implementation status

`src/molgap/bond_message_carry.py` supplies only the small stateless-in-batches
primitive. It preserves autograd through the predecessor, adds no RNG, has
bounded signed coefficients, and performs no parameter/device casting in
forward. `tests/test_bond_message_carry.py` uses synthetic tensors only.
No model registry, native trainer mode, scientific threshold, production model,
existing source identity or remote job is modified by this proposal.

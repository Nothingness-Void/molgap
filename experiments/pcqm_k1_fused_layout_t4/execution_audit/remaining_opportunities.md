# Remaining execution opportunities, 2026-10-11

Follow-up: [2026-10-12 deep profile](deep_profile_20261012.md) owns the later
memory-fill finding, opt-in shared implementation and qualification limits.

Static review of audited owner source `b8f95004`; no training implementation was
changed by that initial review. The native T4 pair used older source `e4876d35` and reduced TRAIN time
4.51%, not >=10%. The newer nonexclusive RTX5060 TRAIN-only10ep run reduced time
3.49%; source/runtime/exposure differ, so these are not a hardware comparison.
These engineering observations do not publish a new RML trajectory or authorize
a next diagnostic/training run.

The subsequently authorized [synthetic checks](synthetic_checks.md) rejected
priorities1/2 as execution replacements: numerical changes without additional
speed over CPU layout. Priority3 gained only about1% on synthetic evaluation;
its bounded-copy hook remains opt-in, with no changed default recipe. The
remaining proposals below are not implemented or qualified by those checks.

## Priorities

1. `gps.py::_PersistentEdgeUpdate.forward` projects `h[source]` and `h[target]`
   at every layer. A node appearing on several bonds is projected repeatedly.
   Candidate: `self.source(h)[source]` and `self.target(h)[target]`. Projection
   rows change from directed-edge count E to node count N for each projection;
   layer-dependent hidden states must still be recomputed each layer. The
   algebra agrees, but GEMM/backward accumulation arithmetic can differ.
2. `qm9_neural_atom.py::NeuralAtomMixer.compute_update` pads192-dimensional
   nodes before both bias-free K/V projections. Candidate: project real nodes
   first, then pack64-dimensional keys/values. Projection rows change from
   batch_size*max_nodes to total_real_nodes, depending on actual padding.
   Preserve masking, assignment, dropout/RNG, return projection and state keys.
3. `k1_screen_training.py::_evaluate` transfers predictions to CPU every batch.
   Consider deferred copies in bounded chunks; selected predictions/order must
   remain identical. T4 development time is about1041-1047s across40 epochs,
   so it matters to wall time, but a copy-only optimization cannot eliminate
   the model forward portion. Do not skip evaluations or change selection.

CPU FP32 synthetic forward/gradient comparisons passed for priorities1/2 at
rtol/atol1e-5. Fixtures used37 nodes/111 sampled endpoints, and37 real nodes in
60 padded positions. These are algebra/gradient checks, not a full-model,
CUDA, dropout/RNG, training-quality or throughput qualification. No molecular
data, accelerator or trained checkpoint was used by these fixtures.

## Lower Priority and Boundaries

- Epoch worker reuse already has `k1_loader_reuse.py`; reuse it rather than
  implement another loader. Global-RNG parity is restricted to stock PyTorch
  2.7.1 and deterministic packed data. The historical4096-row A100 profile had
  small loader waits; it cannot prove an epoch-loader change is negligible on T4.
- CPU layout indices/masks currently upload using blocking `.to(device)`.
  Worker-side layout construction and pinned asynchronous transfer are
  candidates only if their CPU/IPC cost and lifetime are measured.
- PyG's edge-conditioned `ResGatedGraphConv` projects concatenated node/edge
  features inside `message`. Splitting those projections could reduce repeated
  node work, but may add kernels and changes reduction arithmetic: lower-priority
  than the explicit persistent-edge projection above.
- Joint K/V GEMM, single-slot attention specialization, compilation and CUDA
  graphs need separate shape/RNG/equivalence work; no promised speed percentage.
- Do not obtain utilization by changing FP32/TF32, physical batch, objective,
  model depth, development frequency or checkpoint durability without a new
  applicable contract. Do not remove identity/finite checks to hide failures.

No retained-TRAIN quality or40epoch run was launched by this review or its
synthetic follow-up. Reconsider rejected projections only with a new,
decision-relevant execution context, not by repeating this same probe.

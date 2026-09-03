# Body-order moment seed-42 version-1 failure

## Disposition

Kaggle2 kernel `kaseichou/molgap-pcqm-body-order-graphstate-s42`, version 1,
ended in `ERROR`. This is an infrastructure-only preflight failure, not a
scientific result. The body-order candidate completed no epoch; no architecture
comparison or confirmation seed is authorized from this run.

## Cause

CPU initialization acceptance passed: the candidate had the frozen 3,681,329
parameters, its final injection projection was exactly zero, and every shared
baseline tensor was bit-identical. GPU preflight nevertheless required two
separate CUDA forward passes to be bit-identical with `torch.equal`.
PyG scatter reductions can vary at the last bits across separate CUDA kernels,
so this test rejected a mathematically unchanged initial function before
training.

The implementation-only repair keeps the zero-weight and shared-tensor checks,
replaces bitwise output equality with `rtol=1e-6, atol=1e-7`, and records the
observed maximum absolute difference. It does not change architecture, cache,
data roles, seed, precision, batch, optimizer, schedule, target, or budget.

## Retained evidence

- Main failure SHA-256:
  `0698df257a4dcebabc93cb53a10cf00e40f612269327fe5b25e991d4144a0216`
- GPU1 failure SHA-256:
  `4bec3d8fa204a4137bc216da67924f284a14c0486217eb579c2d5c8ed6d23d7b`
- Initialization preflight SHA-256:
  `137d7000beda849cc36fe777029c43da7689aab61343e877955a016cbc3f918f`
- Terminal log SHA-256:
  `44bdeebdc68313f342c5db57b9e24715ec29b45c8be79c75ecd29fc5838c3fb8`
- Local record:
  `platforms/_records/kaggle/training/pcqm_gap100k_body_order_graphstate_seed42_v1`

The completed fresh baseline artifact is retained as failure provenance only;
the unchanged-contract retry remains a matched two-candidate task.

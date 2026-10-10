# Evidence review

- RML trajectory index and complete K1 audit were read before designing this route.
- [Accepted A100 execution attribution](../pcqm_k1_colab_execution_profile/attribution.md)
  shows indexing/scatter/small matmuls and forward/backward dominance on4096 rows.
  Its41.46% single-forward saving changes supervision/BN, not numerical equivalence.
- [Complete audit](../pcqm_k1_complete_local_audit/analysis_zh.md) discourages blind
  width/slot/readout expansion and records unqualified execution/accuracy transfer.
- Current T4 user-pasted timestamps show a stable interval near0.175s/step plus
  an unexplained pause. They are not a retrieved full-epoch performance certificate.
- Current runner retains per-step finite guards, host loss, explicit synchronization
  and RNG capture. Removing only the barrier may have little benefit.
- Local PyG2.7.0 `to_dense_batch` infers graph/max-node counts from GPU tensors;
  Boolean unpacking also has dynamic output size. Existing K1 calls it three times.
  One CPU-derived integer layout is a distinct execution question, not a slot change.
- Official optimization background: [PyTorch tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
  [AdamW paths](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html),
  [PyG dense batching](https://pytorch-geometric.readthedocs.io/en/stable/modules/utils.html).

Alternatives: local GINE/EdgeState kernels may dominate; Windows/PyG/CUDA differ
from remote; guards/RNG or durable checkpoint costs may limit achievable savings.
This route does not repeat a completed architecture screen or consume more GPU hours remotely.

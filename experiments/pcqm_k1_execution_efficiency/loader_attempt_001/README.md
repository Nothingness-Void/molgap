# CPU loader attribution

Desktop bounded NO_TRAIN question,2026-10-10. Earlier scratch timing used
pre-collated CPU batches, excluding loader startup/fetch/pinning. Query the
[host audit](../HOST_OVERHEAD.md) and accepted [CUDA step decision](../combined_attempt_001/terminal_decision.md)
first. This attempt tests the missing CPU collation phase, not another model.

Reuse the hash-accepted4096 TRAIN graph probe and `pcqm_500k_v4_evidence.loader`.
Four alternating windows compare zero/two workers, physical BS128, all32 batches,
same fixed input order and generator. Verify every source member and batch tensor
hash outside timing. No checkpoint/weights, model, gradients, inference, fitting,
development or protected-role consumption. CPU4threads,300s worker ceiling.
Only known owned loader workers are cleaned up. No remote operation or monitor.

CUDA is hidden in this process before Torch import. Pin-memory is requested by
the original factory but disabled effectively without CUDA, and there is no GPU
consumer. Thus startup/fetch timing is a CPU Windows observation, not native T4/
A100 end-to-end, overlap/pinning throughput or justification for changing defaults.
Observe child CPU before cleanup when possible; otherwise total CPU cost is
unknown, never parent CPU mislabeled as complete allocation CPU.

Prospective publication must precede graph decoding. Positive means nominate a
native phase discriminator; negative means deprioritize loader changes. All
scratch loader batches are discarded after hashes. No training replay/adoption.

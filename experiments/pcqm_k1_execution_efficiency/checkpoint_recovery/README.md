# Checkpoint-only infrastructure recovery

Owner desktop; sub-attempt of the execution-efficiency question. The original
scratch comparisons/timing are retained unchanged; do not rerun them.

User-authorized local diagnostic, CPU only, four threads, <=120s cooperative
worker body. Reuse the same initialization and retained training-only probe;
two BS128 batches, three discarded updates, same FP32/AdamW/RNG semantics.
No development/protected roles, GPU, training candidate or remote submission.
The corrected wrapper's checkpoint helper is tested independently for CPU state
placement, atomic serialization, continued dropout/BN/optimizer state and layout
continuation. A new checkpoint path preserves the failed attempt's saved bytes.
Publish its prospective before decode/execution. Exact comparison is required;
failure is infrastructure/numerical evidence, never silent promotion.

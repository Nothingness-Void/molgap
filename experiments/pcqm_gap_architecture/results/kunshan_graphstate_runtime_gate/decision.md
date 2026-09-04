# Kunshan GraphState runtime-gate decision

The single-DCU Kunshan run completed and passed mechanical acceptance. The
frozen GraphState9 source and accepted ETKDG cache were used without changing
the Gap-only contract, and only the official-train-derived 100K train role and
10K internal validation role were read.

The run reached a validation MAE of `0.2967550099 eV` at epoch 3. After the
first-epoch warm-up, throughput was `303.75` and `305.97` graphs/s, with a
maximum epoch time of `337.32 s`. The recorded allocator peak was
`284,145,664` bytes. Both atomic checkpoints and all role/source/cache flags
passed acceptance.

This proves that the Kunshan Hygon DCU runtime can execute the frozen encoder
on real MolGap graph data. It does not establish A100 throughput or authorize
full-data training: the official-train scale is about 33.8 times the 100K
screen, so this DCU result is a compatibility datapoint only. The A100 timing
and memory gate remains the required decision for the 12-hour full-data plan.

The DTK scatter launch-bounds message in stderr is retained as a nonfatal
runtime warning; it did not alter completion or acceptance.

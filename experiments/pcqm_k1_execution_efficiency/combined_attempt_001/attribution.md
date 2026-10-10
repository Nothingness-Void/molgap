# Joint bottleneck attribution

Fused AdamW retains a large local increment after matching CPU layout and context
machinery. The current un-fused optimizer's many small elementwise launches remain
a worthwhile execution-contract target, not just duplicated Python scalar reads.
The final32.34% total is directly measured here and does not justify summing
percentages from independent attempts or projecting a full500K allocation saving.

TF32 still adds5.60% after fusion; its modest increment agrees with the prior
isolated trace's remaining indexing, sorting and non-GEMM work. No new kernel
trace/counter measurement was performed, so this attempt cannot claim a specific
Tensor Core utilization or new GEMM-only gain. Prior dispatch evidence is separate.

All pure-FP32 controls replay exactly, but both optimizer candidates differ;
TF32 adds substantially larger full-tensor short-update error. The direction of
retrained MAE is insufficient_evidence. Three scratch updates are not convergence,
training stochasticity, development generalization or independent-role accuracy.
Keep numerical, cost, resume and quality qualification separate.

Layout/transfer-only timing variation shows that exactness alone does not prove
speed. Repeated matched cost and complete native loader/checkpoint timing remain
required before adopting a deployment preset. Current evidence selects the next
quality question, not a production model, training release or replay-ready run.

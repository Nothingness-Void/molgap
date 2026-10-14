# Execution Attribution - 2026-10-14

Read the [decision](decision.md), [summary](summary.json), frozen
[protocol](../protocol.md) and per-arm inspection records. No new diagnostic,
model forward, training or protected-role access occurred during this analysis.

Supported: shared topology reuse removes redundant mask/index/reachability work
without changing the observed training trajectory or saved50K predictions.
Both families complete the same exposure and select the same endpoint, with
identical contract, initialization, target/row and recorded runtime identities.
The observed worker-wall reductions are14.96% and15.69%; effective end-to-end
presentation rates increase17.60% and18.61%, not exclusive kernel throughput.

The retained16-update, checkpoint-inclusive profiles give narrower evidence:
K1 capture6.137790s ->5.732367s; GPTrans5.366385s ->4.759495s. K1 launches
89892 ->89812; GPTrans91030 ->90286. GPTrans `aten::bmm` calls1456 ->1407
and `aten::mul`7369 ->7191; these count reductions support removal of repeat
topology work. Counts from truncated top50 tables are not a complete operator census.

K1's major GEMM call counts are unchanged (`aten::mm`8832, `aten::addmm`2560).
Its `aten::mm` self-device time is nearly flat at0.535103s ->0.534331s while
host/backward timing varies more. GPTrans optimizer/backward timings also
improve although their numerical algorithm is unchanged. Thus host contention,
clock/driver conditions and instrumentation variability remain compatible
contributors to wall savings; recorded runtime equality does not eliminate them.
No exclusive15% code-only speedup estimate or general hardware claim is justified.

Loader wait totals only6.394ms ->2.411ms and4.412ms ->3.737ms in these windows.
This does not support data loading as the main captured bottleneck. It does not
cover evaluation, full-epoch I/O or prove GPU utilization. Checkpoint phases
remain in the sample; never sum inclusive phase/device-kernel/operator rows or
extrapolate their proportions to the whole run.

No underfitting, overfitting, exposure-shortage or architecture-transfer finding
follows. Those are separate scientific questions. The missing discriminator
for causal wall attribution is a separately authorized, same-source/runtime,
profiler-off repeated old/new execution control; it is not launched here.

Retrieval reliability is separate: Python Requests large-file downloads hit
HTTP502, truncated bodies and timeouts. Native curl recovered the same
API-selected files; retained bytes still require the original SHA pins.
This is not a training failure, a scientific retry or an excuse to relax acceptance.

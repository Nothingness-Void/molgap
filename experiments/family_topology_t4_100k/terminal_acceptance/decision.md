# Execution Comparison Decision - 2026-10-14

This is an accepted execution diagnostic, not a model promotion. The
[summary](summary.json) owns measured values and paired saved-prediction checks;
per-arm mechanical/native reports and the independent review own qualification.
The frozen [protocol](../protocol.md) and [baseline bindings](../baseline_binding.json)
remain unchanged. Git routing follows the diagnostic non-promotion desktop route.

| Family | Prior worker wall | New worker wall | Wall reduction | Internal-dev Gap MAE, both |
|---|---:|---:|---:|---:|
| K1 mean2/live,40passes | 2.989857h | 2.542500h | 14.9625% | 0.1402572856eV |
| GPTrans local/EMA,60passes | 3.246755h | 2.737223h | 15.6936% | 0.1428613083eV |

Both runs complete the same31240/46860 updates and3998720/5998080 sample
presentations. Within each family the training contract bytes, initialization,
runtime fingerprint, profiler runtime/config and all40/60 trace observations
match the retained predecessor. Each50K source-index/target/prediction tensor
is bitwise equal; maximum prediction difference is zero. Selected passes stay
37/live and52/EMA. This is stronger than equality of rounded MAE and provides
no evidence of accuracy sacrifice under these exact100K contracts.

The new Notebook entry interval is2.814810h versus3.341182h previously.
Its observed two-device allocation totals5.629620 T4-device hours; this is not
Kaggle billing, GPU busy time, queue time or the entire platform allocation.

Profiler capture wall decreases6.61% for K1 and11.31% for GPTrans.
CUDA launch counts decrease only80/89892 and744/91030 respectively. The
[attribution](attribution.md) explains why the full15% wall difference cannot
be assigned entirely to those code changes from a single cross-job comparison.
Same-run old-source control and repeat-job timing variance were not measured.

Disposition: retain the already integrated topology-reuse implementation and
its new native/quality evidence; keep Track A/Track B model recommendations
unchanged. RML scientific status is INCONCLUSIVE with explicit missing strict
reference/replay exclusions, not NEGATIVE or a new architecture win. No retry,
profiler-off control,500K/full training, official-role use or successor is released.
The scientific interpretation and execution delivery result remain separate.

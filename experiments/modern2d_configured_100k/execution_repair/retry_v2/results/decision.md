# Two-family execution repair: terminal interpretation

Kaggle3 kernel139975571/version1 completed the bounded train-only diagnostic
on2026-10-21. Both native workers exited0; no formal train phase or Common-Dev
prediction/selection was dispatched. This result concerned execution feasibility,
not model quality, a scientific architecture comparison or training Replay.

## Native observations

| Same-job two-update probe | PPGT | ESA |
|---|---:|---:|
| Original-path cold process wall, seconds |40.498286804|2.922390592|
| Repair-path cold process wall, seconds |8.012679334|2.328195007|
| Maximum model-tensor absolute difference |1.6391277313232422e-7|0|
| Maximum optimizer-tensor absolute difference |6.05359673500061e-9|0|
| Original/repair full-state bitwise equality |false|true|
| Existing atol3e-6/rtol2e-5 update gate |passed|passed|
| Candidate repeat/serialized-resume full-state equality |passed|passed|

Cursor, scheduler, RNG and optimizer metadata matched in both families. The
original and repaired PPGT were not bitwise identical; the successful native
T4 gate did not erase its earlier failing tiny CPU fixture or qualify other
platforms. The scientific PPGT default therefore remained graphwise pending
a separate execution release. No tolerance or parameter exclusion was changed.
ESA's two-step original/repaired state was exactly equal, but this was still
not a long-run equivalence certificate or a measured MAE improvement.

The cold wall ratios were about5.05x and1.26x respectively. They included
loader/model setup, synchronization and checkpoint publication. With only two
updates, fixed ordering and no warmed timing repetitions, they were not
steady-state training speedups and could not forecast60-pass completion.

## Profiling and the remaining bottlenecks

The repaired PPGT two-update capture was10.7423s, with139,585 cudaLaunchKernel
calls and400 backbone/attention calls. The earlier graphwise profile retained
989,230 launches and4096 backbone calls. This cross-job counter comparison
supported reduced graphwise dispatch, not a controlled unprofiled speedup.
The new capture retained3.5532 host-self seconds in backward,1.6601 in RRWP
and0.8212 in loader wait; peak allocation was1,932,432,384 bytes. Positional
input/RRWP was still computed per molecule. A faithful batched/cached version
was a potential further execution question, not authorization to remove it.

ESA's two-update capture was3.3111s, with9,937 launches,0.8683 forward host-self
seconds,0.6602 loader wait,0.3883 checkpoint publication and0.3160 finite checks;
peak allocation was863,286,784 bytes. Cold CUDA module loading appeared in both
profiles. Long-run loader, validation/checkpoint frequency and warmed forward
cost were not resolved by this window. Nested timings must not be added and
host-self time is not GPU utilization. Neither profile supported increasing BS,
changing FP32 or model capacity to repair an observed memory bottleneck.

## Cost, roles and evidence limits

The whole Python allocation window was276.712098143s on two Tesla T4s,
553.424196286 allocated device-seconds (about0.153729T4h), below the1800s/1T4h
ceiling. The inner272.523144006s ledger overlapped and was not added to it.
Queue, pre-Python provisioning and complete billing remained missing. ESA's
idle tail while PPGT finished remained part of allocated cost.

The frozen native path removed the development dataset before scratch work;
only train100K topology/labels were consumed. Common-Dev permit metadata was
present but did not constitute payload use. There was no independent general
payload-IO telemetry. Protected roles,500K/full, geometry and extra seeds were
not used. Producer platform_version remained null; exact physical version was
bound separately through the actual launch receipt and version-specific output
retrieval, not backfilled into producer provenance.

Compact pinned metadata is under [retained](retained/); large states and raw
traces are retained in the bound external inspection directory. Independent
Luna saved-state verification passed both arms through the existing preflight,
repair-proof, context and retention validators, recomputing the reported drift
from CPU-loaded saved states without model execution. All24 pinned files and
the retention manifest matched; producer platform_version remained null.
The [independent report](retained/independent/saved-artifact-verification-4fe6a896cced4796bc055e8a81b458a4.json)
retains its corrected report-export error history as well as final PASS.

## Decision and next eligibility

Retain both repairs as useful execution findings; do not reject either family
on model-quality grounds, nominate a K1/GPTrans competitor, or infer MAE from
these probes. Scientific training/reference binding remains a separate gate.
The useful next qualification is a warmed capture-off budget measurement with
unchanged data, FP32/BS128 and exact source identity, rather than another long
screen whose endpoint cannot fit the budget. The PPGT CPU/native discrepancy
must remain explicit in any platform-specific release.

This diagnostic's terminal scientific classification is NO_TRAIN/context_only.
Its authority releases no automatic successor, scientific training or forced
Replay. The chain is closed after saved-proof acceptance; any new compute
requires its own frozen bounded release under the human research authority.

# Unified V2 Si/two-P data acceptance

On 2026-10-16 JST, the IMS master passed **DATA_ROLE_ACCEPTED** under the
[frozen two-P release](PCQM_UNIFIED_TWO_P_RELEASE.md). Cross-scale infrastructure
retained its separate **INFRA_CPU_ACCEPTED** status. Scientific training stayed
**NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING**. No model was implemented or run.

## Terminal evidence

Job `1733250.ccpbs1`, source `9039ee084ab9ed02e1da2c1a611ed1247e51d2f0`, rule
`f1bce8ca0161193341abea04a8920aea425e364348c029b7bd1b219672d1b077` completed
with native `joblog` exit0. The [scheduler observation](../../platforms/_records/ims/pcqm_unified_roles_v2/two_p_release/scheduler_observation.json)
retains actual elapsed, allocation, CPU time, points and memory. This corrected
the earlier unresolved terminal observation; no retry or dataset rebuild was
needed. Earlier strict/Si/ABI-failed attempts remain unchanged historical facts.

The [raw native acceptance](../../platforms/_records/ims/pcqm_unified_roles_v2/two_p_release/native_acceptance.json)
has SHA256 `673bcf10d10cb16d493c01221eac8edaf085918ebcaec77c26bdbdd5072ead65`.
The [terminal inspection](../../platforms/_records/ims/pcqm_unified_roles_v2/two_p_release/terminal_inspection.json)
verified the source archive/implementation pins, six role-manifest hashes,
member bytes, logical bindings, payload presence and six preparation events.
Retrieved stdout has SHA256
`c0c3e832c5456342fd84120ac8bda8eb7123879d7358bf3ffadf4899c644b4dc` and contains
the same complete acceptance JSON before the trailing memory warning.

The frozen generator called `accept_unified_roles` before atomic acceptance
publication. That independent CPU pass reopened graph chunks and verified
bytes, source-order maps, targets, topology, exact Full complement and held-out
hash ranking. This inspection reused that completed native content validation;
it did not rescan graph payloads on the login host or local machine.

## Data and roles

All3,378,606 train identities passed, including the31 frozen Si and exact two
authorized P structures; none was skipped, replaced or chemically edited.
There were68 durable source-chunk receipts and162 physical derived payloads.

| Role | Accepted rows |
|---|---:|
| train_100k | 100,000 |
| train_500k | 500,000 |
| train_1m | 1,000,000 |
| full_minus_holdout | 3,275,585 |
| common_dev | 50,000 |
| independent_audit | 50,000 |

All six member files matched the retained first-v4-attempt bytes. Small train
prefixes remained nested. Training-to-heldout, Dev-to-Audit and corresponding
exact-molecule intersection checks were zero; nested training-role overlaps
were intentional. Historical Full models remained ineligible for an unseen
Audit claim. Old fixed source manifest/acceptance pins were unchanged.

## Resource warning and costs

The log retained `Cgroup mem limit exceeded:` without a detailed OOM-kill
record. The platform [official FAQ](https://ccportal.ims.ac.jp/en/print/pdf/node/2063)
states that this message can also accompany a normally completed job when
memory is over-allocated but not used. This run's exit0 and complete verified
output support a nonfatal warning, not a failed data acceptance. The exact
allocation-failure source was not profiled; the warning was not erased or
generalized into a future runtime-safety guarantee.

Native accounting reported4 CPUs,0 GPUs,65m07s elapsed,62m00s total process CPU
and7,648MB used memory. Full allocated cost was4.341111 CPU-core-hours and
4.341111 native CPU points for this job only. The2h `walltime` field was the
requested ceiling, not actual runtime. The four-second timestamp-span/elapsed
difference was retained; allocation used the explicit native elapsed field.

The immutable acceptance's preparation wall/parent CPU timers were captured
before independent content validation, so they are not whole-job costs. Its
unknown allocation-cost field was not rewritten; scheduler accounting supplies
the separate measured bound-job cost. Earlier attempt costs and queue time are
not established by this job's accounting.

## Boundary and remaining work

- IMS master data was accepted; Kaggle1/2 mirrors were **NOT_PUBLISHED**.
- Audit payloads stayed private and separate. No Audit metrics, official
  validation/test/OOD access, GPU work or scientific ranking occurred.
- The data prerequisite for the separately specified Modern2D CPU integration
  was satisfied. Indexed V2 loading, family-specific RRWP/edge-mask input
  profiles and independent model-fidelity gates still require reviewed work.
  Old contiguous V1 loaders were not relabeled compatible; no new model or
  native scientific candidate was released by this inspection.
- Peak memory reached the allocation limit. A future authorized maintenance
  task should reduce peak allocations or review a larger CPU memory request;
  this inspection changed no runtime behavior.
- Existing235-test implementation regression and35-test documentation/layout
  results remain in the launch record; they overlap and are not summed. This
  closure only changes evidence/navigation, not the frozen execution source.

Final evidence checks passed:9 documentation tests, targeted navigation/budget
guards and terminal-record consistency. Read-only RML validate/frozen checks
passed without rebuilding or changing canonical/derived records. Counts overlap
the earlier documentation suite; the launch receipt pins the final JUnit SHA.

Canonical RML, historical V4/V5 conclusions, production models, user changes
and remote accepted payloads were not modified. Live status is in
`CURRENT_STATE.md`; native evidence is in the owning platform record directory.

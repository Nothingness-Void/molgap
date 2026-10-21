# Execution-repair CPU verification

The main agent implemented the changes; one reused Luna leaf executed checks.
Synthetic fixtures did not load real graph roles or run a scientific model.

| Check batch | Outcome | Retained report directory |
|---|---|---|
| Ten existing/new model, topology, execution, release and input suites | 191 passed, 2 skipped | `E:/MolGap/artifacts/modern2d-execution-repair-20261021/cpu/e18b8bb148554eb18ccdca46767fd33b/` |
| Batched execution including state-component drift guards | 24 passed | `E:/MolGap/artifacts/modern2d-execution-repair-20261021/cpu/bcaa3fc772aa4e8f91a3100aa84802cf/` |
| Scientific input and bootstrap regressions after source-reissuance repair | 16 passed | `E:/MolGap/artifacts/modern2d-execution-repair-20261021/cpu/9519aad692664ec4bc5a4183749ad0ac/` |

These batches overlap and are not a count of unique tests. The final batch
verified both model families: stale source pins remained rejected by the
ordinary runtime reader, only explicit non-training repair authority permitted
reissuance, retained input bytes were unchanged, and a second policy mismatch
still failed closed. The registered repair policy and its protocol SHA also
passed the owning metadata validator. User-owned RML derived files were not
rebuilt.

An earlier batch had86 passes and1 strict PPGT post-AdamW failure, retained at
`E:/MolGap/artifacts/modern2d-execution-repair-20261021/cpu/41648c91788541e4835cabdc8da5a022/`.
Forward/gradient closeness did not imply optimizer equivalence. The corrected
test retained that negative result; PPGT bucketed execution remained a probe,
not the scientific default. No tolerance or parameter exclusion was relaxed.

The first local preparation froze attempt v1 but stopped before prospective
publication/platform operations on the missing explicit policy. Attempt v2
registered that policy and passed local preparation and release checking with
source329dca4fb0afe9cd034c13fb62437602427a611e and package
e6735dde721be8546dc58176cbc2f1c4724a0a9e051c371781f7e34b0d752a0b.
The v1 freeze/failure was retained, not overwritten.

CPU success did not establish native T4 update equivalence, steady throughput,
scientific MAE, training release or Replay readiness. Those claims were outside
this check. No local real-data training/inference, protected role, desktop job,
SCNet or IMS access occurred.

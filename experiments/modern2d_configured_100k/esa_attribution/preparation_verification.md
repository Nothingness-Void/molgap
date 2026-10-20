# Campaign preparation verification — 2026-10-21 JST

The same configured Luna leaf directly executed the formal-release and configured
output tests:23 passed,0 failed/errors/skips,28.98 seconds. A subsequent scoped
two-case output test verified observed optimizer LR in the canonical trace:
2 passed,0 failed/errors/skips,10.51 seconds. These are overlapping checks,
not25 unique tests. Synthetic CPU fixtures only, no real local model/data run.
The worker created no descendants and was closed after its report. Raw logs
and JUnit were retained under
`E:/MolGap/artifacts/modern2d-esa-attribution-20261021/cpu-verification/`.
JUnit SHA256:
`98fc55984b5457014a0abe4f96e8113437381eaa16a977ec5b6441cef230ec6d`
and`fd3a40a2a1079d2c389e69d9feec82b1a9efa29a64fb6b9e1cd8f059109d5c32`.
The host-only unique-root temp ACL plugin used in earlier verification remained
explicit; this was not native GPU/runtime qualification.

New formal Modern2D outputs use the existing timed, checkpoint-last Session
publication path even for reference/composed modes. Per-pass trace rows bind
observed wall/cumulative time, checkpoint digest and the optimizer's actual
single-group LR at serialization. Scientific update arithmetic and historical
remote records were not rewritten.

## Kaggle3 byte mirror

`nvoid912/molgap-modern2d-100k-commondev-v1`, private immutable version1,
passed exact local and remote roundtrip verification of106 files/467633530
bytes against the accepted Kaggle2 selected-role publication. No graph was
rebuilt, no target/split changed and no protected role was included. This was
opaque byte transport, not local graph deserialization or model inference.
Some transient selected-file requests failed; retained hash-verified files
were reused and only missing files retrieved. The dataset was created once.

Detailed acceptance:
`E:/MolGap/artifacts/modern2d-kaggle3-mirror-v1/acceptance.json`, SHA256
`784e581470f017c38a27d6032060f261fae2b78ec3a8f7959283c5bf02f6ecdd`.
Original acceptance:
`E:/MolGap/artifacts/modern2d-native-preflight/data_roundtrip_acceptance.json`.
This accepts a mirror, not a completed model screen, runtime calibration,
scientific comparison or Replay entry.

Both preparations reused the ordinary scientific-input compiler, Session,
indexed input permits, fixed initialization, source packager and journaled
Kaggle adapter. LR preparation took41.70 seconds; WD took35.90 seconds. These
are local calls, not end-to-end upload/queue timings. Immutable executable
source was`ed4ade246a6a33df8e5eea18b99836fbe9b965a8`.

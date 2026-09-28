# 2026-09-28 profile v1 sidecar binding failure

`kaseichou/molgap-metagin-2d-runtime-profile/1` ended `ERROR` before any
timed optimizer step. Its consumer passed executable source commit
`e415b6b273957ec72e4342f3e9a965e5312d7ecf` to the sidecar verifier,
while the already accepted immutable CPU sidecar was produced by
`c569ca2e333e49de3f0388b816b46c8a278a5da5`. The verifier rejected that
false identity equality. The sidecar bytes were not shown corrupt; the local
three shard hashes, manifest hash and CPU full-recomputation acceptance remain
valid. Profile v2 pins the producer commit, manifest SHA-256 and aggregate
SHA-256 separately from its executable source commit. No training screen,
development metric or scientific comparison was produced.

Retained local artifacts:

- `platforms/_records/kaggle/training/metagin_runtime_profile_v1/metagin-runtime-profile/failure.json`
  SHA-256 `0ec11f27dc4468c42837301e583af32e904843bf0ce61b707069885243c8cd4b`
- `platforms/_records/kaggle/training/metagin_runtime_profile_v1/metagin-runtime-profile/native_cost.json`
  SHA-256 `d7700cb61109265c549d2d0de2006ca29e078e4d337ac561d204ca587f357b58`
- `platforms/_records/kaggle/training/metagin_runtime_profile_v1/molgap-metagin-2d-runtime-profile.log`
  SHA-256 `f6fd71060cd7d4b6ead0d1a6ec353d2d777c5c34da028c858f15821673a13b19`

Native worker wall time was 223.926 seconds; allocated GPU count was not
recorded by this failed profile, so native device-hours cannot be claimed.
No official validation or protected test role was accessed.

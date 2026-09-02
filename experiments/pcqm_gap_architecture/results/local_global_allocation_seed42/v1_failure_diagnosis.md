# Local/global allocation version-1 failure diagnosis

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-local-global-allocation-s42`,
version 1, ended with `ERROR` before any candidate completed preflight and
before any training epoch. The failure occurred after 104.4959 seconds of
script time.

The parent geometry wrapper owns `MODES = {distance, angle, distance_angle}`.
The new local/global subclass accidentally reused the same class attribute name
for its independent communication modes. Because the parent constructor reads
`self.MODES`, dynamic dispatch exposed the subclass communication set and
rejected the valid `distance_angle` geometry value.

This is an implementation-only namespace collision. The repair renames the
subclass constant to `GLOBAL_MODES`; it does not change the architecture, data,
split, geometry, seed, precision, batch, optimizer, learning rate, weight
decay, schedule, patience, parameter ceiling, candidate order, or sealed-role
contract. A static regression test now verifies that the subclass cannot
shadow the parent geometry-mode namespace.

Retained local evidence:

- `platforms/_records/kaggle/training/pcqm_gap100k_local_global_allocation_seed42_v1/pcqm_gap100k_local_global_allocation_seed42/failure.json`
  SHA-256 `08dd3f3a2c0dc81004d05c6025b5b41f293da2fe9f7a9bf3b78989cf1f4793eb`;
- `platforms/_records/kaggle/training/pcqm_gap100k_local_global_allocation_seed42_v1/molgap-pcqm-local-global-allocation-s42.log`
  SHA-256 `24d5d8f270c764218e3d88f7ca3f16218eedf330743f56ab19f2ec8d4ce3e01d`.

The failure record confirms `completed_candidates=[]`,
`official_validation_role_read=false`, and `test_dev_role_read=false`.

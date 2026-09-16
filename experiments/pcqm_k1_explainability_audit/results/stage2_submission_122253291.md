# Stage-2 round-1 submission 122253291

- Platform/account: Kunshan SCNet, `changfeng2006`
- Resource: one `dcu:Hygon` on `kshdtest`
- Source commit: `d10c688`
- Source archive SHA-256:
  `c94a6f4ec294160cfc33ccb0abd92a2cefeb7fcdf84ed675d3e801294858d0af`
- Frozen checkpoint SHA-256:
  `53f9118f34a95e02e3f0d798a56389af53ff55739753b4208c9f53c36d116d95`
- Frozen development payload SHA-256:
  `966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91`
- Submission: `122253291`
- First observation: `RUNNING` on `e06r4n09`

The job performs no training. It first reproduces the frozen K1-v4 development
payload, then suppresses the layer-3, layer-6, layer-9, or all global exchanges
without changing any weight or local EdgeState operation.

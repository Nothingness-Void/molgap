# Kunshan conjugated-component K3 status

- Stage: K3a accepted and attributed; K3b ComponentState preparation released.
- Cache job `121081124` completed in `00:11:43` on `a08r1n17`. CPU acceptance
  checked all 110,000 graphs and passed.
- Source: `2496c67097362d1ab9fe54145ef3fba33e73fdc5`.
- Source archive SHA-256:
  `a539426905fc3bf5967f97f872d06a9f672280c103ec9772160117ce09615605`.
- Parent geometry aggregate SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Remote root:
  `/public/home/scnaqkfcy3/molgap-results/kunshan-conjugated-k3-v5`.
- Two earlier `sbatch` attempts were rejected before queue entry because their
  requested memory-per-CPU exceeded the partition policy. They consumed no
  allocation. The accepted resource declaration is 16 CPU / 32 GB / 2 hours;
  the cache algorithm and scientific contract did not change.
- Job `121079846` then failed after eight seconds before reading graphs; its
  stdout path was outside the authorized root and was not accessed. Diagnostic
  retry `121080017` used in-root logs and identified a missing `libmpi.so.40`
  during Torch import. The CPU wrapper now loads the same OpenMPI module as the
  accepted Kunshan DCU runtime. Neither failure produced cache artifacts or
  changed the scientific contract.
- Job `121080047` then exposed that the DCU Torch build also requires
  `libhsakmt.so.1`, which is absent on CPU nodes. A dedicated account-local
  CPU environment was built with Torch `2.1.2+cpu`, PyG `2.6.1`, and NumPy
  `1.26.4`. A one-graph read-only smoke loaded the accepted `WedgeData` cache.
  The v5 task uses this isolated CPU runtime and does not load DTK/HSA.
- Accepted cache aggregate SHA-256:
  `40dfb281a474a3cb240e867572e398ab6128b8e1977821ac4a0bc0f3f95edc05`.
- K3a job `121082200` completed in `07:34:39` and passed no-model acceptance.
  Descriptor-only improved its fresh GraphState9 control by `0.0001969527 eV`,
  below the `0.001 eV` material-gain gate. The exact attribution is in
  `k3a_decision.md`.
- K3b ComponentState is released as the protocol's required communication
  comparison. No K3b remote job is recorded until its implementation and
  preflight contract are committed and submitted.
- Official validation/test-dev remain unread. No seed43/44 or full-data action
  is authorized.
- Coordinator task: `01a025a1-3b87-7781-8a91-f183193f7865`.
- Luna Max monitor task: `01a04479-ca44-7d31-95c4-6be485f256cc`.

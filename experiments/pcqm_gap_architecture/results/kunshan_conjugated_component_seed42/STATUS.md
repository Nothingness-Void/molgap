# Kunshan conjugated-component K3 status

- Stage: deterministic CPU cache construction and acceptance.
- Cache job: `121081124`, observed `RUNNING` on node `a08r1n17` in Kunshan
  CPU partition `kshctest02`.
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
- After cache acceptance, K3a may submit exactly one paired seed-42 GPU screen:
  fresh GraphState9 versus descriptor-only. K3b ComponentState remains locked
  until K3a is accepted and attributed.
- Official validation/test-dev remain unread. No model runs in this CPU stage.
- Coordinator task: `01a025a1-3b87-7781-8a91-f183193f7865`.
- Luna Max monitor task: `01a04479-ca44-7d31-95c4-6be485f256cc`.

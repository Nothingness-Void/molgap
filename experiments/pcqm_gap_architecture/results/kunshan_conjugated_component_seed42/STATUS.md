# Kunshan conjugated-component K3 status

- Stage: deterministic CPU cache construction and acceptance.
- Cache job: `121080047`, initially `PENDING (Priority)` on Kunshan CPU
  partition `kshctest02`.
- Source: `11bcb77a60ea33b5ce8f8a0a6041c72b33ffc8ae`.
- Source archive SHA-256:
  `bdc9562ae4f55100eca36ac4dd021ba335fa0d026ef305480201012f402a6d35`.
- Parent geometry aggregate SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Remote root:
  `/public/home/scnaqkfcy3/molgap-results/kunshan-conjugated-k3-v4`.
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
- After cache acceptance, K3a may submit exactly one paired seed-42 GPU screen:
  fresh GraphState9 versus descriptor-only. K3b ComponentState remains locked
  until K3a is accepted and attributed.
- Official validation/test-dev remain unread. No model runs in this CPU stage.
- Coordinator task: `01a025a1-3b87-7781-8a91-f183193f7865`.
- Luna Max monitor task: `01a04479-ca44-7d31-95c4-6be485f256cc`.

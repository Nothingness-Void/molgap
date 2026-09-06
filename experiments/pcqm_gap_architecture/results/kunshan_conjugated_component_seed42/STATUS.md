# Kunshan conjugated-component K3 status

- Stage: deterministic CPU cache construction and acceptance.
- Cache job: `121079846`, submitted on Kunshan CPU partition `kshctest02`.
- Source: `0c9445985dd391c44595531cce2039176b5680c5`.
- Source archive SHA-256:
  `0f6db636927a944f31a3dc170b5b5ba3db235832eb97b171e5a53a0c7f970af9`.
- Parent geometry aggregate SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Remote root:
  `/public/home/scnaqkfcy3/molgap-results/kunshan-conjugated-k3-v3`.
- Two earlier `sbatch` attempts were rejected before queue entry because their
  requested memory-per-CPU exceeded the partition policy. They consumed no
  allocation. The accepted resource declaration is 16 CPU / 32 GB / 2 hours;
  the cache algorithm and scientific contract did not change.
- After cache acceptance, K3a may submit exactly one paired seed-42 GPU screen:
  fresh GraphState9 versus descriptor-only. K3b ComponentState remains locked
  until K3a is accepted and attributed.
- Official validation/test-dev remain unread. No model runs in this CPU stage.
- Coordinator task: `01a025a1-3b87-7781-8a91-f183193f7865`.
- Luna Max monitor task: `01a04479-ca44-7d31-95c4-6be485f256cc`.

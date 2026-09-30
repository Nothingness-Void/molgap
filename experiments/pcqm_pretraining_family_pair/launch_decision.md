# Authorized two-family pretraining launch

The user selected K1-v4 and GPTrans Noisy Nodes + Pair Update Norm with a
10-pass pretraining stage, then their frozen 40/60-pass V5 downstream recipes.
Reuse accepted unpretrained references; no old pretrained comparator or baseline
retraining. On 2026-09-30 the user explicitly permits a new pretraining recipe
and requires existing training wheels rather than another training loop.

Implementation reuses qm9_local_hierarchy.pretrain_local_hierarchy with typed
parameters, pcqm_k1_variants_runner.train_arm (verbatim base from reference
source 0006a689b74cf96e994892464830b79687ca60de, plus stage/resume hooks), and
noisy_nodes.run_training_noisy_nodes. Shared source packaging and Kaggle
accelerator submission remain their existing owners.

K1 reference metadata is imported verbatim from server commit
67a6237895ce94d502991d251f22654df3ab06c7. This is evidence reuse, not adoption
of server implementation unrelated to the selected reference. Paired artifact
comparison follows completion; incomplete reference qualification stays pending.

Pretraining uses the existing loop's independent directed-bond masks and
seeded shuffled loader, not bytewise replay of the historical PCQM pretrainer.
The mask rate, 12 SMARTS targets and objective weights remain frozen.
CPU targets cover all 100K training rows, with exact OGB atom/bond alignment,
and preserve the official builder's hypervalent-silicon fallback. No role
is dropped and no protected labels are consumed.

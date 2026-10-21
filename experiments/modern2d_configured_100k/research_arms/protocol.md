# Two-family evidence-driven research campaign

The human authorized four Kaggle2/3 slots and eight independent arms on2026-10-22.
The objective was actual completed scientific training, not another execution-only
probe. The fixed accepted train100K and Common-Dev50K input identities, direct
normalized Gap objective, seed42, FP32/TF32off and physical BS128 were unchanged.
No official validation, audit/test,500K/full, teacher or geometry was authorized.

## Prior evidence and minimal interventions

The [native repair](../execution_repair/retry_v2/results/decision.md) independently
verified candidate repeat/resume and ESA exact original/repaired state. PPGT passed
the native tolerance check, not bitwise identity; the earlier CPU discrepancy was
not erased. New PPGT screens explicitly opt into the bucketed backend and repeat
native qualification for their own exact model/recipe before training.

The old PPGT MolGap budget model had16 backbone/8 pair-stem blocks and RRWP24.
It was not a small author benchmark recipe. The new compact preset is explicitly
a new budget architecture: hidden192,8 backbone blocks,2 pair-stem blocks,
pair64,RRWP8,SPE5,PE-MLP256,8 heads. All four PPGT arms share its one fresh
seed42 base tensor artifact. The paper's published PCQM configuration uses mean
pooling; MolGap's prior budget preset uses sum. Two independently declared factors
test readout normalization and residual-update access in a short horizon:

| Arm | Pooling | Initial learned LayerScale |
|---|---|---|
|ppgt_base|sum|1e-4|
|ppgt_mean|mean|1e-4|
|ppgt_scale|sum|1e-2|
|ppgt_both|mean|1e-2|

This2x2 design estimates both main effects and their interaction. Increasing
LayerScale initialization is not execution-equivalent initialization; that is
the declared intervention. Every other initial tensor is identical. The compact
preset is not claimed to be equivalent to the old9.5M model or a paper replica.

ESA's [partial LR evidence](../esa_attribution/lr/results/decision_2026_10_21.md)
favored3e-4 over1e-3, while1e-4 was a plausible second setting. Its author-style
encoder normalizes the residual stream before adding attention. The sole new
architecture hypothesis keeps the original input as the M/S encoder residual,
using LN only on the attention branch. PMA, masks, tokenization, initial skip,
MLPs, widths and every initial parameter remain unchanged.

| Arm | Encoder residual | Peak LR |
|---|---|---:|
|esa_lr3|author normalized stream|3e-4|
|esa_identity3|original identity stream|3e-4|
|esa_lr1|author normalized stream|1e-4|
|esa_identity1|original identity stream|1e-4|

## Exposure and budget

Each arm trains20 complete100K passes:15,620 optimizer updates and1,999,360
sample presentations, fresh from frozen initialization. The four arms within
each family share its retained schedule: PPGT cosine and ESA four-pass staged
warmup/cosine. Both use minimum LR1e-6, AdamW WD1e-5,
gradient clipping1, noEMA, full Common-Dev50K live-weight evaluation and best
development selection. PPGT peak LR is1e-3. This is an independent20-pass
discovery contract, not an early stop or continuation of the historical60-pass
schedule. Absolute MAEs must not be ranked against incompatible60-pass scalars.

Kaggle2 had about16.687 available T4h at planning; its two ESA notebooks each
reserve at most4 wall hours/8 allocated T4h. Kaggle3 had about22.072 T4h;
its two PPGT notebooks each reserve at most5 wall hours/10 allocated T4h.
The maximum planned reservation is36 T4h across eight devices. Bootstrap,
native qualification, profiler export, idle tail and checkpoint reserve count
inside each ceiling. No task exceeds the human10h ceiling. The platform adapter
rechecks actual quota before each POST; unknown outcomes are reconciled, not
resubmitted. No unapproved fifth slot or extra seed is allowed.

## Runtime and failure controls

The existing FamilyRecipe/Kernel/Session, indexed loader, source package,
check-release, submission journal and all-arm native barrier own execution.
Atomic last/selected checkpoints and bounded retained output chunks are required.
Each worker has its own device, RNG, optimizer and output directory. Profiler
captures four training updates after eight warmups with shapes/memory/stack
disabled; capture-off spans and full allocation are retained separately.

All eight frozen recipes must pass metadata/package checks and focused CPU
forward/backward/update/resume/output-inspection tests. Native preflight must
pass before that notebook's train phase. Failure is execution evidence, not
an architecture-negative result; preserve logs and report rather than weaken
validators. Budget stops stay partial, never completed20-pass endpoints.

## Analysis and evidence

There is one reference per family for this new benchmark, not a new baseline
per candidate: ppgt_base and esa_lr3. Full per-molecule source order, labels,
predictions, selected/last state, curves at matching steps, target transform,
runtime/source/feature identity and cost/role records are retained.
The four-arm campaign membership and references are frozen in campaign.json
before any submission. The configured owner still calls initial training
context_only: no fabricated pre-existing accepted reference or STRICT_CAUSAL
claim is used to bypass release. Scientific paired endpoint/reference/replay
qualification follows actual completed baseline acceptance; missing eligibility
must remain explicit, not be relabeled Replay-Ready.

Do not nominate from a lucky best scalar alone. Compare candidate/control at
5/10/15/20 passes, their best Common-Dev predictions, paired absolute-error
differences and molecular-size residuals. Inspect training fit versus development
generalization, time to a fixed MAE and mechanism interactions. No automatic
multi-seed,500K/full or combination beyond these eight arms is released.

The existing server Luna B polls the four exact receipt-bound jobs every30min,
silent while healthy. It reports completion/error to the same controller A;
it never selects successors, changes code or submits work.

Sources: [PPGT pinned PCQM configuration](https://github.com/LiamMa/PPGT/blob/62cafd3abbb8d72e9a8b3146f947a192cf0aacaf/configs/pcqm4mv2-PPGT-ViT-B-BS256x2GPU.yaml),
[ESA author implementation](https://github.com/davidbuterez/edge-set-attention),
[ESA paper](https://arxiv.org/abs/2402.10793). The new residual variant is a
MolGap research intervention, not attributed to the author's reported model.

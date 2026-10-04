# Frozen K1 pretrained consistency/teacher 100K pair

## Authority and scope

The desktop user authorized submission on2026-10-05 after the two-arm proposal.
Kaggle1 is the retained platform. This is a new bounded100K question;
no retry of closed direct distillation, baseline retraining,500K/full training,
official validation, test-dev or challenge use is authorized.

## Inputs and initialization

Both arms use the exact accepted fixed100K OGB training manifest
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`:
training source_idx[0,100000), development[100000,150000). Development was
previously used for selection and is not fresh. Pure2D strips geometric fields.
Use the unchanged K1-v4 atom192/edge64/slot64/9-layer/RWSE16 factory with
3658817 inference parameters, not another family's weights.

Reuse the exact completed10-pass K1 pretraining checkpoint from the mapped
archived owner. Retain its file hash, source/config/stage metadata and historical
qualification/cost gaps. Extract only the K1 backbone; restore every `head.*`
tensor from the original frozen seed42 state. Record exact state/head/file
digests and the extraction mapping before release. Never use downstream selected
Gap weights. Each arm strictly loads identical prepared state and starts a new
AdamW/cosine/RNG stream. Historical pretraining compute is reused, not zero-cost
or a new measured allocation.

ArmB uses the existing fixed50:50 K1 mean2/consistency2 teacher cache, exclusively
training source_idx[0,100000), joined on CPU indices before device transfer.
Reject duplicate, foreign or development indices and changed payload/manifest.
Teachers remain frozen; no teacher optimizer or online teacher execution.

## Training recipe

Seed42; FP32; TF32 disabled; deterministic algorithms; physicalBS128;
drop_last; no accumulation/EMA. AdamWlr0.0004,weight_decay0.00001,clip1;
CosineAnnealingLR T_max40,eta_min0.000001. Exactly40 completed epochs,
31240 optimizer updates and3998720 presented samples per arm. Two independent
dropout forwards per training batch; one backward/optimizer update. Select the
best clean live development MAE under the unchanged target normalization.

Let p1,p2,y,t be normalized Gap outputs/labels/fixed teacher targets.

ArmA loss =0.5*(L1(p1,y)+L1(p2,y))+0.1*mean((p1-p2)^2).
ArmB adds1.0*mean(((p1+p2)/2-t.detach())^2).
The teacher operates on the mean prediction, avoiding an implicit extra
lambda/4 disagreement term from averaging two per-pass teacher MSEs.
This mathematical distinction is not evidence for the previous failure cause.

Retain each epoch's label-only online MAE, supervised loss, disagreement,
teacher MSE applicability/value and total loss in a hash-bound objective trace.
These online metrics are not a clean fixed-subset training evaluation and must
not be used alone to label underfit/overfit. Keep clean live development MAE,
learning rate, epoch/step/presentation axes and selected state in the existing
canonical trace. Atomic resume retains model/AdamW/cosine/RNG and acknowledged
sampler cursor. Clean inference remains one unchanged K1 forward.

## Qualification, roles and native cost

Use existing source/recipe/init/release checks. One focused final synthetic
batch verifies exact loss/gradient paths, detached teacher, no accidental
double disagreement penalty, initialization binding and resume behavior.
On the actual assigned T4, preflight BOTH arms on training-only BS128 fixtures:
nonzero finite disagreement/gradient, reproducibility, resume and selected-state
roundtrip; clean initialization equivalence, native timing/memory and runtime
certificate. Formal training starts only after the shared all-arm barrier.
Do not change precision,batch or optimizer to pass. A failed qualification is
an infrastructure/NO_TRAIN disposition, not scientific failure.

Estimate4T4 assigned-device hours per arm (8 combined), based on retained
single/two-pass100K costs; not a measured charge. Queue,CPU and wall remain
separate. Kaggle's remote session cap is the execution boundary; retain atomic
checkpoints and incomplete state on interruption. Report actual per-arm
training invocation allocation and preflight separately; no GPU busy-time claim.

## Frozen scientific gates

Primary teacher-increment nomination requires B's best clean development MAE
to improve over same-job A by at least1meV with positive paired-row95% lower
bound (1000 draws,seed42). Successful compression additionally requires B
to be no more than1meV worse than the fixed teacher on the exact same retained
development rows. Report both gates separately. One seed and reused selection
rows do not measure training stochasticity or independent generalization.

ArmA's comparison to historical consistency is contextual until its exact
source/runtime/roles/exposure and declared initialization intervention qualify.
Do not fabricate strict reference or retrain one. Each arm needs independent
mechanical acceptance, runtime,cost,roles,canonical trace and terminal RML.
Same-run replay declaration binds A as this new reference/control and B as
candidate; eligibility is fail-closed under existing V5 validators, never promised
from a scheduler receipt. Historical pretraining provenance gaps stay explicit.

No weight/seed/schedule sweep or automatic successor. On terminal evidence,
write attribution before proposing an unrelated module. Positive adopted work
routes to desktop; negative complete history to archive with accepted canonical
desktop RML discovery. Pending questions retain their owning branch.

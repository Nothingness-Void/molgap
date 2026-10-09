# Prospective K1 native T4 cost-quality100K

Desktop-owned preparation only. Parent owns source commit, profile acceptance,
runtime release and remote operations. No training, inference, API, publication,
500K/full release, model adoption or protected-role use is authorized here.

## Question and frozen arms

Does single-forward K1 retain mean2 quality while removing redundant training
work? Reference `mean2` uses `k1_two_pass_mean/1`, config `{}`, trainer mode
`mean2`; candidate `single` has no addon, trainer mode `reference`. Scientific
role is independent of trainer mode. Mean2 averages two dropout-bearing
normalized Gap L1 losses, with one optimizer update and no consistency penalty.
Mean2 has a distinct objective hash describing two losses, one update and two
BN updates per batch; its source hash pins functional implementation. Both use
clean single-forward development inference and best-development-live
selection. This is not a teacher, ensemble-inference or consistency rerun.

Both reuse the identical native V4 scratch tensor state
`8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`,
retained at `D:/w/k1-dropout-consistency/experiments/pcqm_k1_dropout_consistency/initial_state.pt`.
Preparation checks CPU tensor digest and reports the separate file digest;
there is no warm start or replacement initialization.

Kaggle3 fixed accepted `nvoid912/pcqm4mv2-ogb-fixed-100k-v1`: official-train
prefix train `[0,100000)`, development `[100000,150000)`. Pure2D direct Gap eV;
no3D construction, EMA, teacher or pretraining. Native FP32/noTF32, seed42,
physical BS128/drop-last,40 epochs, AdamW4e-4/wd1e-5/clip1,
cosine eta_min1e-6, normalized Gap L1, frozen epoch shuffle. Each arm has
31,240 optimizer steps and3,998,720 optimizer-batch presentations; two forwards
do not double that counter. Recipe/data/role/target hashes come from the
accepted fusion-distillation recipe and original V4 declaration, not fixtures.

## Profile first and budget

Preparation is permitted; TRAIN remains blocked until parent validates a
passed verified native T4 profile and completed local diagnostic with no hidden
urgent fitting failure. A100 profiling is not native T4 qualification. Source
and executable recipe must be committed/rebound before `prepare-workflow`.

Pair ceiling:8 allocated native T4 device-hours,4 wall-hours on T4x2 including
setup, qualification, idle peers, evaluation, checkpoints and cleanup. Each
recipe adds top-level `allocation_wall_limit_seconds=14400`, outside training
constants. Parent binds both equal limits at bootstrap and passes the remaining
lifetime minus60s cleanup reserve to the existing pair runtime. No new monitor.
Inspect/release this shared timeout implementation before launch; absence blocks
TRAIN. Stop-for-cost is valid, not scientific failure or permission to retry.

Existing K1 trainer retains atomic complete-epoch model/optimizer/scheduler/RNG
state and exposure cursor; resume must reconcile trace/checkpoint identities.
Parent must ensure independently retrievable durable copies and exact source,
config, cache, job/version and resume provenance. Worker-local checkpoints alone
do not satisfy durability. Cross-job continuation is not same-physical-run replay.

Planned cost events allocate4 estimated device-hours to each device, totaling8;
their concurrent4 wall-hours must not be summed as8 pair wall-hours. These are
ceiling-based planning estimates, not measured forecasts. Actual allocation ledger
counts all allocated idle devices; training-step time is a separate measurement.
CPU/queue unknown remain missing; estimates and actuals remain separate events.

## Frozen decision

Define delta = single MAE minus mean2 MAE on aligned finite50K rows. Use paired
row bootstrap1000 draws/seed42, two-sided95% percentile interval. Cost-quality
nomination requires upper95% delta <=0.001eV, >=25% measured native T4 optimizer
step savings (1 - single/mean2 matched step seconds), and <=8 allocated T4
device-hours including overhead. Profile timings must separate loader, forward,
backward, optimizer, development and publication; native step measurements must
match frozen BS128/FP32/software/fixture and exclude diagnostic construction.
Missing runtime/cost/artifacts means pending, not a passing gate.

The1meV tolerance is engineering noninferiority, NOT measured training noise or
a promotion gate. Separately label material precision only if mean2-single
gain >=0.003eV and its paired95% interval excludes zero. Row uncertainty is not
training stochasticity. Neither direction permits posthoc recipe/seed/tolerance
tuning,500K/full advancement or adoption. Terminal attribution precedes another
module. Keep mechanical, science, cost, replay and delivery decisions separate.

## Prospective and acceptance ownership

Before launch, Spec v2 binds `same_run_replay` reference `mean2`, candidate
`single`. `plan_prospective` derives pair bindings; input plans cannot override
them. Mean2 is not accepted until its actual terminal evidence is accepted.
Strict comparison/readiness stays pending until qualified runtime, artifact,
role, cost and same-run provenance exist; close reference before candidate.

Reuse `prepare-workflow` and `accept-workflow`, shared source inventory,
registered family recipe/execution, paired RML and terminal closure owners.
The explicit `molgap-family-same-run-acceptance-plan-v1` contains only
`format`, `spec_identity`, `arms`; each entry contains `arm_id`, `adapter`,
`expected`, `contract`, `target_manifest`. The pinned original target manifest
supplies legacy target encoding context ONLY. There is no historical bundle,
passing prelaunch comparison or invented accepted mean2 evidence. Reference
authority comes from the Spec's frozen prospective pair declaration. Shared
validation/staging of this format is parent-owned. Availability establishes
prospective same-run input availability, not terminal reference acceptance or
strict readiness; `reference_authority` explicitly retains that distinction.
BN update count and dropout/loss-pass differences are intended intervention
mechanisms, not silently assumed causal equivalence. Runtime, trace, role and
cost qualification are required for the eventual same-job endpoint authority.
Policy registration and canonical prospective publication remain parent-owned;
the script only validates or stages unpublished local drafts.

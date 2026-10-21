# PPGT/ESA eight-arm research

Human-authorized four-slot/eight-arm20-pass100K campaign.
The [protocol](protocol.md) freezes two factorial designs, exact data/FP32/BS128,
per-account reservation and scientific limits. All four immutable releases passed
source round-trip and check-release; [actual returned identities](launch_bindings.json)
bind four private T4 version1 jobs, not an unsubmitted plan.

| Slot | Account | Kernel ID/version | Last scheduler observation |
|---|---|---|---|
|ESA1|Kaggle2|140049215/1|COMPLETE; both20-pass endpoints mechanically accepted|
|ESA2|Kaggle2|140049216/1|COMPLETE; both20-pass endpoints mechanically accepted|
|PPGT1|Kaggle3|140049247/1|COMPLETE scheduler; STOP_FOR_COST,10 passes plus partial11|
|PPGT2|Kaggle3|140049200/1|COMPLETE scheduler; STOP_FOR_COST,10 passes plus partial11|

The [independent CPU verification](cpu_verification.md) covers model mechanisms,
update/resume/output inspection and compilation. Pinned Python3.12/torch2.7.1
installation was observed on the three running jobs. Native qualification and
formal training were initially unobserved. Six arms subsequently had positive-step
first-epoch telemetry: [ESA1](esa1/first_epoch.json),
[PPGT1](ppgt1/first_epoch.json) and [PPGT2](ppgt2/first_epoch.json).
ESA2 subsequently completed both20-pass endpoints. Both families therefore have actual native
training evidence, not just CPU fixtures or accepted scheduler submissions.
First-epoch metrics are not endpoint acceptance, superiority or Replay eligibility.

PPGT's two slots stopped under their frozen5h allocation budgets. All four arms
retained10 complete passes and part ofpass11; no20-pass endpoint completed. Their
[slot1 decision](ppgt1/results/decision.md) and [slot2 decision](ppgt2/results/decision.md)
preserve original10-pass snapshots, latest paused states, full observed curves,
profiles and V5 role/cost records. These are partial evidence, not complete
family-output manifests, scientific winners, scale approval or Replay-Ready.
The [four-arm analysis](ppgt1/results/interpretation.md) retains mean pooling as
a partial optimization lead and identifies the shared small-operation/RRWP
bottleneck. A disabled-EMA portable-resume inspection defect was repaired;
five regressions and all four unchanged snapshot bundles passed verification.
The earlier first-pass feasibility warning was realized; the horizon and
checkpoint reserve were not changed retrospectively.
The [startup interpretation](startup_interpretation.md) corrects the frozen
protocol's PPGT warmup shorthand and explains why a literal training START line
is not expected from the epoch-only logger. Actual compiled recipes are unchanged.

The existing Luna B used one30-minute schedule for
[Kaggle2](kaggle2_monitor.json) and [Kaggle3](kaggle3_monitor.json). Initial ticks
authenticated both actual owners and returned only SILENT healthy observations.
Terminal events are delivered idempotently to the same A without overriding A's
model/reasoning. All four bound terminal events are closed; both account bindings
are closed and this campaign's existing heartbeat may pause. No new monitor,
automatic successor or cross-machine custody is created.
Large outputs and operation journals remain outside Git. Original partial and
failed histories are retained, not relabeled as this fresh20-pass benchmark.

ESA1 has a [terminal decision](esa1/results/decision.md): the identity residual
gave a small uncertain point improvement, with no nomination. Both models,
predictions and full curves passed mechanical inspection. Standard RML terminal
staging is retained, but canonical finalization fails closed because this initial
context-only prospective input froze no reference/pair binding. No Replay claim,
post-hoc prospective edit or rerun is used to hide that gap.

ESA2 has a [terminal decision](esa2/results/decision.md) and
[four-arm interpretation](esa2/results/interpretation.md). All four ESA endpoints
passed saved-artifact inspection; identity residual direction reversed between
the two learning rates, without nomination. The second slot's canonical RML
finalization has the same unmodified prospective-reference blocker. Both ESA
monitor bindings are closed. The PPGT partial decisions close the two remaining
bindings without continuation or successors. Validated retained evidence and
canonical RML/Replay admission remain explicitly separate.

# PPGT/ESA eight-arm research

Human-authorized four-slot/eight-arm20-pass100K campaign.
The [protocol](protocol.md) freezes two factorial designs, exact data/FP32/BS128,
per-account reservation and scientific limits. All four immutable releases passed
source round-trip and check-release; [actual returned identities](launch_bindings.json)
bind four private T4 version1 jobs, not an unsubmitted plan.

| Slot | Account | Kernel ID/version | Last scheduler observation |
|---|---|---|---|
|ESA1|Kaggle2|140049215/1|RUNNING|
|ESA2|Kaggle2|140049216/1|QUEUED|
|PPGT1|Kaggle3|140049247/1|RUNNING|
|PPGT2|Kaggle3|140049200/1|RUNNING|

The [independent CPU verification](cpu_verification.md) covers model mechanisms,
update/resume/output inspection and compilation. Pinned Python3.12/torch2.7.1
installation was observed on the three running jobs. Native qualification and
formal training were initially unobserved. Six arms now have positive-step
first-epoch telemetry: [ESA1](esa1/first_epoch.json),
[PPGT1](ppgt1/first_epoch.json) and [PPGT2](ppgt2/first_epoch.json).
ESA2 was last observed QUEUED. Both families therefore have actual native
training evidence, not just CPU fixtures or accepted scheduler submissions.
First-epoch metrics are not endpoint acceptance, superiority or Replay eligibility.

PPGT first passes including Common-Dev took1663–1730 seconds versus ESA's662.
A naive first-pass extrapolation puts PPGT's20 passes at9.2–9.6h, above its5h
allocation. This is a material feasibility risk, not permission to extend a run,
change its horizon or call a partial prefix complete. The original hard budget
and atomic checkpoint reserve remain binding. Terminal interpretation must keep
STOP_FOR_COST/partial truth and separate warmed/capture-off timing from cold
first-pass cost. No scientific winner follows from the first-pass differences.
The [startup interpretation](startup_interpretation.md) corrects the frozen
protocol's PPGT warmup shorthand and explains why a literal training START line
is not expected from the epoch-only logger. Actual compiled recipes are unchanged.

The existing Luna B heartbeat has one active30-minute schedule for
[Kaggle2](kaggle2_monitor.json) and [Kaggle3](kaggle3_monitor.json). Initial ticks
authenticated both actual owners and returned only SILENT healthy observations.
Terminal events are delivered idempotently to the same A without overriding A's
model/reasoning. One terminal job does not pause monitoring for the other three.
Large outputs and operation journals remain outside Git. Original partial and
failed histories are retained, not relabeled as this fresh20-pass benchmark.

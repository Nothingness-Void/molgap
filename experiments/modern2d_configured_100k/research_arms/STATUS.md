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
formal training remain unobserved; there are no MAEs or Replay claims yet.

The existing Luna B heartbeat has one active30-minute schedule for
[Kaggle2](kaggle2_monitor.json) and [Kaggle3](kaggle3_monitor.json). Initial ticks
authenticated both actual owners and returned only SILENT healthy observations.
Terminal events are delivered idempotently to the same A without overriding A's
model/reasoning. One terminal job does not pause monitoring for the other three.
Large outputs and operation journals remain outside Git. Original partial and
failed histories are retained, not relabeled as this fresh20-pass benchmark.

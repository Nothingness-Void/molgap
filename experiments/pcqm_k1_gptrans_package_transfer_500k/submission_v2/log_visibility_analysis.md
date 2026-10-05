# Active invocation log visibility

## Observed facts

On2026-10-05, the authenticated Kaggle1 page for kernel ID137144136/version2
showed RUNNING for8684.9seconds, T4x2, output0B and12main-log rows. The last
row was Python3.11 virtual-environment creation at8.9seconds. The observed
run-editor link identifies script version355380593. The credential-bound CLI
also returned RUNNING, an empty output-file list and no log body.

The uv deprecation warnings did not terminate environment creation. These
observations do not establish whether later dependency installation, CPU
release, GPU qualification or formal training has progressed.

## Confirmed implementation defect

The frozen bootstrap uses quiet pip installation, redirects CPU validation to
`cpu_model_cache_recipe.log`, and redirects each preflight/training worker to
its own file with `stdout=log, stderr=STDOUT`. It waits for exit without forwarding
those files or printing phase/progress events to the main log. The owning trainer
does print progress every500batches, but these lines are hidden from the Kaggle
main log. Even healthy training can therefore leave the visible log at8.9seconds.
RUNNING and output0B are not proof of either healthy training or a stall.

## Required later infrastructure change

Keep the active source/package frozen. Before a later invocation, reuse the
platform runtime owner to retain raw worker logs while forwarding tagged live
output to the main log; print timestamped phase boundaries, interpreter/hardware
identity and per-arm completed updates/epochs. Give bootstrap installation and
qualification explicit time bounds and record their failures truthfully.
Do not change the scientific recipe, invent progress, or duplicate a running
job to compensate for missing observability. Review retained stage artifacts
and exact native cost before continuation or retry.

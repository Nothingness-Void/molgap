# Attempt 002 Kaggle1 launch

The existing Kaggle accelerator adapter submitted the paired T4x2 kernel.
Kaggle assigned `nothingnessvoid/molgap-geometry-channels-100k-s42-v2`
(ID 136152423). The authoritative status query after submission returned
`QUEUED`. The pulled script matches the frozen local entrypoint after
line-ending normalization, and Kaggle reports exactly the frozen private
source dataset plus the accepted fixed-100K graph dataset. The API reported no
invalid dataset, kernel, or model sources.

The [launch observation](kaggle_observation.json) and
[bound platform response](platform_response.json) identify the actual kernel,
source commit, package and Spec. The shared experiment CLI launch receipt is
under `receipts_submitted/`. Queue acceptance establishes neither GPU runtime
qualification nor scientific acceptance. Each arm still requires its own
terminal evidence and replay qualification.

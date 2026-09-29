# GPTrans V5 audit-reference execution status

Observed 2026-09-29 20:38 UTC. The user authorized exactly one matched 100K
baseline, not a candidate or full-scale run. Source archive
`af94a63c3c4626ceec8af4106aad0ea97d398e40aefb04c51b15356b994137a3`
was built from scientific-source commit
`21f70ab93c3b0ddc4745316daf8459df40efdb48`; the private Kaggle2 source
dataset `kaseichou/molgap-gptrans-v5-audit-source-v1` is ready at version 1,
and a downloaded `source_payload.bin` matched the archive SHA. The original
initial-state dataset and accepted fixed 100K graph dataset are ready.

The first physical submission,
[`kaseichou/molgap-gptrans-v5-audit-reference-seed42-segment1`](https://www.kaggle.com/code/kaseichou/molgap-gptrans-v5-audit-reference-seed42-segment1),
had the wrong title-derived slug and assumed Kaggle would publish additional
kernel package files. Remote source pull showed only script and metadata,
and its mounted dataset list omitted the new source. Its frozen SHA/function
gate stopped before training with `ERROR`; keep it as a separate infrastructure
attempt, never a scientific result.

The corrected first 10-epoch segment,
[`kaseichou/molgap-gptrans-v5-audit-reference-s42-v1`](https://www.kaggle.com/code/kaseichou/molgap-gptrans-v5-audit-reference-s42-v1),
was submitted as version 1 with requested `NvidiaTeslaT4`. Remote metadata
ID `136421626` confirms the exact kernel slug, T4 machine shape, and three
mounted datasets (V5 source, V4 initial state, fixed 100K graphs). Scheduler
version 1 completed 10 epochs and passed streaming artifact/trace acceptance.
The accepted checkpoint is `74ce6877...ec7756d`, with 7,810 optimizer steps,
999,680 sample presentations and 0.7927923 allocated T4-device-hours (T4x2,
one used). Its best early EMA development MAE is 0.6528019 eV at epoch 3;
this is not the final reference. The live development MAE at epoch 9 is
0.215775 eV. Its first-ten-epoch live-train/EMA curve closely tracks the
historical P100 V4 baseline, supporting execution parity but not exact
byte-identical weights. The compact [acceptance](results/segment_01_acceptance.json)
and [operation receipt](results/segment_01_operation.json) bind the observed
facts. The prospective trajectory is [`rml_plan/trajectory.json`](rml_plan/trajectory.json).

The same kernel's version 2 completed epochs 10–19 and passed streaming
artifact/trace acceptance at cumulative epoch 20. Its accepted checkpoint is
`8cd4f8a9...f8f90d`; the source archive, fixed manifest and runtime certificate
match version 1. The second segment used 0.8014082 allocated T4-device-hours
(T4x2, one used), bringing accepted segment cost to 1.5942004 allocated
device-hours, below the 20-hour snapshot ceiling. Its best-so-far EMA
development MAE is 0.3450094 eV at epoch 19; this is not a final reference.
The compact [acceptance](results/segment_02_acceptance.json) and
[operation receipt](results/segment_02_operation.json) bind this observation.

The same kernel's version 3 completed epochs 20–29 and passed streaming
artifact/trace acceptance at cumulative epoch 30. Its accepted checkpoint is
`5e2cf00c...24f518ab`; source archive, fixed manifest and runtime certificate
remain identical. The third segment used 0.8680383 allocated T4-device-hours
(T4x2, one used), bringing accepted segment cost to 2.4622387 allocated
device-hours, below the 20-hour snapshot ceiling. Its best-so-far EMA
development MAE is 0.2494885 eV at epoch 29, not a final reference. The
compact [acceptance](results/segment_03_acceptance.json) and
[operation receipt](results/segment_03_operation.json) bind this observation.

The same kernel's version 4 completed epochs 30–39 and passed streaming
artifact/trace acceptance at cumulative epoch 40. Its accepted checkpoint is
`7e434952...b846f280e`; source archive, fixed manifest and runtime certificate
remain identical. The fourth segment used 0.9194067 allocated T4-device-hours
(T4x2, one used), bringing accepted segment cost to 3.3816454 allocated
device-hours, below the 20-hour snapshot ceiling. Its best-so-far EMA
development MAE is 0.1993127 eV at epoch 39, not a final reference. The
compact [acceptance](results/segment_04_acceptance.json) and
[operation receipt](results/segment_04_operation.json) bind this observation.

The same kernel's version 5 completed epochs 40–49 and passed streaming
artifact/trace acceptance at cumulative epoch 50. Its accepted checkpoint is
`d6902f1e...59a103abc`; source archive, fixed manifest and runtime certificate
remain identical. The fifth segment used 0.8047330 allocated T4-device-hours
(T4x2, one used), bringing accepted segment cost to 4.1863784 allocated
device-hours, below the 20-hour snapshot ceiling. Its best-so-far EMA
development MAE is 0.1697378 eV at epoch 49, not a final reference. The
compact [acceptance](results/segment_05_acceptance.json) and
[operation receipt](results/segment_05_operation.json) bind this observation.

The private progress dataset `kaseichou/molgap-gptrans-v5-audit-progress-v1`
version 5 supplied the SHA-verified continuation input. Kaggle's whole-dataset
download returned 404 despite ready status; per-file retrieval recovered the
six files and verified the resume payloads. The same kernel's version 6 then
completed epochs 50–59. The final segment passed streaming checkpoint/trace
acceptance, and the 50K prediction content, model hashes, selection, and role
flags passed independent result acceptance. All six segments have the same
fixed manifest, source archive, and runtime certificate; each allocated two
T4s and used one. The measured total is 5.0155572684 allocated T4-device-hours.
The accepted EMA development MAE is 0.1560144881 eV at epoch 59. The exact
[final acceptance](results/final_acceptance.json), [final segment operation](results/segment_06_operation.json),
and [decision](decision.md) retain the evidence and scope. No successor was
submitted. Strict RML terminal/reference-bundle packaging remains separate
from the completed artifact and trajectory checks; neither G1 nor G2 is
released by this baseline.

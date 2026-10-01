# K1 node-width qualification and single-candidate screen

Date: 2026-10-01. Owner: desktop. User authority: "run once at 256" and "continue".

## Question

Does increasing the K1 atom state from 192 to 256 improve Gap prediction under
the original clean V4 100K exposure? This is a capacity intervention; an endpoint
gain alone would not identify an information-bandwidth failure mechanism.

## Fixed design

Only atom hidden channels change. Keep nine layers, edge width 64, one K1 slot
of width 64, original gated aggregation, readout, categorical stem and RWSE16.
No auxiliary task, pretraining, ensemble, SSMA or geometry is added.
Expected parameter counts: 3,658,817 at 192; 6,035,201 at 256; ratio 1.6495.
The user ceiling is twice the baseline count (7,317,634).

The downstream screen is one seed (42), Gap in eV, fixed official-training-only
100K training prefix and 50K internal development rows. Preserve FP32, batch
128, drop-last, two loader workers, AdamW lr 0.0004, weight decay 0.00001,
40 epochs and the family-owned cosine schedule. Expected exposure: 31,240
optimizer steps and 3,998,720 sample presentations. Family recipe construction,
not this prose, owns executable values. Freeze a complete width-256 initial
state; do not load or pad a 192-dimensional checkpoint into the candidate.

## Qualification action A001

Publish the prospective qualification trajectory before constructing models.
Use CPU only, seed 42. Freeze the full candidate initial tensor state and its
file/state SHA256, count parameters, and confirm reference defaults remain
unchanged. No molecular rows or labels are accessed by initialization.
Required syntax/import/recipe/source checks reuse existing family owners.
GPU real-shard repeatability, resume, selected-state roundtrip and allocation
checks remain release/runtime requirements, not evidence of accuracy.
Runtime overhead is measured and reported; the SSMA-specific 25% veto is not
an appropriate gate for this expressly authorized capacity intervention.

Bound: one local CPU qualification action, at most 0.25 CPU/wall hours; zero
accelerator allocation. No baseline training, successor seed or scale-up is
authorized by this record. Missing cost measurements stay unknown.

## Training release and comparison

Use the registered modular workflow and Kaggle adapter. One candidate arm
uses one assigned T4; record the actual two-T4 allocation separately from useful
device time. A separate training prospective plan freezes its budget after
reference qualification and runtime evidence are available.

The retained old K1 V4 reference is currently unsuitable for a strict causal
comparison: its trace manifest lacks checkpoint_identity and raw checkpoints
were not locally retained. Do not fabricate trace events or silently retrain
192. The independently authorized K1/SSMA pair already contains a new 192
reference. Its scheduler state and eventual artifacts must be reconciled and
accepted by their owner before use. A RUNNING observation is not qualification.

A missing strict reference blocks this workflow's training release. Preserve
the prepared candidate and its qualification without relaxing that gate.
Training acceptance, paired uncertainty, exposure/runtime/role/cost agreement
and comparison class retain the existing Research Protocol owners. One seed
cannot estimate training stochasticity. No automatic promotion or scale-up.

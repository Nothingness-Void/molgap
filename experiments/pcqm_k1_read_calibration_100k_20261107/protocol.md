# Current-layer associative read — same-job calibration

2026-11-07 JST. One user-authorized Kaggle1 nothingnessvoid T4x2 pair,
original Muon versus the fixed current_read side path. This new question
calibrates the side path against original K1; it does not repeat or reopen
the current-versus-history comparison.

## Evidence and choice

Desktop/server RML lookup for read and linear mechanisms was run with
origin/molgap-server. The server replacement decision at
origin/molgap-server:experiments/pcqm_k1_linear_attention_100k/decision.md
closed a different operator: it replaced original slot mixers under AdamW;
its selected 100K delta was +0.398949 meV. That negative is preserved.
The accepted original-Muon reference is
../pcqm_k1_muon_100k_20261106/decision.md: 0.1325206676620245 eV,
40 passes, Live selection; process wall11177.999007s. Historical numbers
are context, not a qualified same-job comparison for this addon.

The original current/history contract and module belong to owner branch
codex/exp/k1-history-read-100k-20261106 at64c3debce85b47bc7a15fd4db10173ed0e4cabec,
experiments/pcqm_k1_history_read_100k_20261106/protocol.md; they are not
represented as integrated desktop evidence. Job143027919/version1 strict
selective retrieval is retained at
D:/molgap-artifacts/retrieved/pcqm-k1-history-read-100k-20261106-s42/v1/version1.
Its original launch_receipt.json is under the corresponding
D:/molgap-artifacts/runs/pcqm-k1-history-read-100k-20261106-s42/v1/operations/.
The retrieved diagnostic is reports/diagnostic_15cad9c279e8f6d2e6bb9df0b2a0edd2009347100a6aacef3ba4908e5edc1600.json
under that run. It reports current selected0.130144503669 eV at pass36,
history0.130040541252 at pass38; history-minus-current -0.103962meV,
paired95%interval[-0.686142,+0.503524]meV and last-five -0.074071meV.
History's early mean gain -1.36165meV shrinks late. This fails a material
history gain and supports selecting the simpler current-only side path.
It does not prove a same-job gain versus original Muon: that control was
absent. The apparent historical current-versus-Muon gain is below3meV.

## Frozen difference and reuse

Both fresh arms use the same original frozen random K1 tensors (not a trained
checkpoint), seed42, fixed100K train/50K internal development and full40passes.
The only scientific difference is registered k1_current_read version1 versus
no addon. Source src/molgap/k1_history_read.py is copied byte-for-byte from
the original owner, SHA256
2e712058fe694652c38ac230df036098264c1bf40b036d8fba7adebe3d08d80c.
Only k1_current_read is registered here; config is exactly
{history:false,seed:42}. Original mixers remain intact. The existing shared
PRE_MESSAGE side path at layers3,6,9 uses affine-free LN192, qkv192,
4heads x16, ELU+1 q/k, per-graph S=sum(k outer v), Z=sum(k),
r=qS/max(qZ,1e-6), output64->192 and dropout.05. Weights are shared across
taps; output begins at1e-3 of the same Torch initialization. No layer-history
accumulation is enabled. This adds49152 parameters, not an identity map.
No new trainer, verifier, scheduler, tests or framework is introduced.

## Recipe, roles and budget

experiment.json reuses the immutable k1-post-bn-100k-v1 preset plus the fixed
k1-muon-adamw-fp32-v1 optimizer for both arms. BS128/drop_last, single forward,
FP32/noTF32/native Torch2.11, LR4e-4/WD1e-5/clip1, cosine40 to1e-6 without
warmup,31240updates/3998720presentations. Best-development Live selection;
EMA.999 is tracked, with no Clean-BN selection. Existing checkpoint/resume,
profiling and standard epoch logs remain frozen. No generated3D is built.
Train/internal development only; official validation, test-dev and challenge
are unauthorized. No protected-role reading before freeze.

One actual worker per T4;5h allocation wall/10T4-device-hour ceiling and
180s cleanup reserve. Estimated4h/8T4-device-hours is not a measurement;
prior current worker wall12273.450884s is context. Retain atomic resumable
state and durable outputs at a wall stop; no automatic extra session.
Credentials stay external and are passed by path without reading contents.

## Gate and completion

The registered submit flow must publish prospective trajectories before any
new diagnostic or training. Preserve exact source/config/input identity,
native repeat/resume proof, aligned finite50000-row predictions, complete
trace, selected checkpoint, durable outputs, actual native cost and observed
role audit. Both arms need strict mechanical/replay acceptance; same-job
reference is muon, never the historical endpoint.

Nomination requires current_read-minus-muon selected Live MAE<=-0.003eV,
paired2000-bootstrap95%upper<0, final delta<0 and mean of last-five
matched-pass deltas<0. Require<=2x original model parameters; report actual
step/process/allocation/memory cost separately. Row uncertainty is not seed
uncertainty. Below-gate gains do not nominate. Invalid evidence remains
INCONCLUSIVE; a cost stop remains STOP_FOR_COST. Inspect aligned full curves
before attributing fitting or generalization causes. Use shared two-arm closure
and its single RML rebuild, then write attribution. No automatic500K/full,
adoption, retries, blind grid or successor is authorized by this contract.

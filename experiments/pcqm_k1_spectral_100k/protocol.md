# K1 single spectral residual 100K contract - 2026-10-06

User authorized two two-arm overnight 100K experiments on Kaggle3 and shutdown
only after verified submission. This question explicitly trains a fresh matched
reference and spectral candidate. Desktop retains custody; no server takeover.

Reference is original pure2D K1 node192/edge64/slot64, active slot1, nine local
blocks, RWSE16, mixers3/6/9 and mean pooling (3,658,817 parameters).
Candidate adds one residual after local block6, before its existing slot mixer.
Down-projection192->64, full symmetric normalized unweighted real-bond Laplacian
EVD, learned8 Gaussian frequency bases centred uniformly0..2, width0.25,
per-channel64 coefficients, inverse spectral projection, up-projection64->192.
The up projection starts at zero; the frequency transfer starts at identity.
This is a compact S2GNN-inspired intervention, not a paper reproduction.
Full V retains repeated eigenspaces; filter depends on eigenvalue, not vector
identity. Gaussian filters are generally nonlocal, not finite-degree polynomial
message passing. No positional embeddings, geometry, target change, auxiliary
loss, new readout, dropout change or teacher is added. Addon ~25K parameters,
well below the user cap7,317,634. Parameter count is checked in remote preflight.

CPU preparation has its own prospective record before graph processing. Original
accepted fixed100K shards are hash validated, geometry fields stripped by the
existing loader. Precompute only topology train0:100000/internal-dev100000:150000,
full EVD float64CPU then float32 storage. Freeze tensor/manifest hashes in the
formal Spec addon config; stage private immutable input via registered owner.
The GPU worker only reads/validates this cache, never computes eigensystems.
No official-valid/test-dev/test-challenge or other roles are released.

Both arms seed42, exact retained initial backbone state, original Python epoch
order, FP32/noTF32/deterministic, BS128/drop-last, AdamW lr4e-4/wd1e-5/clip1,
cosine40/eta1e-6,40epochs,noEMA, normalizedGapL1, best internal-development
live checkpoint. Each arm31240steps/3998720 optimizer-row presentations.
Complete endpoints required; no changed schedule or calibrated early stop.

All-arm remote barrier reuses family train-onlyBS128 qualification. Require
zero-added equivalence, finite nonzero addon gradients after2updates, numerical
repeatability, model/AdamW/scheduler/RNG resume, selected-state reload, node
permutation and repeated-eigenspace basis invariance for trained addon,
parameter<=2xreference, optimizer-step overhead<=25%, peakmemory<=12GiB.
Qualification failure is infrastructure/NO_TRAIN, not a scientific negative.

Primary material gate: new-reference MAE minus candidate MAE>=0.003eV and
positive lower95% paired row-bootstrap interval (1000draws,seed42). Single-seed
training variance remains unknown. Fixed internal development is already
selection consumed; this is a bounded mechanism screen, no READY/promotion.
Cross-cohort transfer remains a later independently authorized gate.

Expected each arm<=4 assignedT4h,total<=8T4h; CPUpreprocessing estimated<=0.25h,
not measured until retained manifest. GPU diagnostics measured separately.
Queue/bootstrap/CPU/device scopes remain separate and unknown is not zero.
Retain atomic epoch resume states, selectedweights, trace, predictions, row
identity, certificates, costs and exact kernel/version durable manifests.
Negative closure requires attribution and archive custody; positive requires
independent review, no automatic500K/full/evaluation/promotion.

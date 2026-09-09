# EdgeState local-geometry pretraining SCNet protocol

## Question

Can local ETKDG bond-length and bonded-angle supervision improve the accepted
pure-2D EdgeState GPS9 without changing its inference architecture?

This is a training-only optimization of an already validated architecture. The
user authorized a new SCNet experiment after cancelling the unstarted SCNet
duplicate of the local chemistry-hierarchy screen. It therefore enters the
frozen PCQM-100K contract directly. It is not a new inference architecture and
does not duplicate the separately running Kaggle chemistry-hierarchy task.

## Evidence and isolated mechanism

The earlier graph-level ETKDG denoising experiment predicted whole-molecule
distance/angle histograms and moments and was negative. That result did not test
local geometry supervision. Distance-plus-angle bottom fusion improved the
matched encoder at three seeds, while adding torsion and arbitrary global state
usually regressed. The isolated hypothesis is therefore that training-time
local bond and angle targets can shape the existing sparse chemical states while
avoiding a 3D inference branch.

The model input remains OGB atom/bond categories, real-bond topology, and
RWSE16. A training-only head predicts each valid ETKDGv3+MMFF94s directed bond
length from the final 64-channel EdgeState. A second training-only head predicts
each valid bonded-angle cosine from the two adjacent EdgeStates and center-node
state. Gap labels are not read during pretraining. Both heads are discarded
before Gap fine-tuning and inference.

## Frozen paired contract

- accepted PCQM train-derived cache: 100,000 train / 10,000 internal validation;
- geometry aggregate SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`;
- pure-2D graph parent SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`;
- inference model: `ogb_edge_state_structural_gps9`, 4,771,073 parameters;
- seed 42, FP32, physical batch 96, AdamW `1.6e-4`, weight decay `1e-6`,
  gradient norm clip 1.0, cosine decay to `1e-6`;
- scratch arm: 40 direct-Gap epochs;
- candidate arm: 20 local-geometry epochs then 20 direct-Gap epochs;
- identical initialization and exactly 40 encoder passes in each arm;
- official validation, test-dev, shadow, external rows, and the molecular-
  research server remain unread.

Batch 96 is the accepted Xi'an exploratory comparison contract. Each replicate
runs both arms sequentially on one DCU; cross-device arms are never compared.
Every epoch atomically stores model, optimizer, scheduler, complete trace, RNG,
and DataLoader shuffle state. Retries must resume or fail closed.

## Decision and platform boundary

Two independent Xi'an paired replicates are submitted after one real-cache DCU
forward/backward preflight. The mechanism is nominated only if both accepted
replicates improve by at least `0.003 eV`. Xi'an is non-authoritative because
its old HIP/scatter reductions are not bitwise repeatable; a nomination only
permits one canonical paired batch-48 Kunshan confirmation under a separately
frozen protocol.

Failure in either replicate closes this exact mechanism without changing seed,
width, target, batch, objective weights, optimizer, schedule, or horizon. No
result here authorizes shadow access, 1M/full training, official evaluation,
production changes, or a second architecture.

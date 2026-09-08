# EdgeState Local-Hierarchy Pretraining PCQM-100K Protocol

## Question

Can locally attached atom, real-bond, and chemistry-defined functional-group
reconstruction improve the already validated pure-2D EdgeState GPS9 model under
an equal encoder-exposure budget?

This is optimization of an accepted model, not architecture discovery. The user
therefore authorized a direct PCQM-100K entry rather than QM9 triage.

## Frozen data

- official PCQM4Mv2 training prefix only: 3,378,606 rows;
- accepted internal roles: 100,000 train and 10,000 validation graphs;
- parent geometry-cache aggregate:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`;
- parent pure-2D graph aggregate:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`;
- official validation and test-dev remain unread;
- no shadow role is read during seed-42 training.

The GPU model consumes only OGB 9-field atom categories, OGB 3-field real-bond
categories, topology, and RWSE16. Existing ETKDG fields in the accepted cache
are ignored. A CPU-only sidecar attaches twelve SMARTS-derived atom labels to
the same row and atom identities; it does not alter the graph split or Gap
targets.

## Frozen inference model

Both workers use `ogb_edge_state_structural_gps9`:

- 9 layers, hidden width 192, 4 attention heads;
- persistent 64-channel real-bond EdgeState;
- RWSE16, mean pooling, direct scalar Gap head;
- dropout 0.1;
- exactly 4,771,073 inference parameters.

Training-only reconstruction heads are discarded before Gap inference.

## Paired training

One Kaggle T4x2 job isolates one worker per GPU. Both models start from a
bit-identical seed-42 encoder state and use FP32, physical batch 48, AdamW,
learning rate `1.6e-4`, weight decay `1e-6`, and cosine decay to `1e-6`.

- scratch: 40 epochs of direct normalized-L1 Gap training;
- local hierarchy: 20 train-only reconstruction epochs followed by 20 epochs
  of direct normalized-L1 Gap fine-tuning with a fresh optimizer;
- each arm receives exactly 40 encoder passes over the 100K train role;
- no early termination is allowed in this paired question; the best internal
  validation checkpoint is retained from each fixed budget;
- atomic checkpoint, trace, best model, and validation tensor payload are
  written independently for each worker.

Pretraining masks 15% of atoms and 15% of undirected real bonds. The losses are
the mean categorical reconstruction loss across all nine atom fields, the mean
categorical reconstruction loss across all three bond fields, and 0.5 times
the multilabel functional-group loss at masked atoms. Gap labels are not used
during pretraining.

## Decision gate

No historical metric decides the result; the fresh paired scratch arm is the
comparator. Seed 42 nominates the method only if local hierarchy improves
internal-validation Gap MAE by at least `0.003 eV`. Nomination does not
authorize extra seeds, shadow reading, 1M/full training, official validation,
test-dev, desktop handoff, or molecular-research-server access.

Later scale-up must obey `scale_up_protocol.md`: only eligible train-row count
may change across 100K, 1M, and full.

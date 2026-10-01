# G1 recipe and path-endpoint falsifiers

Frozen question: does either one isolated intervention improve accepted degree-scaled GPTrans G1 with EMA0.999? The comparator is the accepted terminal `degree_scale_ema999` candidate of `pcqm_gptrans_input_ema_100k`, enrolled as a hash-bound reference view without retraining, inventing observations, or duplicating cost.

## Two independent arms

1. `degree_group_decay_ema999`: AdamW weight decay is zero for biases and 1D tensors; matrices and embeddings retain weight_decay=0.05. Architecture, EMA, initial tensors and all remaining optimization fields match the comparator. An optimizer comparison, not an architecture claim. Record group inventory/digest, preclip norms, clip frequency and epoch-end weight/Adam moment norms.
2. `degree_path_endpoints_ema999`: preserve the existing pair initialization and add half the contrast between shortest-path first and last chemical bond embeddings. Average the first bonds of all neighbors belonging to equally short paths, not an arbitrary selected path. Apply only distances 2..20. Reversal is antisymmetric and relabeling is permutation-covariant in exact arithmetic. No new parameters, 3D, cache, or path-dependent RNG. This is a minimal endpoint-order encoding, not a full Graphormer/GPTrans path-sequence reproduction. It cannot distinguish every permutation of internal bonds.

The first arm targets optimizer regularization of degree/norm/bias parameters; the second targets mean-path encoding's loss of endpoint arrangement. Neither intervention is combined with the other. The prior path-mean + degree arm did not establish additivity; this motivates an order-sensitive, nonduplicate falsifier, not a promised gain.

## Fixed contract

Accepted cross-platform PCQM fixed assets only: 100,000 train and 50,000 internal development rows. Seed42, physical batch128 per model/device, FP32 and TF32 disabled, 60 epochs, 46,860 optimizer steps and 5,998,080 sample presentations. Direct Gap; frozen target transform, accepted degree initialization, lr0.001, existing warmup/cosine schedule and clipping. EMA0.999 on both arms. No official validation, test-dev, test-challenge, external submission, shadow evaluation, new geometry or pretraining.

One Kaggle3 T4x2 kernel; one isolated worker/model/RNG/optimizer/atomic checkpoint directory per GPU. Existing scheduler, native trainer, acceptance and RML closure are reused. Estimated 4 wall hours / 8 allocated T4 hours; hard cap 6 wall hours / 12 allocated T4 hours including idle devices. No automatic successor, seeds43/44 or scale release.

## Release and acceptance

Require source/recipe/inventory SHA binding, real reference evidence, immutable prospective trajectories, separate intervention-purpose declarations, accepted runtime calibration and frozen initialization before compute. Toy CPU checks must cover equal-mean distinct arrangements, equal-distance ties, relabeling covariance, reversal, finite gradients and exact parameter grouping; these do not constitute all-data or GPU validation.

After completion, saved-artifact acceptance must verify both arms' identity, row/target/prediction alignment, finite full traces, protected-role flags, actual device allocation/cost, models/checkpoints/chunks and hashes. Finalize each terminal transaction and build RML replay; both candidate/reference pairs must enter their correctly qualified comparison worlds. A training completion alone is not Replay-Ready.

Decision: compare EMA-selected endpoint predictions with the accepted comparator. Material advancement for this screen requires improvement >=0.003 eV and paired bootstrap95% upper delta<0. This is this experiment's conservative policy, not a universal V5 scientific threshold. Smaller positive evidence is retained below gate; negative evidence closes only the exact intervention. Interpret train/dev/EMA curves and optimizer diagnostics before proposing further changes. One seed does not establish seed stability or scale transfer.

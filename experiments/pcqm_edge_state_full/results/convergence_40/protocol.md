# EdgeState Convergence Continuation

This is an isolated diagnostic continuation of the accepted rich-feature
full-data run. It starts from the immutable
rich_full/training/last.pt and keeps the original OGB graph cache, split,
seed, model width, depth, and optimizer state.

The continuation resets only the learning-rate schedule to a recorded
20-epoch warmup-cosine phase from 1e-4 to 1e-6. It writes to a new output
directory and checkpoints every completed epoch atomically. The source
checkpoint, source best checkpoint, and graph-acceptance hashes are embedded
in the continuation contract.

The result is a convergence diagnostic, not an architecture claim. It does
not use official test labels, external data, pretraining, or the production
registry. The run closes after 20 additional epochs, seven non-improving
epochs, or the 13.5-hour hard job budget. A decision record is added only
after the remote artifacts are retrieved and independently verified.

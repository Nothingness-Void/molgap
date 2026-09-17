# MolGap V5 TPU Preflight

This bounded platform diagnostic asks whether Kaggle TPU v5e-8 can execute the
current MolGap PyTorch/PyG operator path. It mounts the accepted Kaggle3 V5
runtime and fixed 100K training cache, verifies their identities, checks JAX
and PyTorch/XLA device discovery, runs a dense XLA backward step, and attempts
one reduced GPTrans forward/backward step on eight training graphs.

The probe does not train a scientific model, select hyperparameters, read the
development role, or modify a V4 comparator. Failure closes direct TPU use for
the current stack until a separately justified adapter is implemented.

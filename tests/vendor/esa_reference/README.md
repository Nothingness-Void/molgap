# Pinned ESA test reference (MIT)

Upstream: https://github.com/davidbuterez/edge-set-attention/tree/861a0b12d445ae80cc69b3948af623b4331321d4

Copyright David Buterez2024. The included LICENSE applies to these reference
files and to the locally adapted ESA equations. Raw mha.py, masked_layers.py,
mlp_utils.py and norm_layers.py are byte copies of that commit. They are test
assets only; source bundles never include them or the upstream trainer.

estimator_forward.py.txt is the AST-identical Estimator.forward method from
esa/models.py at the same commit, with indentation/comments/whitespace adapted.
No optimizer, trainer, dataset or Lightning implementation is copied.
The test harness checks raw source hashes and substitutes CPU SDP/SwiGLU math
and device plumbing only. Pre-LN, FP32, unpadded configurations are its scope;
this is not native xformers/flash parity or the full published benchmark.

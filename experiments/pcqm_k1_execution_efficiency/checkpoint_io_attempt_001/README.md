# Checkpoint metadata-copy diagnostic

Desktop NO_TRAIN continuation of the execution review. Inspect the frozen
500K sender's `_cpu`, `_snapshot`, `_save` and `_paired`: every128 steps it copies
and publishes the full retained step history. This is a specific unchecked host
cost, not a new model experiment. Original remote source/packages remain immutable.

Reuse a hash-verified44MB scratch model/optimizer/RNG checkpoint from the accepted
local GPU diagnostic, decoded on CPU only. No model/optimizer construction,
gradients, inference, graph/data loading, label/role access or accelerator work.
Use clearly synthetic seven-field telemetry fixtures solely for copy/IO sizing:
0,128,8192,32768,65536 rows per fixture arm. They are not historical training
records, RML training traces or evidence of learning. No fake run/role/cost records.

Compare the sender's exact hash-pinned `_cpu` callable (isolated AST extraction
avoids importing the remote runner) with additive `checkpoint_to_cpu`. Both
retain tensor detach/CPU/clone, container recursion and deepcopy fallback;
the candidate only short-circuits immutable exact builtin scalars. Three balanced
windows per size, repeated decoded-payload equality, CPU AdamW step preservation,
independent snapshot storage and unchanged input/RNG checks. Atomic torch/JSON
publication and SHA256 reuse existing owners; no format/frequency or log removal.

CPU4threads,180s cooperative worker ceiling. Record clone and full publication
separately; timer includes actual local writes/hashes, not remote Drive/Kaggle IO.
Do not extrapolate synthetic measurements to a real training speedup. Actual
history length/native checkpoint cost remains unknown without accepted artifacts.
The result may nominate an opt-in helper, never patch an active job or release
quality training. Terminal attribution precedes another mechanism.

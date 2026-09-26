# RRWP-pair seed42 terminal training decision

On September 27, 2026 Kaggle2 kernel 136015256 version 1 completed the frozen
40-epoch FP32/BS128 screen. After correcting the documented acceptance-field
mismatch, saved-artifact verification passed without model inference.

The selected internal-development MAE was **0.1389245242 eV**, versus immutable
K1 at **0.1413736343 eV**: a gain of **0.0024491102 eV**. Candidate-minus-reference
paired 95% row-bootstrap bounds were **[-0.0033276410, -0.0015529428] eV**.
This was positive below the predeclared 0.003 eV material gate, not a promotion.
The bootstrap does not measure training stochasticity or establish independent
generalization on repeatedly used development molecules.

The matched-epoch gain narrowed from 0.009505 eV at epoch 9 to 0.002296 eV at
epoch 39 (zero-based). The best candidate checkpoint was epoch 36. Training
error was lower at each of those endpoints; the early advantage did not
fully persist. These observations do not isolate RRWP from its receiver-pair
parent until that arm is accepted, and they do not prove scale robustness.

The model had 3,681,953 parameters (23,136 more than K1). Mean epoch time was
142.32 seconds on one Tesla T4, with 646 MiB peak training reserved memory.
Kaggle assigned **two T4s despite the P100 request**; only one was used. The
notebook recorded 6,005.77 wall seconds and 12,011.53 allocated device-seconds
including bootstrap and the idle device. Both worker and allocated notebook
cost are retained; allocated device time is not an assertion about Kaggle's
quota-billing formula. No cross-GPU speedup claim is made against the P100
reference.

Training ended without continuation, multi-seed confirmation, scale-up or
protected-role use. The separately predeclared frozen500K internal-development
NO_TRAIN audit remained conditional on acceptance of the study's training
arms. A new training mechanism requires separate prospective authority.

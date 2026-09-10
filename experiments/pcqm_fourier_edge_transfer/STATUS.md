# Status

The QM9 candidate passed its frozen Track C gates. The paired PCQM-100K
implementation passed 31 static and policy tests without local model execution.
Private source dataset `kaseichou/molgap-pcqm-fourier-edge-source` freezes
commit `47f99cf9da7fee306f5165175b4020c6c4aa9fb3`. Kaggle2 T4x2 kernel
`kaseichou/molgap-pcqm-fourier-edge-s42` version 1 completed and passed
no-model acceptance. Fourier-Edge failed its architecture-matched K1 gate and
is closed. The K1 causal control produced a large paired gain over full GPS and
is frozen unchanged for one independent label-sealed shadow audit; no repeat
selection run, extra seed, or new architecture is released. See `decision.md`.

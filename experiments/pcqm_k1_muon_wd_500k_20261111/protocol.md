# K1 batched Muon weight decay at 500K: protocol

2026-11-11 JST. Desktop-owned Track B. Seed arm: S42 (kernel `kaseichou/molgap-k1-muon-wd-500k-20261111-s42`, account `kaseichou`). Companion seed: S43.
Arms: `muon_wd1e5` (control) and `muon_wd5e2` (`training.weight_decay` 0.05), both original K1 `neural_atom_k1/3`
under the registered `k1-muon-batched-adamw-fp32-v1`, same job, one T4 per arm.

## Limits of this experiment

1. This experiment does not reopen the closed Muon-vs-AdamW question; no claim about Muon vs AdamW is made. The control arm re-executes the CLOSED batched-Muon reference identity (source f055c137, k1-muon-batched-adamw-fp32-v1, wd 1e-5) solely as a same-job control.
2. Single predeclared point. No second weight_decay value, no learning-rate change, no post-hoc grid, no module stacking regardless of outcome. NEGATIVE/NULL closes the weight-decay lever for K1 batched Muon at 500K.
3. training.weight_decay applies to both the Muon hidden-matrix group and the AdamW remainder group (parameter_groups); the isolated variable is recipe-level decay 1e-5 -> 0.05 under batched Muon, not a Muon-only decay. The Muon mechanism is the motivation, not what is isolated.
4. NOMINATE authorizes only a separately authorized confirmatory pair against the AdamW reference; no adoption, no full-data, no automatic successor.
5. The control arm's selected MAE of seed 42 versus the read-batched reference 0.101863694 eV is reported as reproducibility description only, with no threshold.

## Hypothesis and evidence

H1: with everything else unchanged, recipe-level weight decay 0.05 instead of 1e-5 lowers the selected-point development MAE by >= 3 meV.
Motivation: the Muon gain over AdamW at K1 500K shrinks with training (-7.93 meV at 20 passes, -4.42 at 40, -1.81 at 60; Muon 500K terminal
record `ev-terminal-T-pcqm-k1-muon-500k-20261106-s42-muon` and its curve in `experiments/pcqm_k1_muon_500k_20261106/`). Alternatives: AdamW catches up with
more exposure; 0.05 decay is too strong and underfits; the change also decays the AdamW remainder group; seed or execution specificity.
The control reruns the closed identity of `experiments/pcqm_k1_read_batched_muon_500k_20261107/` (muon arm, selected 0.101863694 eV) as a same-job control only.

## Fixed conditions

Fixed 500K dataset as in the read-batched experiment: TRAIN[0,500000), development[500000,550000), pure 2D, direct PCQM Gap, OGB atom9/bond3/RWSE16,
manifest SHA256 630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751. Initial state SHA256 8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd
(frozen seed-42 K1 state shared with the read-batched reference). Sampler/dropout seed stays 42 (the compiler permits only the seed label 42; same precedent as the init-noise and
GPTrans WD s43 runs). Batch 128 drop-last, torch-pass-randperm, FP32/noTF32, Torch 2.11, normalized L1, clip 1, LR 4e-4, no warmup,
cosine_epoch 60 passes to 1e-6, EMA 0.999 tracked, selection best-development-live. 60 passes = 234360 updates = 29998080 presentations per arm.
Official validation, test-dev and test-challenge remain sealed; only TRAIN and the internal development role are authorized.

## Reading and decision (preregistered, joint over both seeds)

Per seed: development 50K saved predictions; Delta_s = candidate minus control selected-point MAE; paired 2000-resample row bootstrap (seed 42) 95% interval.
Offline tail_holdout_v1 is secondary and descriptive only, with no gate; first verify with `--dev-range 500000:550000` that its development MAE matches the trace.
With m = (Delta42 + Delta43)/2:
- NOMINATE: Delta42 <= -3 and Delta43 <= -3 meV, both 95% uppers < 0, both final and last-five matched-pass deltas < 0, both kernels strict native/replay VERIFIED.
- INCONCLUSIVE_SMALL: both Delta < 0 and -3 < m <= -1.5 meV.
- INCONCLUSIVE_NOISE: Delta signs differ.
- NEGATIVE: both Delta >= 0, or m >= +1.5 meV with both Delta > 0; closes the lever.
- NULL: all other cases.
Single-seed termination (the other kernel not BOUND or not complete): report per-seed only, status INCOMPLETE_PAIR; a single seed never nominates.
Whether to run one seed must not depend on the other seed's result. Row uncertainty is not training-seed variance.

## Execution, durability and cost

Two physical sessions per kernel; planned pause 187488 (pass 48) then terminal 234360; no scientific schedule reset; 60-pass cosine unchanged;
continuation submitted by the desktop owner next morning; no unattended polling service. Each session entry wall <= 9h (32400 s), cleanup reserve 180 s;
cumulative <= 18h and 36 T4-device-hours per kernel. If a wall limit stops a session earlier, resume only the exact retained state
(optimizer/scheduler/RNG/EMA/counters) inside the remaining budget. Protected roles: none authorized, none opened.

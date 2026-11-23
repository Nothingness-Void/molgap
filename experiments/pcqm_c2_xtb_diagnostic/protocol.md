# C2 GFN2-xTB residual diagnostic (pilot gate + scaffold & block-transfer check)

Track B, local only, 2026-11-23. Authorized local execution of item 6 (C2) from
the supplied `final_ruling.md`. No backbone training, remote job, tail,
official validation/test, DFT 3D coordinates or model promotion is released.

## Question

Does semi-empirical quantum chemistry (`GFN2-xTB` single-point on `ETKDGv3 +
MMFF94s` 3D conformers generated purely from 2D SMILES) carry residual signal
for the 500K GPTrans-T PairNorm pair (`init_s43`, `init_s44`):

1. **Primary (`final_ruling.md` C2 gate):** beyond P0's strong 2D descriptor +
   Morgan stacker (`P0_2d`) on P0's exact 30,155 / 19,845 scaffold-disjoint
   development split (`[500000, 550000)`). Note: `final_ruling.md` originally
   named `S1 + Hückel`, where Hückel was already self-falsified at -0.14 to
   -0.35 meV; testing beyond `P0_2d` is slightly more lenient (a fail against
   `P0_2d` automatically fails against `P0_2d + Hückel`; a pass would still
   require checking Hückel).
2. **Predeclared secondary (block-transfer check):** because earlier studies
   showed `P0_2d` fails outside its fit block, test whether a low-dimensional
   physics-only corrector (`ŷ + 20 xTB/MMFF features`, no 1,247 2D descriptors
   or fingerprints) fitted on P0's 30,155 development rows generalizes to the
   distant unseen block `far` (`[3200000, 3250000)`, 50,000 rows, reusing
   `block-transfer`'s sealed raw predictions).

## Frozen design

- Tooling: official `grimme-lab/xtb v6.8.0` Windows binary
  (`D:/molgap-artifacts/tools/xtb-6.8.0/stage/bin/xtb.exe`, SHA256
  `9e825df458f43e2bb4b48a58a269cc6c40d2228ecc127eb23bd7b87373c8c0a1`, archive
  SHA256 `abd7f74d60908925a7e88a030f83d448253ff4e087451f9b13aeda755eb7b98d`).
- Features (20 per molecule, pure 2D SMILES input): `ETKDGv3(seed=42)` (with
  `useRandomCoords=True` fallback) -> `MMFF94s` (max 200 iterations, energy in
  kcal/mol) -> `xtb --gfn 2 --chrg C --uhf U --json --norestart` (timeout 20 s,
  single-threaded per worker). Extract `etkdg_ok`, `mmff_ok`,
  `mmff_energy_kcal`, `xtb_ok`, `xtb_gap_eV`, `xtb_homo_eV`, `xtb_lumo_eV`,
  `xtb_homo_m1_eV`, `xtb_lumo_p1_eV`, `xtb_fermi_eV`, `xtb_total_energy_Eh`,
  `xtb_elec_energy_Eh`, `xtb_dipole_norm_au`, `xtb_q_min`, `xtb_q_max`,
  `xtb_q_std`, `xtb_q_abs_mean`, `xtb_n_orb`, `xtb_n_elec`, `xtb_n_unpaired`.
  Failed molecules stay in with `xtb_ok = 0` and `NaN` xTB values; never drop.
- **Stage 1 — 2K pilot gate:** 2,000 rows drawn deterministically
  (`default_rng(20261123)`) from `[500000, 550000)`. Measure exact failure rate
  (`xtb_ok == 0`) and projected 50K CPU-hours. **Hard kill:** if failure rate
  `>= 1.0%` OR projected 50K CPU-hours `> 50`, stop and close C2 as FAIL
  (`NEGATIVE_UNDER_CONTRACT`) without running Stage 2.
- **Stage 2 (only if pilot passes):** compute the 20 features for the remaining
  48,000 rows of `[500000, 550000)` and all 50,000 rows of `[3200000, 3250000)`.
  Reuse P0's pinned `features.npz`, `replay.npz`, `s43-dev.npz`, `s44-dev.npz`,
  and `block-transfer`'s pinned `features.npz` and `predict.npz`. Fit using
  P0's unchanged `make_gbm()` on P0's 30,155 dev-fit rows for `s43` and `s44`:
  - `p0_2d` (reproduced and verified `<= 0.0001 eV` against P0 saved outputs);
  - `p0_2d_plus_xtb` (`P0_2d` + 20 xTB features);
  - `xtb_only` (`ŷ` + 20 xTB features only).
  Seal all feature and prediction arrays with row IDs and SHA256 before reading
  score labels.

## Endpoints and verdict

Paired row bootstrap (2,000 draws, seed 20261122, models fixed).

- **PASS:** pilot passes (`fail_rate < 1%`, projected 50K CPU-h `<= 50`) AND on
  P0's 19,845 scaffold-disjoint score rows BOTH `s43` and `s44` have
  `MAE(p0_2d_plus_xtb) - MAE(p0_2d) <= -1.5 meV` with 95% upper bound `< 0`.
- **FAIL:** pilot fails, OR either init has 95% lower bound `> -0.75 meV`.
- Otherwise **INCONCLUSIVE**.

Secondary descriptive checks (cannot override primary verdict): `xtb_only` and
`p0_2d_plus_xtb` on `far` `[3200000, 3250000)` (overall and on unseen fit
scaffolds), `xtb_only` on P0's score rows, raw correlation of `xtb_gap_eV` with
DFT Gap and residuals, and performance on the N4 all-single-bond subgroup.

## Custody and cost

Publish the prospective trajectory before any data read. Expected local CPU
cost: ~0.5 min wall for 2K pilot; ~18–25 min wall (~4–6 CPU-h) total for both
50K blocks across 12 workers if pilot passes. No GPU training, no protected
role, no retry on a scientific miss. Row outputs stay outside Git in
`D:/molgap-artifacts/diagnostics/pcqm-c2-xtb-diagnostic/`.

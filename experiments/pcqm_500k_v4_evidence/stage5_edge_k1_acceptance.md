# Stage 5 EdgeState/K1 terminal acceptance

Kaggle kernel `nothingnessvoid/molgap-500k-v4-edge-k1-s42` version 6 reached a
terminal state on 2026-09-16. Frozen no-inference acceptance was rerun locally
for both arms under contract fingerprint
`ddbfdd8d9f88a7315efad1fd446f96da61052488a823848c1302baed6d3efd4a`.

## Neural-Atom K1

- Acceptance: complete, `next_epoch=60`.
- Best internal-development Gap MAE: `0.1048598662018776 eV` at epoch 48.
- Final epoch development/training MAE: `0.10592232644557953 / 0.07680446459041357 eV`.
- Completed exposure: 234360 steps and 29998080 presentations.
- Stage manifest: `6de0d6a576729030138a5f0a6914f7afe9cac75860b4355fe49c1e3c5e09f7e1`.
- Last checkpoint: `452d6fa0bfaffb1392e8b1092d32d18375a10f1888ecf4374bc7d1a36048f8b8`.
- Best model: `e022f8f982191bed99c75694a24994af134987165fa5ba519dd35974822d6535`.
- Best predictions: `68fba785a0b028a8445fe8d94d348df2c3a8d7d8caab708acb574d13cac9831b`.

## EdgeState GPS9

- Acceptance: valid resumable partial, `next_epoch=59`; not a scientific failure.
- Best internal-development Gap MAE so far: `0.11134880781173706 eV` at epoch 54.
- Epoch 58 development/training MAE: `0.11135336011648178 / 0.10311244582042647 eV`.
- Completed exposure: 230454 steps and 29498112 presentations.
- Stage manifest: `30d89f33437cd4047f30741429d362ac032f98fb91b5e881563f3446fa64f825`.
- Last checkpoint: `2adcb8f1777194c38b9cf609c04c86ce0abf2869ea1c6112a48786ef390952b5`.
- Best model: `ae95f4b3cca5b0ea1fe019ee6294690401ee27c021f935ae1d10e8d0fb71e8e1`.
- Best predictions: `c31fafb651fcfef08381627ac86f6639cae91f664ae7a39574d4ca0383b91ce0`.

Both arms retained false official-validation/test-dev/test-challenge read flags.
The incomplete EdgeState arm was released only for a one-epoch unchanged resume;
K1 was not resubmitted. Final paired bootstrap and nomination remain blocked on
accepted EdgeState epoch 60.

Raw evidence:
`platforms/_records/kaggle/training/pcqm_500k_v4_stage5/edge-k1-v6/`.

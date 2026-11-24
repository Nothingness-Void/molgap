# Status

CLOSED, 2026-11-24 JST: gpspp_mpnn NEGATIVE_UNDER_CONTRACT (see `audit_closure/decision.md` and `audit_closure/correction_note_20261124.md`). The text below is the pre-submission status, kept as written.
At preparation: PREPARED_FOR_PLATFORM, not submitted, waiting for the Opus pre-submit review. Run `pcqm-gpspp-mpnn-100k-20261124-s42` (`v1`),
kaggle1 (`nothingnessvoid`), T4x2 one arm per card. Gates and bands: `protocol.md`.

## Local checks (RTX 5060, shared with other desktop programs, deterministic FP32, workers 0)

Timing is noisy: the same code ran 0.16-0.37 s/step across repeats. Medians of per-step
times are used below. Data: real fixed-100K train rows only, no development labels.

| Check | Result |
|---|---|
| Catalog | `gpspp_mpnn/1` runnable=true, registered for `gptrans_t/2` |
| Parameters, control `pair_update_norm` | 5,871,201 |
| Parameters, `gpspp_mpnn` (`d_n` 192) | 5,819,425 (-0.88%) |
| Step time, candidate / control (3 pairs, 220 steps each) | 0.486, 0.484, 0.485; limit 1.25 |
| RWSE16 overhead (in-process A/B, 12 repeats, real vs zero stub) | mean +4.3%, median +3.5%, min-based +4.7%; earlier runs on a busier machine gave -0.7% and +7.2%. Kept: the quiet-run values are under the 5% limit. The layout-based version (no host sync) replaced the first one. |
| 200-step train stability (constant lr 1e-3, batch 128, clip 1.0) | train L1 (normalized) mean of first 20 steps 0.733, last 20 steps 0.364; no NaN. Control: 0.833 to 0.357. Not a tuning run. |

Peak GPU memory: candidate 681 MB, control 1,023 MB.

Deviation recorded: the plan expected a control near 5.25-5.28M parameters; the built
control has 5,871,201, so `d_n` is 192, not 176-184 (see `protocol.md`).

## Preparation checks (no credentials, no `--execute`)

- `submit_workflow.py --experiment ... --repo-root .`: `PREPARED_FOR_PLATFORM`, source commit
  `32260744c`, package identity `062bdc3adb5f4ab8ec5c8e6f674c82f4d78dc03d8c4ff714e04880e48dcd5f45`,
  spec identity `a60577e9237762dfe95d121e567955f3dbf32b57a29b857c5b577520bfd6784c`,
  240 s local call. Kernel `nothingnessvoid/molgap-gpspp-mpnn-100k-20261124-s42`, 14,400 s wall cap.
- Prospective trajectories for both arms are published under `runs/.../prospective/` (record mode
  prospective, before any kernel POST). `research_memory rebuild` ran once;
  `research_memory check --frozen` reports "frozen and current".
- `experiment_cli sandbox --arm gpspp_mpnn`: `PASSED` (finite gradients, fresh-load round trip). It
  needed a fix: the sandbox compared every candidate with a baseline and failed all standalone
  models (the closed `cutlass_gt` run fails the same way on the old code). `fast_sandbox.py` now has
  a standalone branch (+29 lines, own commit). The control arm still fails `addon_taps_fire`
  because it has no addon; that is the old behavior and the control is the reference.
- `infra_budget.py`: passes (src_py_files 281/281, src_py_lines 97,477/97,643).

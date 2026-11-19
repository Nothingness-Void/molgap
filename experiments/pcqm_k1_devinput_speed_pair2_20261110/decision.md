# Decision

Date: 2026-11-19 JST. Run `pcqm-k1-devinput-speed-pair2-20261110-s42` (attempt `v1`, Kaggle kaseichou, one kernel, two T4 devices). Outcome: **INFRASTRUCTURE_ONLY**. This is a speed measurement, not scientific evidence.

## Result

Both arms stopped at optimizer step 1562 (two passes), as designed. The stop is a planned pause, not a failure.

| Arm | Device-resident inputs | Wall (s) |
| --- | --- | --- |
| `muon_off` | off | 660.39 |
| `muon_on` | on | 691.41 |

- Epoch-1 and epoch-2 live train and dev metrics match to every printed digit across the arms (epoch 2: train 0.18898, dev 0.22529). The device-resident path computes the same thing. The stop rule "loss mismatch means invalid" did not fire.
- Device-resident inputs were **4.7% slower** (+31.0 s) on this T4 pair. One run, one pair, so the size of the gap carries no error bar.
- No crash and no CUDA error in either arm.

## Decision

Do not change the default. Device-resident inputs give no speed gain on T4. No longer run follows from this.

## Attribution

- Mode: `infra_failure` is not the right label. There is no failure, so the closure fact `failure_mode` stays `null` (speed work).
- Missing discriminator: why the device-resident path is slower (extra copies, memory pressure, or the two devices sharing the host). This run did not time those parts, so the cause stays `insufficient_evidence`.

## Closure status

Formal closure (`draft-closure`, `accept_run.py --closure-input`) is not done. Both fail with `FileNotFoundError` for `runs/<run>/v1/operations/retrieved_outputs/muon_off.json`. The pause files in `D:/molgap-artifacts/retrieved/<run>/v1/version1` have no retrieval map, and the current tools do not make one for a paused run. No hand-written map or closure file was made. The prospective trajectories stay open.

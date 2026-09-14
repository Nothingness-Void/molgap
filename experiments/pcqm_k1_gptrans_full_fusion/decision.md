# Desktop integration (2026-09-14)

The desktop integrated the full-training, acceptance and official-validation
implementation through fb05260. GPTrans training passed mechanical acceptance
in `results/gptrans_full_acceptance_r3.json`. Official-valid scoring had been
submitted under `results/gptrans_official_eval_protocol.md`; no evaluation MAE
was available in that integration snapshot. K1 recovery and fusion remained
subject to the frozen `protocol.md` and their independent acceptance.

The merge moved the unchanged desktop categorical GPS classes to
`../../src/molgap/categorical_gps.py` and updated their consumers. This preserved
the byte-level frozen K1 GPS source and the existing categorical behavior.
Source packaging moved beside the full-run contract; recovery template
directories use revision names while remote artifact paths remain unchanged.
It did not promote a model or authorize another training run.
Live routing is owned by `../../CURRENT_STATE.md`.

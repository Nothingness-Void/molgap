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

## Final outcome (2026-09-15)

The R3 K1 continuation, mechanical acceptance, fusion study, and fusion
acceptance all completed. K1 scored 0.106672 eV and GPTrans-T scored 0.109012
eV on the 73,545-row official-validation role. Their fixed 50:50 blend improved
to 0.102227 eV, while the pre-registered 55.2:44.8 blend scored 0.102186 eV on
the frozen four-fifths holdout. This proves error complementarity but does not
beat the 0.099638 eV EdgeState reference. The fusion is mechanically accepted
and scientifically closed without promotion. Full evidence and exact deltas
are in `results/accepted_k1_gptrans_fusion_r3/decision.md`.

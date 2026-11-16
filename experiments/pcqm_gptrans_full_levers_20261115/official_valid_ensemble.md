# Official validation, final candidate (2026-11-16)

Read once, report only; not used to change any choice. Candidate: equal-weight ensemble of the four B1024 GPT-T minus-tail inits s42-s45 (`ema_model.pt`), treated as one model (decision_A.md). Script `platforms/local/pcqm_gptrans_ensemble_valid.py`; the gate passed first (4,000 train rows from shards 0/22/44/64: zero field mismatches, stored vs rebuilt prediction diff 0.0).

Official valid MAE (73,545 rows): **0.082812 eV (82.81 meV)**. For reference, K1 single model final-live 0.09934 eV (different model, not a selection signal). Tail 83.992 meV and official valid 82.81 meV are different sets and are not compared.

Evidence: `D:/molgap-artifacts/official_valid_gptrans_ensemble_20261116/` (`gate_evidence.json`, `official_valid_evidence.json`, `authority_note.md`). A first attempt was killed mid-run at the user's request before any evidence file or number existed; the script refuses a second read once the evidence file exists. Official test, test-challenge and sealed 20K: not read.

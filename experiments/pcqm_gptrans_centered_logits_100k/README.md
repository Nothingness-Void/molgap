# GPTrans centered-logits 100K screen

Desktop-owned, single-variable GPTrans-T screen on the accepted 100K/50K
PCQM4Mv2 internal roles. The frozen reference is the accepted seed-42 V4
GPTrans-T 100K result; it is not retrained.

`protocol.md` owns the question and decision gate. `launch_decision.md` owns
the dated one-attempt release; `submission_decision.md` points to its machine
declarations. `training_contract.json` owns the fixed
execution identity. `experiment_spec.json` and `prospective/`
bind the shared local experiment core and RML plan. Kaggle packaging and the
single-arm entry script remain thin adapters around the existing GPTrans V4
trainer.

The Kaggle3 version-1 push and observed platform state are recorded under
`launch/`. Training, terminal acceptance, and the scientific decision remain
pending until durable outputs can be reconciled.

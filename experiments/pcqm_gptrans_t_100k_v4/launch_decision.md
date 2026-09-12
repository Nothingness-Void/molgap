# GPTrans-T 100K V4 launch decision

Decision date: 2026-09-12

The user authorized one GPTrans-T baseline on the accepted fixed PCQM4Mv2
100K/50K role. The machine-readable scientific contract was frozen in
[`training_contract.json`](training_contract.json) before remote submission.

The authorization covered one deterministic runtime preflight and, only after
that preflight passed, one seed-42 baseline training run. It did not authorize
a candidate, another seed, official validation, test-dev, test-challenge, full
training, or a production-registry change.

The result could become a reusable V4 reference only after the frozen
mechanical acceptance recomputed the development MAE, verified every artifact
hash, and accepted its runtime certificate. A successful scheduler state or
completed epoch count alone was not scientific acceptance.

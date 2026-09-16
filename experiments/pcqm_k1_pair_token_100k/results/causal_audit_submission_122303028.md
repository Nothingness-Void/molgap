# PairToken causal-audit submission

- Job: `122303028`
- Resource: `kshdtest`, one Hygon DCU
- Wall-time ceiling: 45 minutes
- Source commit: `54a28528bae22ab43383fec483f99f81936acfac`
- Source archive SHA-256:
  `71ab364ff5f30046c12cb765deeb1d5edcd54d296066eea82b63d4f9d100db46`
- Training: prohibited
- Input checkpoint: accepted PairToken best model from job `122275244`
- Role: fixed 50,000-row train-derived development set only

The job evaluates the frozen interventions in `causal_audit_protocol.md` and
cannot promote or submit a successor by itself.


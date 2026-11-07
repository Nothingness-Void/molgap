# Preparation status

2026-11-07 JST: scientific documents, unchanged current-read addon and registered
batched-Muon input/recipe are committed. No input freeze, prospective publication,
diagnostic, training or remote submission has occurred for this question.

Owner: `codex/exp/k1-read-batched-muon-500k-20261107`, based on verified clean
desktop `eca56da4ca5e7a26040714252219fc11318a919f`.
Scientific source commit `6af974fc`; reviewed infrastructure `dd9dbb42` is
cherry-picked as `a99bccd3`. Kaggle1 account `nothingnessvoid`, accepted own-account
500K dataset `nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1`; exact submitted
job/source version remain unassigned.

`experiment.json` is the real registered input; `control_recipe.json` was generated
with `build_family_recipe` using the registered batched optimizer. Its metadata-only
data projection binds the consumed recipe bytes and preserves all dataset identities.
Both arm models are complete family3 declarations, reference/candidate training
fields are empty, and the explicit target manifest is the500K one.

Main owner must review and use the existing single preparation/submission entry.
Prospective trajectories and native batched-execution qualification are pending;
old unbatched Muon proof does not qualify this execution. No remote action was
performed by the preparation subtask.

The original AdamW/Muon500K owner and Kaggle2 job143011302/version2 are
independent and must not be modified or used as a strict comparator here.

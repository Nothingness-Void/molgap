# K1 slot96 submission handoff

Owner: `codex/exp/k1-slot-width96`, checkout `D:/w/k1-slot96`.
Source: `13b8e23638461eff4d6313030f6af8f238cb9d2f`; evidence commits are separate.

## Last authenticated observation

2026-10-01 17:25:18 UTC: Kaggle3
[`nvoid912/molgap-k1-slot96-100k-s42-v1`](https://www.kaggle.com/code/nvoid912/molgap-k1-slot96-100k-s42-v1)
was **RUNNING**, kernel ID136686300, version1, explicit NvidiaTeslaT4.
Remote entry bytes exactly matched the submitted entry. Both private mounts
matched the frozen launch; all nine downloaded source files matched local bytes
and directory placement. The API had no startup log/output yet: actual T4
preflight completion, formal training progress and science remain pending.
Do not equate RUNNING with passed qualification or an accepted model.

Exact hashes, authenticated observation and existing immutable launch receipt
are linked from [offline_handoff.json](offline_handoff.json).
The source create response omitted a dataset version and script-version ID
was absent from the kernel push response; these remain unknown rather than guessed.
The reported platform lastRunTime differs from the local observation date;
preserve it as returned, and use kernel ID/version/source hashes for reconciliation.

## Frozen question and evidence

Single candidate: atom192 / persistent EdgeState64 / latent slot96 / nine layers /
one active slot / original mean readout and RWSE16. Measured3853793 parameters,
5.329% above3658817 reference. Only latent64->96 is the scientific intervention.
V4 fixed100K/50K, seed42, FP32/noTF32, BS128/drop-last,40epochs,31240steps,
3998720presentations. Reuse the retained original192 reference; no baseline training.
Read [protocol](protocol.md) and [training prospective](kaggle3_v1/slot96/trajectory.json).

CPU usefulness/initialization qualification is closed separately at
[cpu_decision.md](cpu_decision.md), with attribution and canonical RML overlay.
Final-slot removal increased subset MAE142.315meV; this establishes usefulness
on consumed development rows, not that64 dimensions are saturated. The actual
packaged source also passed a finite64-row pure2D forward with strict initial state.
RML's dated reference alias registration preserves original records and explains
the planning label repair. Local preparation first stopped on an RML reference
identifier mismatch before any training prospective/GPU submission; its corrected
preparation retained identical executable source/recipe. See
[preparation reconciliation](preparation_reconciliation.json).

## Reconcile after desktop returns

1. Read the Kaggle skill, this owner and exact version1 receipt; query authoritative
   state for this same kernel. Never submit a successor from stale local state.
2. Retrieve only small `experiment/pair_state.json`, `experiment/slot96/preflight.log`
   and `experiment/slot96/output_manifest.json` when available. On failure retain
   its output/logs and distinguish infrastructure from model evidence.
3. Retrieve manifest-bound required artifacts through the existing family adapter.
   Verify source/config/initial/cache identity, atomic checkpoint exposure, finite
   aligned50K predictions, runtime provenance, actual native cost and roles.
4. Use the frozen family acceptance plan and shared acceptance workflow; report
   paired gain/row uncertainty,3meV gate, strict runtime qualification and single-seed
   limits separately. Baseline NumPy2.0.2 versus standard bootstrap NumPy<2 means
   software equivalence must be assessed, never presumed. Then write attribution,
   terminal RML and explicit adoption/archive routing.

No automatic retry, scale-up, heartbeat, server fallback or protected-role access.
The pre-existing unrelated GPTrans run remains outside this question. The user
authorized local shutdown after verified submission; remote execution is independent.
Large raw reference weights and CPU paired rows remain locally retained/ignored;
the published private source dataset retains the exact initial state and source.

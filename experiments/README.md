# Experiments

One directory per **question**, never per calendar phase. A question that does
not enter the model registry or change the recommended predictor belongs here,
not in `production/`. Each directory owns its decision record and machine
evidence; read that decision before opening metrics or remote logs.

Live operation is routed by [`CURRENT_STATE.md`](../CURRENT_STATE.md) and the
single-group [active queue](../ROADMAP.md#active-queue). Track ownership is
defined only in [`TRACKS.md`](../TRACKS.md): general-model work is Track A,
PCQM leaderboard specialists are Track B, and architecture discovery is Track C.

Use the [directory index](DIRECTORY_INDEX.md) for stable directory traceability
and the [evidence index](EVIDENCE_INDEX.md) for question/verdict history. These
conditional indexes keep this entry protocol stable as experiments accumulate.

For new experiment plumbing, start with the
[new-agent quickstart](../docs/operations/EXPERIMENT_QUICKSTART.md), then the
[addon guide](../docs/operations/EXPERIMENT_ADDON_GUIDE.md) and [local CLI](../docs/operations/EXPERIMENT_CLI.md).
Keep the scientific question, trainer integration, and evidence beside the
owning experiment. Do not copy shared packaging, planning, recovery, or terminal
logic. The local RML reconciliation routed by the tracked
[`research_memory/README.md`](../research_memory/README.md) preserves ended attempts
while removing stale ACTIVE outcomes through receipts.

## Evidence contract

Every active experiment must satisfy this chain:

1. Its directory appears in [DIRECTORY_INDEX.md](DIRECTORY_INDEX.md) and its
   question/verdict appears in [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) when an
   indexed verdict exists.
2. Its `README.md` points to at least one dated decision owned by that directory.
3. The decision points to compact machine evidence such as `metrics.json`,
   `summary.json`, `acceptance.json`, or `manifest.json`.
4. Remote submission and retrieval provenance stays under `platforms/` or the
   experiment's `STATUS.md` and a dated remote log under its `results/` tree.
5. Live state links to the decision and never copies its metrics.

`tests/test_repository_layout.py` enforces directory coverage, decision
reachability, machine-evidence presence, and bounded live control documents.

## File roles inside an experiment

| File | Holds |
|---|---|
| `decision.md` | What the experiment concluded; the entry point. |
| `STATUS.md` | Live operational state of remote jobs, when any. |
| A dated remote log under `results/` | Finished remote rounds, so live documents stay short. |
| `results/*.json` | Exact metrics behind the decision. |

`CURRENT_STATE.md` owns live status and the recommended model; `ROADMAP.md` owns
task order and gates; `platforms/` owns compute-environment adapters; the
conditional indexes own cross-experiment navigation.

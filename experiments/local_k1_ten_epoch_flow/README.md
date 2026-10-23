# Single-model local lifecycle exercise

RML review: the complete K1 audit and slot-readout diagnostic do not establish a harmful aggregation path or a general accuracy ceiling. Existing normalization and DropPath experiments are separate mechanisms. No channel-wise, identity-initialized post-aggregation gain was found in the trajectory index. This bounded exercise tests registration, actual training, checkpoint restoration and closure only; it cannot establish channel-gain benefit.

Owners: [K1 audit](../pcqm_k1_complete_local_audit/analysis_zh.md), [slot diagnostic](../pcqm_k1_slot_readout_diagnostic/terminal_decision.md). The sole model adds 192 gains at layer 6, initialized to one. Cheapest failure signals are absent optimizer ownership, nonfinite updates, failed exact native resume, wrong final exposure or incomplete terminal artifacts.

Data: fixed PCQM official-train source rows 0:2048 and internal-development rows 100000:100512; manifest and shards remain in D:/molgap-data/local_flow_2048_v1. Frozen original initialization remains in D:/molgap-artifacts. No Kaggle training.

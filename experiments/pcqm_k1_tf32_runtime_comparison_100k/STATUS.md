# K1 A100 precision diagnostic

IMS CPU-only runtime preflight `1573042.ccpbs1` completed with `runtime_ok`.
The sole paired A100 job `1573113.ccpbs1` produced two complete 40-epoch arms;
its independently accepted terminal result and decision are in `decision.md`
and `results/terminal_summary.json`. Both arms used the fixed PCQM 100K/50K
export and untouched protected roles. The TF32 arm was ~1.05% slower during
training despite a lower single-seed development MAE. The execution hypothesis
is closed; strict FP32 remains in force, and no full run or successor follows.

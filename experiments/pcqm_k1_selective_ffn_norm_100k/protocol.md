# Local K1 selective mechanisms100K

Three independent single-arm jobs run sequentially on the same RTX5060.
The original BN reference has no MsgNorm and no addon. Candidate1 replaces only
the local-block post-FFN normalization using `k1_ffn_node_rms`, config `{}`.
Candidate2 applies `k1_local_residual_gain`, config `{}`, with normalization
unchanged. These are separate hypotheses, never a stacked candidate.

The owning input pins the historical original-BN Spec/recipe, real local
manifest and original seed42 initialization. Every new recipe uses Torch2.7.1
(local environment2.7.1+cu128, PyG2.7.0), not historical Torch2.11.0.
Freeze100K train rows0:100000 and internal-development rows100000:150000,
40 full passes, BS128, drop_last,31240 updates,3998720 presentations,
LR4e-4, WD1e-5, cosine to1e-6 without warmup, clip1, normalized-Gap L1,
native AdamW, FP32, noTF32, original sampler and original target statistics.
Track EMA0.999 but select best-development Live, preserving the baseline rule.
Shared epoch logging and resumable checkpoints remain owned by the family trainer.
Use a4-core platform budget; existing ExecutionTuning owns cpu_threads=1 and
training_workers=1. Do not pin OMP/MKL threads. No competing training,
installations or official valid/test.

Hypothesis1: removing batch dependence only after the FFN may improve internal
development generalization without changing local-message BN or edge/mixer LN.
Hypothesis2: a learned local residual gain may improve the same baseline without
changing normalization. Alternatives include no gain, worse optimization, and
runtime-dependent behavior. The cheapest authorized falsifier is each fixed40-pass
candidate against the newly trained40-pass BN arm, not historical remote scores.

Prior evidence: `../pcqm_k1_prenorm_norm_100k/decision.md` rejects the specific
Pre-Norm AdaRMS replacement; `../pcqm_structural_bias_norm_100k/terminal_custody.md`
retains negative whole-block Post-LN evidence. These do not answer FFN-only RMS
or learned local residual gain. RML is navigation; these canonical records own
the conclusions. Consult `literature_review.md` if present; this script never
rewrites it.

Comparison remains PENDING across distinct local jobs. Both candidates share the
new BN reference context, not an accepted strict reference bundle. Do not declare
same-run replay, strict cross-job comparability, published-model reproduction,
native speed superiority, adoption or scale-up. Report selected-Live MAE and
candidate-minus-BN paired row errors only after identity/artifact acceptance;
missing runtime/native-cost/trace qualification remains explicit. A gain of at
least3meV with paired-row95% interval below zero is a diagnostic signal only,
not promotion or training-seed uncertainty. No early stop or additional grid;
all three recipes retain40 passes. Results may reject a mechanism or motivate a reviewed
next question; positive scores alone do not authorize another run.

`prepare_inputs.py` only builds declarations and optionally calls the existing
`prepare_family_training`; it never calls a model or trainer. Its RML proposal
uses the question-specific research-action policy in `policy_registration.json`.
The parent must register its exact bytes at
`research_memory/policies/pcqm-k1-selective-mechanisms-local-100k.1.json`
before prospective publication; this delegated agent writes only this experiment.
Canonical prospective publication must precede graph/label loading and training.
Formal packaging requires reviewed committed inputs/source. Source-freeze commits
are normal authorized parent operations; this delegated agent does not commit.
If publication predates a
source change, retain it and reconcile; do not overwrite its trajectory.

From the owning worktree:

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' experiments/pcqm_k1_selective_ffn_norm_100k/prepare_inputs.py
```

Freeze source, policy, recipes and protocol first. Generated `plan_input.json`,
Spec and workflow are execution envelopes: regenerate them against that frozen
HEAD without committing them into a later HEAD, otherwise their explicit source
commit becomes stale. `--prepare` reads these envelopes without rewriting them.
After the parent freezes the reviewed source and inputs, add
`--prepare bn_reference` (then separately `ffn_rms` or `local_residual_gain`).
This packages only that arm and publishes its prospective through the existing
owner. It does not train. A successful report supplies the prepared directory
for `platforms/local/train.py --prepared ... --repo-root . --data-root
D:/molgap-data --output ...`; execution is deliberately not invoked here.

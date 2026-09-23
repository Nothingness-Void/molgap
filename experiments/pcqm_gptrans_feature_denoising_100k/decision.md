# Terminal decision: expanded feature denoising at PCQM-100K

The Kaggle1 T4x2 run `nothingnessvoid/molgap-gptrans-feature-denoising-s42`
version 2 completed both frozen arms at 60 epochs and 46,860 optimizer steps.
The mechanical audit record and paired arrays are under
`platforms/_records/kaggle/training/pcqm_gptrans_feature_denoising_s42_v2/`.
Both arms used the fixed 100K training rows and the aligned 50K internal
development rows. Official validation, test-dev, and test-challenge were not
read. This is a development result, not an official benchmark score.

| Arm | Development Gap MAE (eV) | Delta vs 0.14724505 eV reference (eV) | Paired 95% interval (eV) |
| --- | ---: | ---: | ---: |
| `full_atom` | 0.15068287 | +0.00343782 | [+0.00254239, +0.00432759] |
| `full_atom_bond` | 0.14938110 | +0.00213605 | [+0.00125778, +0.00302134] |

Bond reconstruction improves on full-atom reconstruction by 0.00130178 eV
(paired 95% interval [-0.00216371, -0.00041557] for bond minus atom), but
neither arm improves on the frozen reference. No 500K successor is justified.

## Contract discrepancy and disposition

The prospective `training_contract.json` records AdamW LR 0.0002, weight decay
0.01, and a 0.000002 schedule endpoint. The executed source imports the V4
runtime constants, and the retained optimizer/trace show LR 0.001, weight
decay 0.05, and endpoint 0.000001. The contract and prospective trajectories
also name the pre-v2 source commit; launch attempt 2 and completion artifacts
bind source commit `fa564238e33c70bd377edf4d64f3182238a3ee49`.
The last checkpoints omit a source-commit field, although the completion
manifests and best-model artifacts bind the v2 source. See
`terminal_discrepancy.json` for the machine-readable reconciliation.

These are not grounds to discard the completed training or alter the frozen
contract after the fact. They do prevent strict acceptance *under that frozen
contract*. Both trajectories should close as `INCONCLUSIVE` for contract
qualification, with the observed negative comparison preserved. Their
canonical traces may be retained in RML, but they must be excluded from strict
comparison replay and from any promotion claim. Do not retrain either arm for
bookkeeping; do not consume protected roles.

The contract/source mismatch existed in the first frozen package. The v2
graph-loader retry changed the source commit without requalifying that frozen
identity. Startup preflight checked graph deserialization and model behavior,
but not contract-to-source optimizer and schedule equality. The required
pre-submission gate is now documented in `platforms/kaggle/README.md`; the
minimal terminal procedure is in `research_memory/LIFECYCLE.md`.
The feature-denoising packager's new fail-closed check covers the frozen
source commit, optimizer, schedule, batch, epoch, and exposure fields. It
does not by itself certify every V5 identity; the remaining checks still
require the existing package and runtime preflights. No successor is released
by this corrective gate.

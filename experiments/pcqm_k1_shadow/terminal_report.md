# PCQM K1 shadow cache v1 terminal evidence

- Kernel: `kaseichou/molgap-pcqm-k1-shadow-cache`, version `1`.
- Final Kaggle2 status: `ERROR`.
- Read-only status/output commands used `C:\Users\Adminn\Documents\molgap\.venv\Scripts\kaggle.exe` with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved evidence root: `platforms/_records/kaggle/training/pcqm_k1_shadow_cache_v1`.
- Retrieved artifact: `molgap-pcqm-k1-shadow-cache.log` only; no cache manifest, shard, failure artifact, or acceptance payload was published.
- The frozen `experiments/pcqm_k1_shadow/accept_cache.py` acceptance was not run because the job is incomplete and no cache root was produced.

## Frozen identity and sealed-role contract

- Launch manifest: `experiments/pcqm_k1_shadow/results/cache_launch.json`.
- Launch manifest SHA-256: `D227B47F0F2240B5F7350A8915045BA99F44EA21C3F95AE7DF03435D5776EC0C`.
- Source commit: `2d768fc628c972a6e02172f01a13d9d6dc7eb9ac`.
- Parent graph-cache aggregate SHA-256: `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Parent geometry-cache aggregate SHA-256: `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- The launch contract declares 10,000 shadow graphs, CPU execution, CSV reads restricted to `idx`/`smiles`, `shadow_labels_read=false`, `official_validation_role_read=false`, and `test_dev_role_read=false`.

## Mechanical diagnosis

The worker failed while constructing an unlabeled graph because the Kaggle runtime did not provide RDKit:

```text
File "/kaggle/input/datasets/kaseichou/molgap-pcqm-k1-shadow-source/src/molgap/pcqm_shadow.py", line 118, in _make_unlabeled_graph
    from ogb.utils.mol import smiles2graph
File "/usr/local/lib/python3.12/dist-packages/ogb/utils/mol.py", line 3, in <module>
    from rdkit import Chem
ModuleNotFoundError: No module named 'rdkit'
```

The cache builder then wrapped that dependency failure as:

```text
RuntimeError: Shadow reserve exhausted
```

This is a remote dependency/environment failure. Per the monitoring contract, it is preserved and handed off without repair or retry in this turn. No shadow Gap labels, official validation/test-dev roles, model execution, audit GPU, or successor task were accessed or submitted by this monitor.

## Retrieved evidence hash

| artifact | SHA-256 |
|---|---|
| `molgap-pcqm-k1-shadow-cache.log` | `6F9F432A3D3E5057A5DD5E7A4A823CD0933A65F39FBCDE925419816EE135C0C` |


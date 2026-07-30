# Repaired-2M Colab 3D

Upload these two files to `MyDrive/MolGap/notebooks/`:

- `molgap_repaired_2m_3d.ipynb`
- the generated `molgap-*.whl`

Required Drive inputs:

- `MyDrive/MolGap/raw_data/phase8_repaired_2m.csv`
- `MyDrive/MolGap/raw_data/phase8_expansion_1m.csv`
- `MyDrive/MolGap/results/pyg_3d_graphs_etkdg_expansion_1m.pt`

Run cells 0-3 to build and validate the graph cache. Full training remains
disabled by default. The bounded-residual decision has passed, so the primary
Track A SchNet may start only after all 100 shards pass acceptance.

The graph builder reuses original-1M coordinates by CID plus canonical SMILES,
updates labels and repaired-2M `source_idx`, and builds only missing rows. It
writes 20K-row graph shards, per-shard SHA256 reports, atomic progress, and a
strict final validation record to Drive.

Training reproduces the GPS split on the complete 2M `source_idx` space before
filtering failed ETKDG rows. This is required for leakage-free embedding
alignment. The active SchNet contract is `176/160/6`, 50 Gaussians, cutoff
10 A, and dropout 0.05; it is intentionally distinct from the legacy
cutoff-6/dropout-0 SchNet contract.

After the primary cache passes acceptance, generate
`molgap_repaired_2m_second_conformer.ipynb` with
`build_secondary_notebook.py`. The second notebook builds an independently
seeded ETKDG view only for accepted primary identities, keeps a separate Drive
directory, and uses the same 100-shard durability contract. It does not start
SchNet training.

## Verified Primary Recovery Bundle

`upload_primary_recovery/` contains the fail-closed replacement bundle:

- `molgap_repaired_2m_primary_stable_recovery.ipynb`
- `molgap-0.1.0-py3-none-any.whl`

Exact file and required-input hashes are in
`primary_recovery_bundle_manifest.json`.
The notebook requires wheel SHA256
`f38a93f558dd5c13e879740e7af7eb5e6135b520a3b220b53ff5863e6ed7cce5`
and writes only to
`MyDrive/MolGap/checkpoints/molgap_repaired_2m_primary_verified_recovery_v2`.
The rejected `molgap_repaired_2m_primary_stable_recovery` directory is not a
resume source. The incident and controls are recorded in
`experiments/repaired_2m_scaling/results/dual_schnet_full_2m/colab_recovery_incident.md`.

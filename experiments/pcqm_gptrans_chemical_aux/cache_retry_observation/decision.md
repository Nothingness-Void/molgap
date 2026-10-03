# Repaired component-cache diagnostic, 2026-10-01

Outcome: NO_TRAIN. Both independent caches passed the existing ChemicalLabelCache acceptance for the exact 100,000-row frozen V4 official-train prefix. The manifests show complete ordered export coverage, complete failure inventories, and no row-level label failures.

The descriptor-only policy was pcqm_topology with mask_nonfinite: 369 missing cells across 92 rows and 5 columns. The affected columns were MaxAbsPartialCharge (92), MaxPartialCharge (92), MinAbsPartialCharge (92), MinPartialCharge (92), and qed (1). Original rows are retained with a validity mask. The fingerprint-only policy was pcqm_topology and has no descriptor-missing cells.

Measured cache-stage elapsed wall durations were 1115.801618 seconds for descriptors and 54.362066 seconds for fingerprints. CPU busy time and aggregate end-to-end task wall time are unknown. No accelerator was allocated and GPU cost is not applicable.

Both caches bind role SHA256 626fb2ff7dc44e9e6a88dfd301b3b295d1414c89b1dbac708af8379c66d4932c, ordered SMILES SHA256 64c10233a76f1f36d9f0da3b3c26a83c4166a44bbf70c3e58fcc6d24cb78df46, and official-train row-manifest SHA256 c329fbde935324118f4f62aa6e2b01b209d6f54642549637853419deb1761c4b. The export verifies official-train membership and records that protected target columns were not read. Only official_train_prefix_0_100000 was used for derived labels; development, validation, test-dev, and challenge roles remain untouched.

The prospective trajectory retains inherited source_config_identity 3ba6a0cf4f1f5b68f12ebfa3ca6c48f16bdbb9b955871bc69a9391284689161e. That opaque digest is not treated as attestation of the retry's effective label policies. This acceptance binds the actual cache manifests, effective component/parser/missingness policies, encoder identities, row/export hashes, cache array hashes, and source-file hashes. The exact process command line was not retained; effective cache parameters are taken from the accepted manifests.

This closes the CPU cache diagnostic only. No model training, inference, comparison, or promotion occurred. The two training arms remain separate prospective questions.

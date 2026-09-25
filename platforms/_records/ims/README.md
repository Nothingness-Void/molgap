# IMS Evidence Records

Read the latest timestamped snapshot directory for immutable remote logs,
metrics, manifests, scripts, inventories, and SHA-256 verification. Experiment
decisions should link to their own compact evidence under `experiments/` and use
these snapshots only for complete remote provenance.

- `experiment_records_20260825_133205JST/` - six IMS MolGap project roots,
  1,272 verified record files.
- `v4_submit_infrastructure/` - accepted reusable V4 launcher deployment,
  source identity, and remote registry-audit evidence.
- `geometry_scratch_audit/` - compact IMS numerical audit manifests; tensor
  shards remain local and are ignored.
- `full_chain_reacceptance_20260926.md` - read-only remote/local recheck of the
  already accepted desktop K1/GPTrans-T full chain.

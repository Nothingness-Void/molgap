# PubChemQC acquisition acceptance

Follow the owning sampling and role contract. Freeze source identity, bucket
definitions, quotas, and the complete prior accepted exclusion inventory before
launch. For concurrent scans, partition by stable source-file shard before
sampling windows; different seeds alone do not make jobs disjoint.

Each bounded output needs a durable manifest with its source shard, exclusions,
row counts, file hashes, worker return codes, elapsed time, and completion
state. Write progress and final manifests atomically so a stopped worker can be
reconciled without guessing what completed.

For acceptance, preserve raw output and verify the declared files and hashes,
required CID/canonical SMILES and HOMO/LUMO/Gap columns, finite labels, and the
contract's Gap relationship and tolerance. Deduplicate by both CID and
canonical SMILES across groups, concurrent shards, and prior accepted output;
apply the frozen group priority and exclusions. Regenerate accepted counts and
hashes from the retained rows.

Treat `complete_partial` as source exhaustion only when the manifest explicitly
supports it and all workers exited successfully. Keep rare-group quota
shortfalls visible rather than filling them with general rows. Accepted
candidate inventory is not automatically a promoted training table or a model
quality result. Publish an accepted private dataset only when a dependent stage
needs it, with its manifest and reconciliation evidence.

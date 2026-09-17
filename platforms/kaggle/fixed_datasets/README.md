# Fixed PCQM4Mv2 Kaggle Mirrors

This adapter prepares a new Kaggle account to host byte-identical private
mirrors of the accepted PCQM4Mv2 100K and 500K graph payloads. It does not
rebuild graphs or change the inherited V4 scientific identities.

Run `prepare_account_mirror.py` with the repository virtual environment. The
adapter verifies the frozen manifest SHA256, every declared graph shard, row
counts, and protected-role flags before writing account-specific Kaggle
metadata. Publish only after this preflight succeeds, then download the remote
manifest, metadata, and file inventory. Run `accept_account_mirror.py` to
verify those remote records against the same frozen identity.

Only official-train-derived 100K and 500K roles are allowed. Official
validation, test-dev, test-challenge, 1M, and full payloads are outside this
adapter.

# IMS repaired-2M secondary ETKDG split

This payload builds repaired-2M secondary conformer shards 40-99 as independent
PBS array jobs. It is isolated under:

`/lustre/home/users/sm2/chou/molgap-phase8-secondary`

The source CSV, accepted-primary sidecars, and primary graph hashes are frozen
into `input/manifest.json`. Each task reads one compressed 20K-row CSV, skips
the primary conformer failures, and atomically writes one framework-neutral NPZ
shard plus one SHA256 report. The accepted NPZ is converted to standard PyG on
Kunshan, so IMS needs only the exact RDKit and NumPy wheels, not a 700 MB Torch
wheel. It does not require or modify the primary graph cache.

Submission order:

1. `jsub -q H setup_env.pbs`
2. After environment acceptance, submit one preflight with
   `jsub -q H -v SHARD_INDEX=40 build_shard.pbs`.
3. Retrieve and validate shard 40 before submitting `jsub -q H -J 40-99
   build_shard.pbs`.

Do not cancel the Kunshan serial build until the school preflight is accepted.

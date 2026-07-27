# Kunshan repaired-2M primary 3D handoff

Upload the three immutable data inputs to:

`$HOME/molgap-inputs/repaired_2m_3d/`

Upload `molgap_repaired_2m_3d_code_patch.zip` to `$HOME/`.

After upload, extract the code patch from `$HOME` so its repository-relative
paths update `$HOME/molgap/`. Verify every SHA256 in `upload_manifest.json`
before submitting:

`$HOME/molgap/scripts/phase8/remote/scnet/build_phase8_repaired_2m_3d_kunshan_cpu.slurm`

The job uses `kshctest02`, 32 CPUs, 110 GB RAM, 28 ETKDG workers, 20K atomic
shards, and no accelerator. Colab stopped with 25 complete Drive shards; these
are not transferred because the first 500K rows are reconstructed from the
accepted original-1M cache without new ETKDG work.

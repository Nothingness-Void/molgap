# Preflight submission failure

On 2026-09-17 the first Kunshan preflight submission was rejected by Slurm
before a job ID was allocated because 32 GB with eight CPUs exceeded the
partition memory-per-CPU limit. No accelerator was reserved, no model code ran,
and no scientific artifact was written. The adapter was corrected to the
previously accepted 24 GB request; the scientific contract is unchanged.


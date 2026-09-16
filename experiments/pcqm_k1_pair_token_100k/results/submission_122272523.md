# Kunshan submission

- Job: `122272523`
- Partition/resource: `kshdtest`, one Hygon DCU
- Maximum wall time: five hours
- Source commit: `396c2872b40e78c14bc588a0fe9985594236d1e1`
- Source archive SHA-256:
  `61022724d7eb28bafb73ab342ae618a8cd753cb75707f2948939bf9ad0f2ade7`
- Fixed cache:
  `/public/home/changfeng2006/molgap-v4-bootstrap/molgap-v4-100k/input/pcqm4mv2-ogb-fixed-100k-v1`
- First observed state: `PENDING (Priority)`
- Other account jobs at submission: none

An initial `sbatch` command requested 32 GB with eight CPUs and was rejected by
the scheduler before creating a job. Commit `396c2872` changed only the Slurm
memory request to the previously accepted 24 GB; the scientific contract and
source implementation were unchanged.


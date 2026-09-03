# SCNet training compatibility probe

## Result

Kunshan card1 can run the current MolGap GraphState stack after selecting the
platform's DTK 23.10 runtime and an isolated Python 3.10 environment. The
successful probe performed three synthetic forward/backward/AdamW steps on one
allocated DCU. It did not read an official dataset and is not a training
quality result.

The earlier failure is retained as evidence of the old module's limitation,
not of the accelerator: DTK 22.04 exposed Python 3.7 without PyG. The new
runtime uses the official DTK 23.10 PyTorch 1.13 wheel and the matching
platform PyG extensions. OpenMPI was required to expose `libmpi.so.40`.

## Evidence

- Scheduler submission probe `120816243`: `COMPLETED`, exit `0:0`, three
  seconds.
- Legacy GraphState probe `120816310`: `FAILED`, exit `1:0`, three minutes and
  two seconds on `kshdtest` node `j15r3n10`; no model step ran.
- Isolated GraphState probe `120822635`: `COMPLETED`, exit `0:0`, three
  minutes and thirty seconds on node `e06r4n11`.
- Runtime: `compiler/dtk/23.10`, `mpi/openmpi/gcc-9.3.0/4.1.5`, Python 3.10.18,
  DTK PyTorch 1.13.1, HIP 5.4.23392, PyG 2.5.3.
- Candidate: `ogb_distance_angle_triangle_edge_state_graph_state9`,
  3,665,809 parameters, three finite losses decreasing from 0.515642 to
  0.464154, peak allocation 58,338,304 bytes.
- Raw probe, Slurm stdout, and final environment freeze are retained in
  `kunshan_job_120822635/`.
- Estimated accelerator use for all three probes: 0.1097 card-hours.

## Disposition

The Kunshan card is now runtime-compatible for bounded MolGap training
experiments. Do not treat this synthetic probe as evidence that a full PCQM
run fits the card. Before any dataset job, stage an accepted graph cache in a
dedicated result directory, run a throughput/memory gate, and keep atomic
checkpoints and independently retrievable outputs.

The Wuzhen/A-zone account remains SSH-reachable but exposes no schedulable
partition under the current account.

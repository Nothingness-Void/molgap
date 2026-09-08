# Local hierarchy screen status

The Track C seed-42 screen was released to SCNet Kunshan from source commit
`e9e7e6a` on 2026-09-08. Mechanical state and terminal metrics belong in this
file and the adjacent launch record; scientific interpretation belongs in a
dated decision after acceptance.

- DCU preflight job `121316216`: completed and accepted; inference model
  `3,665,809` parameters, training-only heads `36,743`, finite forward/backward
- first CPU cache job `121316249`: infrastructure failure before graph build
  because the ROCm environment could not load `libhsakmt.so.1` on a CPU node
- first dependent training job `121316265`: automatically cancelled without
  starting because its cache dependency failed
- online CPU dependency job `121320565`: infrastructure failure because SCNet
  CPU nodes could not resolve the package index
- its cache `121320569` and training `121320574`: dependency-cancelled before
  starting
- offline CPU dependency job `121323044`
- offline CPU cache job `121323061`: failed before graph construction because
  importing shared `qm9_screen.py` eagerly loaded the unused 3D `torch_cluster`
  extension
- its training job `121323070`: dependency-cancelled before starting
- decoupled CPU cache job `121325769`
- current training job `121325771`, dependent on `121325769` and reusing the
  accepted preflight
- remote root:
  `/public/home/scnaqkfcy3/molgap-trackc-qm9-local-hierarchy-e9e7e6a`
- decoupled-chain initial state: cache pending; training pending on its
  `afterok` dependency
- official PCQM roles and QM9 held-out role: not read

An earlier submission attempt from `d4c0548` produced no job IDs because all
three resource requests exceeded Kunshan's `DefMemPerCPU=3569M` ratio. Commit
`e9e7e6a` corrected resource declarations without changing the scientific
contract.
Commit `d834486` then isolated pinned CPU-only chemistry dependencies after the
first cache attempt exposed a CPU/DCU runtime mismatch. It changed only remote
infrastructure; the frozen scientific source identity remains `e9e7e6a`.
Commit `a1c4ec6` removed SCNet network dependence. A locally resolved Linux
wheelhouse with SHA-256
`71f730affc2c3060d64bf87003ea60451b7af8d62edfe32e56613e5517fe557d`
was uploaded and verified before the offline chain was submitted.
Commit `43adfc8` moved shared QM9 acquisition and split primitives into a
model-free module. This removes an accidental 3D-extension import from the CPU
cache path without changing the frozen graphs, model, targets, or training
contract.

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
- decoupled CPU cache job `121325769`: failed before graph construction because
  the offline node had no staged QM9 source and the fallback download could not
  resolve DNS
- its training job `121325771`: dependency-cancelled before starting
- processed-only CPU cache job `121327106`: loaded the staged processed tensor,
  then failed because the raw `gdb9.sdf` archive was not yet staged
- its training job `121327125`: dependency-cancelled before starting
- fully-offline CPU cache job `121327900`: failed while rebuilding source index
  `96635` (`gdb_98534`) because its canonical SMILES did not sanitize
- dependent training job `121327905`: dependency-cancelled without starting
- remote root:
  `/public/home/scnaqkfcy3/molgap-trackc-qm9-local-hierarchy-e9e7e6a`
- no GPU/DCU training result exists
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
The next cache attempt exposed a second independent offline prerequisite: the
processed QM9 source itself had not been staged. The previously used Kaggle
asset `qm9_v3.pt` was uploaded under a temporary name, verified remotely at
SHA-256 `90052e9288b669cc41ecf4899b28ff99e1082e47f2c05eccfb1899572524d721`,
and only then atomically installed. Jobs `121327106 -> 121327125` reuse that
immutable source without changing the scientific contract.
That attempt confirmed the processed tensor was found, then exposed the missing
raw SDF prerequisite. The frozen DeepChem QM9 archive was staged with SHA-256
`f11d3f8ecc3097a72656a35c4847767852e6f346bf99a54619a1a3b6389ddc02`;
jobs `121327900 -> 121327905` therefore have both source assets available fully
offline.
The fully-offline run then exposed a data-contract inconsistency rather than an
environment failure: raw QM9 intentionally contains molecules that RDKit can
load only with sanitization disabled, while this protocol requires every graph
to survive a canonical-SMILES round trip. A train/validation-only structural
probe found 254 such records in the frozen roles. Filtering them before the
split or vectorizing their unsanitized SDF molecules would change the frozen
data contract, so neither repair was submitted automatically.

## V2 release boundary

The v1 GraphState/WedgeState question is closed without a training result. Its
failed jobs, logs, source assets, and protocol remain immutable evidence.

`protocol_edgestate_v2.md` defines the replacement question. V2 freezes and
hashes the complete canonical-roundtrip-valid source pool before a deterministic
30K/3K/3K split, independently accepts the train/validation-only graph cache,
and uses the accepted pure-2D EdgeState GPS9 inference backbone. This is a new
scientific contract, not an infrastructure retry. The v2 cache, preflight, and
training jobs have not yet been released from this status record.

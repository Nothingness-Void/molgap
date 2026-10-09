# Reviewed CPU runtime recovery

2026-10-09. Attempt2 stopped at runtime-manifest capture, before cache/checkpoint
access: the shared recorder attempted CUDA device-property discovery on a CPU
diagnostic. Retain its input JSON and failed reports unchanged.

The worker now sets CUDA_VISIBLE_DEVICES=-1 before Torch/source imports. Runtime
capture additionally requires CUDA unavailable, zero visible CUDA devices and
no accelerator in the manifest. No scientific precision or model change.
A real subprocess regression (no mocked runtime recorder) passed. Parent's
separate import/runtime-only preflight observed accelerator=null,
torch=2.7.1+cu128, cuda_visible_count=0; no model/cache execution.

The parent authorizes one reviewed infrastructure-recovery attempt within the
same already-authorized diagnostic, using inputs_attempt3.json and a new result
directory. Procedure, sample, selected states, roles, gates and600/1200second
ceilings remain fixed. Retain all failed attempt costs; no automatic retries.

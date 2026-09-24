# K1 A100 FP32/TF32 diagnostic — 2026-09-25

The single paired seed-42 job `1573113.ccpbs1` produced two complete 40-epoch
arms. Independent no-inference acceptance verified the immutable 100K/50K
fixed cache, common initial model SHA, all artifact hashes, 50K aligned
development rows and targets, saved-prediction MAEs, full traces, and untouched
official validation/test roles. The job no longer appeared in `jobinfo -c`;
the scheduler exit code was not independently verified. The complete terminal
artifacts, rather than queue absence, establish execution completion.

| Arm | Development Gap MAE (eV) | Best epoch | Training seconds | Graphs/s | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict FP32 | 0.1412893981 | 37 | 3552.5264 | 1125.599 | 596316160 |
| FP32 storage, TF32 matmul | 0.1400451362 | 37 | 3590.1769 | 1113.795 | 653515776 |

The TF32 arm's MAE was lower by `0.0012442619 eV` on this one internal
development split, but its training throughput was **0.989513×** the strict
FP32 arm (~1.05% slower). The A100 GEMM probe did register a nonzero
strict-versus-TF32 arithmetic difference, so TF32 was actually enabled. This
job does not demonstrate the intended runtime benefit. One paired seed does
not establish an accuracy advantage or numerical equivalence, and the arms
have different precision identities; this is a noncausal `PAIRED_ENDPOINT`
execution diagnostic, not a V5 strict-causal architecture comparison or a
screening-replay winner.

Decision: close this execution hypothesis and retain strict FP32. Do not
change the desktop full-run contract, scale, seeds, protected-role access, or
submission workflow. No successor training is authorized by this diagnostic.
The result may motivate a separately scoped profiling question, but the
present evidence cannot attribute the missing speedup to a specific kernel.

Compact authority is `results/terminal_summary.json`; both exact canonical
traces and explicitly replay-ineligible manifests are repository-retained.
Raw models, predictions, checkpoints, role histories, runtime certificates,
comparison, and no-inference acceptance remain in
`platforms/_records/ims/k1_tf32_paired_100k_v1/retrieved/`. The prospective
trajectory records both arms under one action, with separate observed costs and
role events. Per-arm elapsed costs exclude any unmeasured scheduler setup or
queue time. This diagnostic is deliberately not promoted to a strict causal
replay pair.

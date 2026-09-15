# Next-day terminal inspection

The account scheduler returned an empty active queue. Neither continuation
chain produced accepted new metrics.

GPTrans-T R2 (1511027/1511028/1511029) passed preflight, then training raised
RuntimeError in pcqm_gptrans_full_runner.py EMA.update: CUDA model tensors were
combined with CPU EMA tensors after checkpoint restoration. Its train stdout
was empty; no completed new evaluation was accepted.

K1 R1 (1511905/1511906/1511907) failed at preflight import:
ModuleNotFoundError: No module named 'molgap.pcqm_k1_full_runner'. The staged
runtime was incomplete. This failed before encoder training.

Remote evidence roots, both inside the authorized user directory:
- /lustre/home/users/sm2/chou/molgap-k1-gpttrans-full/gptrans_convergence_20260915_r2/logs
- /lustre/home/users/sm2/chou/molgap-k1-gpttrans-full/k1_convergence_20260915_r1/logs

Existing accepted full scores remain unchanged. Neither continuation establishes
convergence. Inspection did not resubmit, tune or read sealed evaluation roles.

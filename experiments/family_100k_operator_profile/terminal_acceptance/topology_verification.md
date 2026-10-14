# Topology verification - 2026-10-14

Independent tester: delegated `gpt-6-luna`, max reasoning. Parent implemented
the shared changes and tests; the tester did not edit code. No real dataset,
protected role, molecular inference or remote platform was accessed.

## CPU Checks

Using desktop `.venv`, owning `src` on PYTHONPATH, all CUDA/HIP/ROCR visibility
variables set to `-1`, a fresh process reported CUDA unavailable. The following
combined suite passed207tests, with1native test explicitly deselected:

```text
tests/test_family_topology_reuse.py
tests/test_k1_accelerated_execution.py
tests/test_family_speed_equivalence.py
tests/test_family_training_kernel.py
tests/test_family_training_session.py
tests/test_family_full_lifecycle.py
tests/test_gptrans_execution_hotpath.py
tests/test_gptrans_execution_tuning.py
tests/test_family_execution_profile.py
```

Command: `python -m pytest -q <above paths> -k "not native_layout_transport"`.
Touched-code/test compilation and `git diff --check` passed. The new18tests
include exact outputs, gradients, optimizer state and RNG against previous-
expression CPU oracles, nonzero local paths, empty edges, distance codes and
unchanged5871201-parameter initialization identity.

The previous BMM-count assertion was updated to the proved simple-path bound;
the original exact distance/RNG assertions remain. K1's test oracle now clears
old gradients between branches rather than accumulating the control's gradients.

## Bounded Native Check And Preserved Anomaly

A first compatibility-suite invocation used empty accelerator visibility
variables, which did not hide CUDA on this Windows runtime. It accidentally
ran the old native transport case and failed with differing loss snapshots
(0.4358 versus0.4244). This invocation is not CPU-only acceptance, and its
failure is not erased or assigned a speculative cause.

Ordinary pytest now requires explicit `MOLGAP_NATIVE_UNIT_TESTS=1` in addition
to CUDA availability for that native case. The case also checks initial model
hash and starting CPU/GPU RNG; it does not weaken endpoint equality.

With the previously authorized idle short-GPU scope, an independent fresh
process ran only that synthetic native case. Immediately before execution,
RTX5060 showed2% utilization and1179/8151MiB allocated. With CUDA visibility0,
HIP/ROCR `-1` and explicit opt-in, the case passed1test in1.76s: initial model,
starting RNG, loss, gradients, optimizer, final model and final RNG matched.

This is blocking/nonblocking K1 transport equivalence, not a full optimized
versus frozen-old T4 training comparison. The first-run anomaly remains
unexplained. No new T4 certificate, native GPTrans qualification, quality
endpoint or measured throughput gain follows. Future packaged jobs retain the
mandatory source-bound native repeat/resume barrier.

# Verification

One consolidated synthetic batch executed with the project Python and owning
worktree PYTHONPATH: `pytest tests/test_k1_component_diagnostic.py
tests/test_k1_weight_average_diagnostic.py tests/test_k1_bn_calibration.py`.
21 passed,0.54s; one upstream PyG deprecation warning. Tests verify parameter-only
averaging, unchanged buffers, incompatible-shape rejection, message attenuation
preserving root/bias, and exception restoration of hooks/BN buffers/modes.
AST parse passed for the three shared workers/closure and six prepare/run/close
wrappers. Static review identified thread/role guards; parent added explicit
four-thread runtime binding and exact observed role membership before closure.
Terminal acceptance exercises those guards through the real result packages.

Two prospective records preceded execution. Both local CPU workers completed;
all source/artifact hashes and alignment/restoration checks are true. Neither
optimizer nor gradients nor training executed. Finalizer receipts and generated
RML checks independently record publication and derived-state validation.

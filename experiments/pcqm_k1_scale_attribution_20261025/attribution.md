# Attempt 001 infrastructure attribution

The local diagnostic adapter imported MolGap before installing the frozen
package path. Package initialization had already loaded `molgap.constants` from
the checkout. The later source guard correctly rejected the mixed source roots.
This is an adapter import-order error, not a failed model or a scientific result.

The correction uses the constants reload discipline already implemented by the
existing frozen BN diagnostic. Frozen source, graph bytes, scientific cases,
thresholds and execution budget remain unchanged in the separately planned retry.

The execution receipt is the retained process observation. The source establishes
that graph decoding precedes the rejected guard, and that prediction follows it.
No original worker-log file was created; any copied execution-tool excerpt is
identified as such. Missing process CPU time is not reconstructed from wall time.

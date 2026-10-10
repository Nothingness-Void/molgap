# Optional global loader RNG

2026-10-11 JST. Infrastructure-only synthetic CPU qualification; no model,
molecular role, training, inference or GPU execution. Frozen payloads unchanged.

`K1LoaderReuse(..., preserve_global_rng=True)` explicitly allows `generator=None`
only for the inspected stock Torch2.7.1 Map iterator, deterministic supported
datasets and CPU tensor factories. The original iterator consumes one global
CPU int64 base-seed draw. A later persistent reset does not; this option supplies
that same draw before reset. Worker RNG remains continued, not fresh/reseeded.
The default still requires a dedicated CPU generator and has no new version gate.

Parent regression:98 passed,7 warnings in59.64s across global-RNG, existing reuse
and packed-reader tests. Real Windows spawn uses two workers, original ordered
batches, a resumed subset and an empty terminal stream. Main CPU RNG states and
subsequent draws match fresh loaders. Additional cases reject invalid indices,
custom iterator/factory modes, legacy/default accelerator devices and unsupported
versions without CUDA initialization. Existing100K `epoch_order` supplies one
small synthetic fixture; its Python shuffle algorithm is not replaced.

The preceding wider run had228 passes,1 skip and one test failure: an empty
fresh-loader control reached its assertion before a Windows worker exited.
The fixture now explicitly joins only its own handles in `finally`. The98-pass
repeat includes the corrected real-spawn test; the earlier failure is not hidden
or counted as a passed wider run. The implementation's existing owned cleanup
was unchanged. An execution subagent stopped at an account usage limit; these
final verification results were observed by the parent.

This option is not installed in the registered100K loop. GPU pinning, full
training/checkpoint resume, quality, native cost and adoption remain separate.
The [packed diagnostic](loader_packed_attempt_001/terminal_decision.md) predates
this option and retains its original dedicated-generator qualification.

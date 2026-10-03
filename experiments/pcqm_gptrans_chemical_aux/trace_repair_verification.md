# Same-exposure marker repair verification

The real second kernel version passed both chemical runtime and synchronized
optimizer-step gates, completed 781 steps per arm, and failed when the owning
trainer appended a checkpoint at the already observed epoch exposure.
Shared Trace now keeps all event coordinates nondecreasing and requires strictly
increasing coordinates for metric observations only. Checkpoint/resume/terminal
markers may use the acknowledged exposure without counting as extra epochs.
The family roundtrip regression covers both null and exact marker coordinates,
and rejects duplicate actual observations even with intervening markers.

One final focused pytest invocation by the GPT-6 Luna max test subagent passed:
36 passed, 0 failed, one deprecation warning, 19.85 seconds. It included the
family roundtrip/count guards and chemical labels/cache regression files.
No local model training or inference was performed. Release input bindings
and actual frozen-shard loader gates were then checked for the new package.
Scientific terminal acceptance and full 60-epoch execution remain pending.

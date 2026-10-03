# Auxiliary-label repair verification, 2026-09-30

## Implementation scope

- Reuse the existing PCQM topology parser for an explicit compatibility policy.
- Prepare descriptor and fingerprint caches independently.
- Retain missing descriptor cells and mask only their auxiliary loss.
- Check cheap export structure and known exceptional rows before bulk labels.
- Atomically retain the first failed manifest and stop by default in the CLI.
- Preserve legacy strict behavior, role membership and the prior NO_TRAIN record.

The policy owner is [label_policy.md](label_policy.md). No full cache rebuild,
model training, remote submission or protected-role evaluation was executed.

## One combined test invocation

A GPT-6 Luna max subagent executed exactly one pytest invocation with the
experiment checkout's `src` on `PYTHONPATH` and the project virtual environment:

```text
D:/文档/molgap/.venv/Scripts/python.exe -m pytest tests/test_chemical_aux_labels.py tests/test_chemical_aux_cache.py tests/test_gptrans_objective.py tests/test_chemical_train_export.py tests/test_gptrans_family_events.py tests/test_chemical_profile.py -q
```

Result: **81 passed, 1 failed, 1 warning**, pytest time **3.05 seconds**.
The failing test was
`test_pcqm_topology_real_hypervalent_si_has_fingerprint_and_masked_qed`.
For source row 51128, `O[Si]123O[Si]3(O1)(O2)O`, the upstream normalizer caught
a QED calculation exception and returned finite `0.0`. The initial repair
incorrectly marked that descriptor valid because it checked only finiteness.

## Correction after that invocation

An instance-local normalized-descriptor hook now uses the same upstream raw
functions and CDFs while mapping computation failures to NaN before validity
masking. It avoids a second descriptor pass and global library mutation.
This correction was reviewed in source but **was not rerun**, following the
user's instruction to test only once. The existing failing real-molecule test
is retained as the regression discriminator.

The recorded result is not an all-passing qualification of the final source.
Full train-cache coverage, pinned per-arm identities, profiling and remote
release remain separate outstanding gates. No RML scientific milestone or
training replay-ready claim is created by this implementation repair.

# Versioned Modern2D integration attempt — 2026-10-17 JST

The human authorized filling the PPGT/ESA integration gaps after the pending
two-arm inventory review. Implementation commit
`91bc746a2989759905799ca9b4267626642d8f21` reused the existing training and data
owners. No remote job, model ranking, protected-role read or Desktop change was
performed. The [API guide](../../docs/operations/MODERN2D_INPUT_INTEGRATION.md)
owns reusable invocation and boundaries; the machine evidence record is
[integration_attempt_report.json](integration_attempt_report.json).

## Scientific implementation changes

PPGT/2 is a new scoped port of author commit
`62cafd3abbb8d72e9a8b3146f947a192cf0aacaf`, with MIT notices and raw source slices
in the test-only reference tree. It does not alter the earlier equation-based
prototype. The port follows the released attention scaling, additive theta,
`phi+1` multiplier, epsilon, learned residual scales, stem norms, RRWP channel
count/normalization, SPE ordering and head. Fixed-weight forward and all-parameter
gradient tests execute the pinned raw author definitions in an isolated harness.
Dependency/layout substitutions and categorical adaptation are explicitly
disclosed; full constructor RNG, upstream environment and benchmark replication
are not certified.

The budget adaptation kept 16 blocks, width256/eight heads and pair128; its
meta-only measured count was 9,536,769. The released base preset was separate at
86,826,529 parameters. The paper-small dense interpretation had 35,235,585;
Table9's 17.6M ambiguity was not hidden with sharing or removed mechanisms.
The fixed-weight tests observed maximum output error 5.960464477539063e-8 and
maximum parameter-gradient error 1.7881393432617188e-7 over 82 parameter tensors.
These are normalized synthetic-output errors, not eV MAE.

ESA/3 retained the v2 permutation-invariant PMA and introduced an explicit
atom-only token for every isolated atom, including wholly unbonded graphs.
Absent endpoint/bond slots are zero vectors, not invented chemical categories;
a token-type embedding distinguishes them from directed real bonds. Padding-aware
batched attention was checked against a graphwise explicit-softmax oracle and
mapped v2 weights. This is a disclosed MolGap adaptation, not unchanged whole
author-model fidelity. No accepted real member was dropped or rescanned.
Its default meta-only parameter count was5,015,233; the two defaults together
had14,552,002 parameters, within the predeclared17,811,268 combined cap.

## Data and lifecycle integration

The accepted-data permit and shared indexed loader support exact100K/500K train
and50K Common-Dev declarations without changing V1 or selecting new members.
Synthetic permits and old CPU profiles kept their original limits. The test
positive accepted-ABI path used generated compact100K/50K tensors, not actual
PCQM assets or a new real-data acceptance. Model-facing derivation ignores y/RWSE.

Both new versions used the actual shared Session at BS128 for four updates,
serialized interruption/resume, exact model/optimizer/scheduler/RNG/cursor state,
explicit indexed evaluation and read-only restored evaluation. Output inspection
and profiling remained mechanical/preview-only. No loop, optimizer, sampler,
checkpoint or platform submitter was rewritten.

The private export helper checked selected decoded identities, immutable chunk
recovery, tampering, inventory exclusion, private scratch and label-free census
on generated fixtures. Real filtering/export and mirror publication were not run.

## Verification and unresolved release

CPU batches passed: indexed/topology105; ESA92; PPGT33; accepted-ABI39; private
export59. Counts overlap regression files and must not be added as distinct tests.
Session/owner checks passed217, with three source-bundle checks initially refused
because shared source was uncommitted. The refusal preserved the package gate;
post-commit package verification is recorded separately in the machine report.
The three refused cases then passed; both new model closures passed exact
committed-byte extraction and isolated CPU128 forward, without vendor/trainer
dependencies in the archive. A later metadata hardening bound feature-source
changes into the new recipe identity; its focused recheck is separate.
No whole-repository all-green or native qualification is claimed.

The14 pre-existing dirty files, including the user's RML test, matched their
pre-task raw hashes. GPTrans source retained raw SHA256
`04edcb6f928617d1142c624d071b7d7accb15c92d2f66e3473ed666ca7e5b2b2`.
Raw reference whitespace was retained for exact hashes; first two leaf reports
were copied byte-identically from C temporary storage to E durable storage.

Real data/mirror/census, feature preparation cost and native BS128 qualification
remain missing. PPGT published-small lineage and adapted ESA whole-family scope
remain explicit limitations. Learning rate, scientific horizon, reference
orientation, symmetric tuning/selection and full-allocation budget still need
a prospective executable comparison. Native dispatch remains fail-closed.
No new seed, 500K training, official/Audit/test evaluation, automatic successor,
STQS decision, 100K veto or historical V4/V5 conclusion change occurred.

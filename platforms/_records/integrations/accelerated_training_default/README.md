# Accelerated Training Default Integration

Date:2026-10-12. User requested one accelerated live training wheel, not opt-in
execution addons. Desktop base:`67da44f2ed593ea50da02ac0a02c74d17305a452`.
This is infrastructure adoption, not a scientific trajectory or job release.

## Scope and Provenance

- Registered K1/GPTrans100K recipe, preflight, train, source-package and CLI path.
- K1 uses the adopted initialized-layout implementation with native AdamW.
- GPTrans automatically binds the reviewed cache/transfer/EMA composition and
  foreach AdamW; generic architecture construction remains available to inference.
- Old execution switches/addons are not live training modes. Historical Spec
  parsing and saved-speed inspection remain read-only compatibility interfaces.
- Old source/config/checkpoints/evidence are not rewritten; historical execution
  requires its immutable source package. New recipes bind execution identity.
- No500K/full independent trainer, production registry, server implementation,
  platform submitter or V5/RML gate is promoted or changed by this integration.

Policy/evidence owners: [K1](../../../../docs/operations/K1_EXECUTION.md),
[GPTrans](../../../../docs/operations/GPTRANS_EXECUTION.md). Native speed/quality
and pending terminal RML limitations remain there; CPU tests do not replace them.

## Local Verification

Project Windows venv, isolated desktop worktree, `PYTHONPATH` bound to its source,
`CUDA_VISIBLE_DEVICES=''`; only synthetic CPU fixtures, no retained molecular
role consumption, accelerator work, remote command or training submission.

Expanded36-file regression:1477 passed,9 skipped,1 deselected (462.90s).
Final entry-guard/navigation rerun:73 passed (6.52s). These batches overlap;
do not sum them. Coverage includes dispatch, recipe identity, package/CLI,
all-arm preflight, synthetic loop/resume/output, cache overflow/order/RNG,
EMA tensor/weight restoration and public model-constructor compatibility.
CPU bootstrap smoke also works with no package context.

The deselected historical A100 profile receipt test requires
`platforms/_records/colab/staging/k1-profile-a100-20261007/payload/selected.pt`,
absent in both primary and isolated checkout before integration. No artifact
was invented or historical hash changed. Host/native-specific tests remain skipped.

`compileall` and `git diff --check` passed. Primary desktop
`research_memory check --frozen` passed before integration. No canonical RML or
derived output was edited; this does not certify whole-repository portability.
Root navigation prose was compacted to its existing limits, without changing
contracts, routing authority, scientific conclusions or test thresholds.

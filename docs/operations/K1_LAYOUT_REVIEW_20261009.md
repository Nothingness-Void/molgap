# K1 integration layout review - 2026-10-09

RML `validate` and `check --frozen --portable` passed at desktop `ade57cfb`.
The whole repository layout suite returned48 passed /5 failed, not full acceptance.
This audit does not weaken evidence validation or rewrite frozen records.

## Remaining maintenance

- Experiment navigation: several older K1 README/decision entries lack executable
  links to terminal decisions/JSON. Clean-fit now has a separate navigation
  decision; its hash-bound README and terminal decision remain unchanged.
- Markdown traversal treats exact-byte input/source snapshots as active documents;
  their owner-relative links do not resolve in the integration checkout.
- Imported consistency500K historical evidence keeps dated directories and
  links to archived helpers. It is custody, not active implementation. Its
  [copy inventory](../../experiments/pcqm_k1_consistency_ablation_500k/terminal_custody/copy_plan.json)
  and [archive routing](../../BRANCHES.md) distinguish the two.
- Older K1 scripts and one frozen constants snapshot use directory-depth roots.
  Any refactor must distinguish executable wrappers from receipt-bound source.
- ROADMAP exceeded120 lines by one; the live text was shortened separately.

Fix active navigation with new pointers; any layout-test archival allowance must
verify exact custody/receipt identities. Do not rename frozen artifacts or relax
SHA checks solely to pass structural conventions. Training/accuracy tests and
RML portability are independent of this remaining layout debt.

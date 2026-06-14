# AGENTS — Reading Protocol

MolGap: ML prediction of HOMO/LUMO/Gap (eV) for organic electronic molecules,
trained on PubChemQC B3LYP/6-31G* data. This file is **how to navigate the repo**,
not a project description. One fact lives in one place — follow the links.

## Read in this order
1. **This file** — protocol + hard constraints (below).
2. **`CURRENT_STATE.md`** — the only source of "what's true now": recommended model,
   conclusions, blocker, next actions. If anything conflicts, this wins.
3. **`ROADMAP.md`** — task priorities / backlog (read the relevant section only).
4. **`docs/phaseN.md`** — background, experiments, conclusions for one phase
   (history & method, not live status). Phase 7 = current best.
5. **`ARCHITECTURE.md`** — code map; tells you which file to edit for a change.
6. The specific code files your task touches.

Do not read all docs to find "the current truth" — it's in `CURRENT_STATE.md`.

## Hard constraints (do not break)
- **Python**: always `.venv\Scripts\python.exe` — system Python lacks torch/pyg.
- **Train-inference consistency**: training and inference MUST use the same conformer
  method (ETKDG). Never mix PM6 training coords with ETKDG inference.
- **Targets**: `homo`/`lumo`/`gap` (eV, B3LYP Kohn-Sham), NOT experimental values.
- **Reuse, don't fork**: reusable logic lives in `src/molgap/` only; `scripts/` are
  thin CLI wrappers. Don't redefine model classes in scripts. See `ARCHITECTURE.md`.
- **Don't re-run completed experiments** — cite `results/phase{N}/`.
- **Test scripts locally before delivering.**

## Conventions
- Docs in English (LLM efficiency). One file answers one question.
- Don't double-write a fact; if it must appear twice, the second is a link.
- Comments explain *why*, not *what*.
- Install: `pip install -e .` (editable, via pyproject.toml).

## Doc map (single sources of truth)
| Question | File |
|----------|------|
| What's true now? | `CURRENT_STATE.md` |
| What to do next? | `ROADMAP.md` |
| How was it done? | `docs/phaseN.md` |
| Where to edit code? | `ARCHITECTURE.md` |
| How to install / basic inference? | `README.md` |

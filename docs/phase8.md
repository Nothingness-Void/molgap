# Phase 8: Chemical-Space Mapping & Molecule Screening

## Goal
The Phase 7 model is fixed and is a faithful B3LYP surrogate **inside its
training distribution**. Phase 8 defines that distribution and turns it into a
screen, so that when we predict commercial molecules (Phase 9) we know which
predictions to trust. Answers: *which commercial molecules look like the
training set?*

## Training-set chemical space (P8.1, done)
`scripts/phase8/characterize_training_set.py` → `results/phase8/training_space.json`.
300k PubChemQC molecules — **general organic molecules, not OLED-specific**.

- **Elements** (fraction containing): C 100%, H ~100%, N 94%, O 90%, S 33%,
  Cl 20%, F 19%. Hard-filtered to {C,H,N,O,S,F,Cl} — **no Br, B, P, Si, Se, I,
  or metals (Ir/Pt)**.
- **MW**: 200–1000 (p1–p99 only 203–709 — large molecules are rare).
- **Labels (eV)**: HOMO −7.1…−4.7, LUMO −2.9…+1.6, **Gap 2.9…7.4, median 4.8**.
- **Topology** (50k sample): aromatic rings 0–5 (median 2), aromatic-atom
  fraction 0–0.84 (median 0.43), heavy atoms 14–51, rotatable bonds 1–16.

### Implications for OLED screening
Element-correct ≠ in-distribution. Two distribution edges matter:
- **Low conjugation**: training median is 2 aromatic rings / 0.43 aromatic
  fraction. Highly conjugated emitters (5–10 rings, >0.8 fraction) sit at the
  high edge → lower confidence.
- **Wide gap**: training Gap median 4.8 eV vs OLED emitters' typical 2–3 eV.
  Narrow-gap materials sit at the low edge, compounding the known B3LYP
  narrow-gap blind spot.

So screening must combine an **element hard-filter** with **continuous OOD
scoring** (topology + fingerprint/embedding distance), not elements alone.

## Plan (see ROADMAP.md for task IDs)
- P8.1 Characterize training space — **done**.
- P8.2 In-distribution screen: element + MW + topology gates.
- P8.3 Fingerprint/embedding nearest-neighbor distance → continuous OOD score.
- P8.4 Curate commercial molecule universe (TCI / Sigma-Aldrich / Ossila).
- P8.5 Apply screen → in-distribution candidate list + per-molecule trust tier.

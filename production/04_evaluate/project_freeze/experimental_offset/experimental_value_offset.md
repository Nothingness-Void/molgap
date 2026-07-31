# Offset Against Experimental Values

**This is a limitation record, not an accuracy result.** The model predicts gas-phase B3LYP/6-31G* Kohn-Sham eigenvalues; the reference is thin-film and solution CV/UPS literature values. Different quantity, different phase.

- Reference: `oled_experimental_v2.csv`, 17 commercial OLED molecules
- Methods: CV, UPS (HOMO), CV, HOMO+optical, UPS (LUMO)
- Conditions: Solution (DCM), Thin Film

## Signed offset, `repaired_2m_dense_2d` minus experiment

| Target | Mean | Median | Std | Range | Systematic share |
|---|---:|---:|---:|---|---:|
| HOMO | +0.547 eV | +0.588 eV | 0.195 eV | +0.09 to +0.86 eV | 74% |
| LUMO | +1.257 eV | +1.304 eV | 0.391 eV | +0.42 to +1.93 eV | 76% |
| GAP | +0.719 eV | +0.724 eV | 0.328 eV | -0.14 to +1.19 eV | 69% |

Every HOMO and LUMO offset is positive: predicted levels sit above the measured ones. LUMO is shifted far more than HOMO, so the predicted gap is systematically too wide (one of 17 molecules is the lone exception, at -0.14 eV). This is the direction the literature reports for Kohn-Sham eigenvalues versus condensed-phase measurements.

The offsets are large but consistent, and the standard deviations are well below the means. That pattern is what makes a solid-state Delta head plausible: a mostly systematic shift is learnable, whereas scattered disagreement would not be. It is not evidence that such a head works — none has been trained.

This is a 17-molecule literature compilation with mixed measurement techniques, not a controlled benchmark. Treat it as an order-of-magnitude statement about a known gap.

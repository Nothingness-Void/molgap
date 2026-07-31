# Public API Smoke Test

`PACKAGE-A` requires a smoke test over representative valid, invalid, and
out-of-domain SMILES. This directory holds that record for both frozen
repaired-2M pure-2D presets.

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\smoke_test_repaired_2m_inference.py
```

The runner fails closed: it raises if an in-domain suite row is dropped, if an
invalid row is kept, if any prediction is non-finite, if a returned index falls
outside the input range, or if the in-domain suite turns out not to be in domain.

## Cases

| Case | Rows | Expected | Observed |
|---|---:|---|---|
| valid | 6 | every row predicted, all in domain | 6 valid, 0 dropped, all in domain |
| invalid | 5 | every row dropped, none imputed | 0 valid, 5 dropped |
| ood | 5 | finite predictions, flagged as out of domain | 5 valid, 4 flagged out of domain |

Both presets behave identically on validity and dropping. The invalid suite
covers a parse error, an unclosed ring, an over-valent carbon, an empty string,
and an unknown element token.

## Applicability is reported, not enforced

The loader has no domain gate, so the record carries the signals a reader needs.
Training was CHONSFCl at MW 200-1000; a row outside that window still returns a
number.

The OOD suite deliberately mixes both failure axes: `Si`, `Se`, and `B` are
unsupported elements, methane is far below the MW window, and a `C50` alkane is
inside the MW window but far outside the trained topology. That last row is
flagged `in_domain: true` by the MW-and-element rule while remaining a genuinely
unsupported prediction — a reminder that these two signals are necessary, not
sufficient. An embedding-distance OOD score is a separate delivery item
(`ROADMAP.md` P10.4).

Never present an out-of-domain value as a calibrated prediction.

## Alignment contract

`predict_smiles_batch_repaired_2m_2d` returns `(valid_idx, preds)` where
`preds[i]` belongs to `smiles_list[valid_idx[i]]`. Rows that fail 2D graph
construction are absent from `valid_idx` rather than filled with a sentinel, so
a caller must join on `valid_idx` instead of assuming positional alignment.

Row-level output including per-row applicability is in
`repaired_2m_smoke_test.json`.

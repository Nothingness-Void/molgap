# Stage 1 Pipeline Archive

Archived at: 20260602_144731

## Completed files

```text
src/utils.py
src/02_clean.py
src/03_features.py
src/04_train_baseline.py
requirements.txt
TO-AI/todo.md
TO-AI/handover.md
```

## Verification command

```bash
python src/02_clean.py && python src/03_features.py && python src/04_train_baseline.py
```

## Verification result

```text
raw rows: 281
clean rows: 281
feature rows: 281
final feature columns: 1634
split: train=223 valid=29 test=29
best validation model: extratrees
ExtraTrees valid avg MAE: 0.3236
ExtraTrees test avg MAE: 0.2663
```

## Important note

The 281-row dataset is only a smoke-test/sample dataset. Metrics from this archive are not final scientific results. The next step is dataset expansion to 10k+ filtered rows.

# Observed capability fields at desktop b05ecb41

- FamilyRecipe has scalar weight_decay and single ema_decay; allowedEMA values are.999/.9999.
- FamilyKernel creates AdamW on model.parameters() and constructs one BoundForeachEMA.
- FamilyModePolicy has no parameter-group configuration; development weight names are live/ema.
- Kernel forward state selectors are live/ema/clean-ema.
- Scientific compiler training overrides accept only learning_rate/weight_decay.

Read-only source inspection; no failing remote job, no new implementation or
synthetic test result. No scientific recipe was substituted. Infrastructure
conversation notified: codex://threads/01a14421-e9d1-73a1-9059-3723d77d0f02.

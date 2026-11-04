# Corrected release lineage

Same frozen science, roles and candidates as the original experiment.
Original job142743304/version1 failed both preflights before training;
see attempt_v1_failure.md. Shared repair87ac7861 addresses target binding.
The same-slug local attempt v2 prepared but did not reach POST: the existing
kernel was rejected before a submission journal was created.
A new explicit r2 physical slug and logical run carry the corrected release;
no old journal is cleared, overwritten or reused for a new physical job.
No baseline is retrained. Same question and experiment branch retain all records.

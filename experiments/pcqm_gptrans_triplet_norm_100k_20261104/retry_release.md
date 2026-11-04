# Corrected release lineage

Same frozen science, roles and candidates as the original experiment.
Original job142743304/version1 failed both preflights before training;
see attempt_v1_failure.md. Shared repair87ac7861 addresses target binding.
The same-slug local attempt v2 prepared but did not reach POST: the existing
kernel was rejected before a submission journal was created.
A new explicit r2 physical slug and logical run carry the corrected release;
no old journal is cleared, overwritten or reused for a new physical job.
No baseline is retrained. Same question and experiment branch retain all records.

R2 SDK validation rejected its52-character source slug before upload.
The infrastructure owner observed both its dataset and kernel absent on Kaggle1.
The shorter r3 physical name is molgap-triplet-norm-100k-s42-r3;
its source slug is36characters (within SDK6..50). No r2 journal is cleared.

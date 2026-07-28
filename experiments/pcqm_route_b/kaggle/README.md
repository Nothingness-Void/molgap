# Track B PCQM 1M Kaggle Encoders

`package_variants.py` creates four self-contained GPU kernel packages. They
consume only accepted immutable graph datasets and a warm-start dataset.

Submission order:

1. `gps9` and `augmented_schnet`
2. `gps11_160` and `primary_schnet`

Submit packaged kernels directly and verify their Kaggle remote state. Kaggle
owns queueing; no local slot poller or delayed trigger is required.

Every kernel selects epochs only on primary-view scaffold-development Gap MAE,
writes atomic best/last checkpoints, and exports float16 embedding parts. The
augmented SchNet still trains on both conformers. Official validation labels
are not evaluated in encoder jobs.

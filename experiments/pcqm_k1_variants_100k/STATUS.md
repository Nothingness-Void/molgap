# Status

The architecture and v4 execution contract were released as one Kaggle2 P100
reference job and one Kaggle2 T4x2 candidate job. Candidate version 1 stopped
before importing model code because Kaggle expanded `src.zip` into a dataset
directory; it performed no training. The source-discovery repair preserves the
scientific contract. No scientific result exists until both repaired outputs
pass `accept.py` without model inference.

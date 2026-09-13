# Status

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-k1-combined-s42` version 3 is
running. Versions 1/2 failed before model loading: the dataset was initially
not mounted, then Kaggle's dataset service expanded `src.zip` instead of
retaining the archive. Version 3 consumes that expanded immutable source tree
directly and installs the established P100-compatible PyTorch wheel first. The
scientific contract is unchanged.

No official PCQM role is authorized.

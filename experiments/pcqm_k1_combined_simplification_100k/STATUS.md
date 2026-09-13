# Status

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-k1-combined-s42` version 2 is
running. Version 1 failed before source loading because the newly created
private source dataset was not yet mounted; it also exposed the stock-PyTorch
P100 compatibility requirement. Version 2 keeps the scientific contract
unchanged, uses the ready source dataset, and installs the established
P100-compatible PyTorch wheel before importing the model.

No official PCQM role is authorized.

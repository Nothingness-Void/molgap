# Status

K1 is frozen as the sole candidate before shadow-label access. The label-sealed
10K shadow-cache implementation passed its static and policy tests without
model execution. Version 1 failed before graph construction because its Kaggle
CPU image lacked RDKit; the terminal evidence is `terminal_report.md`.

The unchanged-contract infrastructure repair pins a NumPy-2-compatible RDKit
wheel and performs an OGB/RDKit schema probe before reserve replacement.
Private source dataset `kaseichou/molgap-pcqm-k1-shadow-source-v2` freezes the
repair commit `aa81281edf15fc1516f036ef9850a39b7ee669d0`. Kernel
`kaseichou/molgap-pcqm-k1-shadow-cache` version 2 was submitted once and
confirmed `RUNNING`. The retry still reads only official-train `idx/smiles`;
no shadow Gap label, official validation, or test-dev role is released.

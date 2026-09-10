# Status

K1 is frozen as the sole candidate before shadow-label access. The label-sealed
10K shadow-cache implementation passed 35 combined static and policy tests
without model execution. Private source dataset
`kaseichou/molgap-pcqm-k1-shadow-source` freezes commit
`2d768fc628c972a6e02172f01a13d9d6dc7eb9ac`. CPU kernel
`kaseichou/molgap-pcqm-k1-shadow-cache` version 1 was submitted once and
confirmed `RUNNING`. It reads only official-train `idx/smiles`; no shadow Gap
label, official validation, or test-dev role is released.

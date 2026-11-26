# Shared-bond ring-state exchange in DeMol-CIN 100K

The [prior ring result](../pcqm_demol_ring_identity_100k_20261125/closure_v1/attribution.md) localized improvement to multi-ring graphs. Both new arms preserve exact ring-specific return and uniform DropPath 0.10. CPU topology extraction computes one directed adjacency entry for each ordered pair of chordless rings that share a real bond. A shared atom alone produces no link. Keep the same projection, degree normalization and active-neighbor mask in both arms. The candidate projects the mean neighboring ring state and the control projects the readout of its own state repeated over the same incoming edges. Existing ring-atom, ring-bond and global paths remain in place.

Use fixed 100K train and 50K internal development only, FP32/no TF32, seed 42, BS128, 60 passes and 46,860 updates. Preserve the previous AdamW, cosine with four-pass warmup, primary EMA .999 and auxiliary EMA .9999. Cap the paired allocation at six wall hours with the standard cleanup reserve.

The full-cohort gate is fixed pass-60 main EMA difference <= -3.0 meV with upper paired 95% bound below zero. Additionally require a negative point difference in the label-free stratum of molecules with at least one shared-bond ring pair. Report that stratum and its complement with row counts. Use the existing scorer on exact internal development rows and 2,000 paired bootstrap draws, matching the native endpoint trace. Retain the selected comparison separately and close under the strict RML contract. No automatic scale transfer.

[CWN](https://proceedings.neurips.cc/paper_files/paper/2021/file/157792e4abb490f99dbd738483e0d2d4-Paper.pdf)

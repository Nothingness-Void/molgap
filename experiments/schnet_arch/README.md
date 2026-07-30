# SchNet Compute-Shape Architecture

**Question:** Which SchNet width and filter configuration preserves useful 3D
signal below the project's compute limit?

**Verdict:** `176/160/6` uses 77.8% of the parameters and 48.2% of the training
time of `192/192/6`, with a negligible fused delta in the accepted screen.

Read `schnet_arch_repaired_2m_30k/decision.md` for the transfer decision.
Earlier 50K and smoke screens remain in their named subdirectories.

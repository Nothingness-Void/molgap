# Corrected physical attempt authorization - 2026-10-13 JST

The user explicitly requested resubmission and investigation/prevention of the
input-path failure. Attempt1's NO_TRAIN terminal evidence remains unchanged.
This authorizes one fresh Kaggle1 version2 on the same kernel and owning branch,
using the same mean2/live and single-forward/EMA0.9999 recipes, initial tensors,
500K/50K rows, seed42, FP32, BS128, optimizer and60-epoch cosine schedule.
The nine-hour T4x2 allocation ceiling and180s cleanup reserve are unchanged;
there is no automatic continuation, successor or protected-role access.

Implementation corrections resolve the unique frozen manifest hash across
flat/nested mounts and resolve shard files relative to that manifest. The
release gate checks supplied manifest bytes and tests the actual packaged
resolver on both layouts, without decoding real graph/label data. The bootstrap
retains the resolved paths and verifies selected shard presence before workers.
These corrections do not change the scientific recipe or qualification gates.

New prospective arm records bind the corrected source and version2 separately
from failed version1. The private source dataset gets a fresh `-v2` identity;
the fixed graph dataset and complete original data manifest hash are retained.

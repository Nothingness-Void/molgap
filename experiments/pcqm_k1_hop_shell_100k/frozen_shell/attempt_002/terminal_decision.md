# Frozen-shell diagnostic — 2026-10-24 JST

Close NO_TRAIN, completed. Original50K predictions reproduced under the explicitly
pinned local runtime with max absolute error1.9073486328125e-6eV, below1e-4eV.
Original LIVE MAE0.1418446451eV; equal-mixture contents at those same ordered
weights produce0.1526593119eV. Mixed-minus-original is+10.814667meV,
paired-row95% interval[+10.227010,+11.395648]meV,2000draws/seed42.

Weights and BN buffers are unchanged, all three ordered flags restored, exact
50K rows aligned. No optimizer, gradient, training, model selection or protected
roles. This is frozen functional reliance on the distance-partitioned contents,
not a10.815meV training gain or independent-role evidence. The parent training
decision remains NEGATIVE_UNDER_CONTRACT because its true paired gain is1.116meV.

Observed process wall50.2417844s, CPU157.1875s, RTX5060 allocation41.0362019s.
Together with closed attempt001, cumulative process wall60.5335427s is below
the original1800s ceiling. No further retry, training or promotion is released.

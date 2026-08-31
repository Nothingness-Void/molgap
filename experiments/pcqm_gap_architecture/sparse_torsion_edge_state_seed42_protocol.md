# Sparse Torsion EdgeState GPS9 Seed-42 Protocol

## Question

Does one sparse persistent torsion state improve the accepted distance-plus-
angle Sparse Triangle EdgeState GPS9 on the frozen official-train-derived
PCQM 100K/10K roles?

## Immutable data and geometry contract

- The freshly trained comparator is
  `ogb_distance_angle_triangle_edge_state_gps9`.
- Its frozen parent source commit is `e083bee19ee6a13cd9f72e91229752a9d5f56389`.
- The accepted parent graph cache SHA-256 is
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- The accepted sparse-wedge cache SHA-256 is
  `dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`.
- The accepted ETKDGv3+MMFF94s single-conformer geometry cache SHA-256 is
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Exactly 100,000 train and 10,000 internal-validation rows are retained.
  The 315 invalid geometry rows remain in their original roles and use the
  explicit zero geometry mask. Official validation and test-dev remain unread.
- The CPU torsion job derives only sparse path ids and periodic features from
  the accepted geometry cache; it does not generate a second conformer.

## Torsion representation

- Enumerate only directed non-backtracking bonded paths `i-j-k-l`.
- Store three participating directed bond ids and the two adjacent wedge ids
  for every path. No dense four-body or bond-pair tensor is allowed.
- Encode the signed ETKDG dihedral with the fixed FP32 feature
  `[sin(phi), cos(phi), sin(2phi), cos(2phi)]`.
- Invalid geometry rows retain all path ids and receive zero features and a
  zero path mask. Degenerate individual paths also receive a zero path mask.
- Torsion shards, the invalid/path-mask ledger, and the aggregate manifest are
  written atomically and must pass no-model acceptance before GPU submission.

## Paired model contract

- Comparator: freshly initialized
  `ogb_distance_angle_triangle_edge_state_gps9`.
- Candidate: freshly initialized
  `ogb_distance_angle_torsion_triangle_edge_state_gps9`, with node width 192,
  real-bond state 64, wedge state 16, torsion state 16, nine GPS blocks, mean
  pooling, and one direct scalar Gap head.
- Torsion state initialization reads the three current bond states, two
  adjacent wedge states, and the fixed four-feature encoding. One shared gated
  update cell is reused across all nine blocks.
- Each block exchanges torsion context sparsely only with the same three bond
  states and two adjacent wedge states. New torsion-to-bond and
  torsion-to-wedge injections are zero-initialized, so the candidate’s initial
  function equals the accepted distance-plus-angle comparator.
- Seed 42 only; FP32; batch 48; AdamW with learning rate `1.6e-4` and weight
  decay `1e-6`; cosine schedule; at most 40 epochs; patience 8; direct Gap
  normalized L1 loss; parameter ceiling 5,200,000.
- The comparator is trained first, and both models use an explicit
  per-seed `torch.Generator` with seed 42 for DataLoader order. Both are fresh
  random initializations; there is no pretraining, checkpoint transfer,
  residual target, prediction fusion, auxiliary loss, second 3D encoder,
  width/grid search, or sealed-role access.

## Execution and stop rule

1. Verify Kaggle1 connectivity and that no Kaggle1 GPU kernel is queued or
   running. Mirror only exact accepted source/cache inputs to private
   `nothingnessvoid` datasets when Kaggle2 assets are not cross-account
   readable; record hashes without printing credentials.
2. Run the separate CPU torsion cache job and accept its immutable output
   without model inference.
3. Submit exactly one sequential Kaggle1 GPU notebook bounded to 23,400
   seconds (6.5 hours). It must checkpoint atomically, emit independently
   retrievable output files, and stop after the paired comparator/candidate.
4. Accept both model artifacts without inference-side recomputation of the
   architecture contract. The torsion candidate advances only if its internal-
   validation Gap MAE is strictly below the freshly trained comparator.
5. A non-improvement closes the torsion mechanism. A strict seed-42 win records
   eligibility for a separately authorized confirmation and stops this task;
   seed 43/44, full-data training, official validation/test-dev, desktop
   submission, and molecular-research-server work remain unauthorized.


# Frozen 500K readout diagnostic protocol

## Question

The Pair Update Norm, Noisy Nodes, and joint GPTrans candidates retained only
1-2 meV of their larger 100K gains under the accepted matched-500K V4 contract.
Does a readout limited to the final virtual node and virtual pair hide useful
real-atom information, or do the gains mainly reflect faster early learning?

This is not a new 500K training or promotion contract. The endpoint models and
development role have already been used for model selection. Every inference
below is exploratory, and no result can be called an unbiased evaluation.

## Frozen inputs and roles

- Baseline checkpoint: accepted matched-500K GPTrans-T `best_model.pt`, SHA256
  `e311f1972ba74ade32e1f16be717c44848235eeafbafd592758775553445c79e`.
- Candidate checkpoint: accepted Pair Update Norm 500K `best_model.pt`, SHA256
  `d46b0bbb9709a5933396a262a1e42323fc45b23e92c7121ffb66468f099529a5`.
- Fixed graph manifest SHA256:
  `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
- Train graph shard 0009, source indices `[450000,500000)`, SHA256
  `4753e34c24d535b56f0da81ff92c5bd70348dee4d1acf216f2cc2c2bca669104`.
- Development graph shard 0010, source indices `[500000,550000)`, SHA256
  `1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1`.
- Frozen-head fitting uses train indices `[450000,490000)`; internal head
  selection uses `[490000,500000)`; final exploratory check uses only the
  first 10,000 internal-development rows `[500000,510000)`.
- The zero-training prediction audit uses all 50,000 internal-development rows.
- Official validation, test-dev and challenge are forbidden.

The graph cache is read-only in the earlier matched-500K worktree. Verify
manifest and selected shard hashes again; do not download or modify it.

## Cheapest falsifier and staged budget

1. Recompute aligned per-row 500K errors for baseline, Pair Norm, Noisy Nodes,
   and their joint candidate. Report signed gains by Gap and atom-count bins.
2. Extract features from the already trained baseline and Pair Norm encoders.
   Reconstruct each stored selected-checkpoint prediction from the frozen
   original readout before fitting any new head. A mismatch aborts.
3. Fit residual readouts only on the frozen training-role features. Compare:
   constant bias correction, 288-d virtual state with width-256 MLP,
   288-d virtual state with width-512 MLP, and 544-d virtual plus mean real-atom
   state with width-256 MLP. All use the same fit/selection rows and seed.
4. Select each residual head using only the train-heldout 10K; inspect the
   development subset once after selections are fixed. No 500K encoder retrain.

Stop the head probe if the local 8 GB GPU cannot pass the extraction smoke test
or the checkpoint/graph identity check. Stop rather than altering the accepted
model or using a protected role.

## Interpretation

If only the wider 288-d head helps, head capacity is plausible. If the
atom-aware head beats the similarly sized wide head for both frozen encoders,
readout information loss becomes plausible. If both encoders improve equally,
that may improve absolute MAE but does not explain the candidate/reference
scale-transfer gap. Null results do not falsify an end-to-end readout redesign:
the frozen encoder was optimized for its original head. One seed, repeated
development use, train-role reuse, and a 10K subset forbid promotion claims.

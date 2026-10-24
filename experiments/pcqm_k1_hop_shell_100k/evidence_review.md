# Evidence review — before execution

Desktop RML trajectory/replay indexes and canonical decisions were read before
this question. No named shell-content experiment was found; an absent name
alone is not proof of novelty. Relevant mechanisms were checked directly:

- ../pcqm_k1_batch_renorm_100k/final_attribution.md: extra detached mean
  correction worsens1.6455meV; selected-LIVE calibration increases its loss.
  Do not retry that correction or explain it away by more exposure.
- ../pcqm_k1_message_norm_100k/attribution.md: learned message amplitude gains
  1.364meV below3meV; neither arm establishes absolute benefit over K1.
- ../pcqm_k1_local_mixing_clean_aux/kaggle3_accuracy_reconciliation_v1/terminal_decision.md:
  SSMA one-hop message mixing loses2.067meV with48.98% step overhead. This
  motivates a25% admission cap, not a guarantee of cheap or useful shells.
- ../top20_architecture_qm9/multihop_edge_state_r8_decision.md and
  sparse_path_attention_r9_decision.md: virtual edges and path-biased attention
  failed their QM9 contracts. This proposal keeps real-bond transport and
  aggregates node content per receiver outside that transport operator.
- ../pcqm_k1_500k_bottleneck_diagnostic/terminal_decision.md: slot deletion
  hurts frozen inference; slot inactivity is contradicted, extra capacity is
  unproven. Hold node/edge/slot widths fixed here.
- ../pcqm_k1_complete_local_audit/historical_evidence.md records cached server
  SPD-PairToken: shared scalar path bias did not improve its parent. The
  pinned primary decision at f426350a was read without adopting server work.
  No shared scalar topology prior or PairToken change is introduced here.
- Active owner pointers for DropPath, BondLocal/pairnorm and WD regularization
  retain their own unresolved states. Do not duplicate or take over those jobs.

GRED (Ding et al., ICML2024), https://arxiv.org/html/2312.01538v3, groups node
content by shortest distance and encodes the hop sequence using a diagonal
linear recurrence. It is an architecture precedent, not measured PCQM4Mv2/K1
benefit. This finite four-hop real residual is an adaptation. NeuralWalker
adds walk coverage/sampling identities; ESA changes the model family and
edge-attention budget. Their greater implementation scope defers them here.

Unresolved deficiency: K1 may benefit from keeping multi-hop chemical content
separated by distance, conditional on its useful existing global exchange.
Alternative: RWSE/local/global processing already provides enough topology,
or an extra residual harms fit/transfer. The cheapest decisive learned test is
the proposed equal-parameter100K pair; frozen inference cannot learn the new
content encoder. Neither outcome implies the historical negative routes reopen.

# Attribution

The 1e-3 GPTrans and K1 residual correlation is 0.8602; K1 has lower
absolute error on 48.07% of development rows despite a worse mean MAE.
These observations support complementary errors, not a causal architecture
claim. Three-way averaging gains 1.224 meV over two GPTrans passes, at one
additional model pass. It has not been compared against a third GPTrans pass
and was evaluated after repeated development-set use.

The two-model oracle is 19.397 meV below its equal average. The three-model
oracle is 26.883 meV below its equal average. These gaps measure only theoretical
room using labels, not achievable supervised routing gains. Generalization,
router training, native deployment cost, and seed variability remain unknown.

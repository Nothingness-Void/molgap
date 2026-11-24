# TokenGT-GPTrans Local 100K Candidate Screen

Evaluates `gptrans_tokengt/1` (`TokenGTGPTrans`, 5,928,961 parameters,
Chebyshev T_0..T_15 node+edge tokenization with 4-class token-incidence bias)
on `gptrans_t/2` with AdamW and EMA 0.999 over `pcqm_fixed_100k_v1`.

- Implementation: [gptrans_tokengt.py](../../src/molgap/gptrans_tokengt.py)
- Contract and gate: [protocol.md](protocol.md)
- Authority: [authority.json](authority.json)
- Role plan: [role_plan.json](role_plan.json)
- Training recipe: [recipe.json](recipe.json)
- Decision: [decision.md](decision.md)

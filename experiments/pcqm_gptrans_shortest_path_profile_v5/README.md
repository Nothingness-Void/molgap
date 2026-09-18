# GPTrans shortest-path runtime profile

This profiling-only experiment measured whether replacing repeated online
shortest-path construction with an exact immutable cached tensor could
accelerate the frozen GPTrans-T execution shape. Read `protocol.md` for the
frozen question, `results/decision_122484011.md` for the terminal decision and
`profiling_contract.json` for the compact frozen contract.

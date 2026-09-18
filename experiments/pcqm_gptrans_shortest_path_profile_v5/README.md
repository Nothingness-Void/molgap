# GPTrans shortest-path runtime profile

This profiling-only experiment measures whether replacing repeated online
shortest-path construction with an exact immutable cached tensor can accelerate
the frozen GPTrans-T execution shape. Read `protocol.md` for the frozen question
and `STATUS.md` for the remote job. The release decision and its compact frozen
contract are in `decision.md` and `profiling_contract.json`.

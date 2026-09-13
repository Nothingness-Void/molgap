# Terminal handoff incident — 2026-09-13

The kernel reached `COMPLETE`, but the active heartbeat produced no coordinator
message and remained active. Inspection showed that its persistent task still
used a detached, stale `.codex/worktrees` checkout that did not contain this
experiment protocol.

The coordinator manually retrieved and accepted the terminal artifacts, then
paused automation `molgap-k1-v4-kaggle2-monitor`. A controlled readiness test
then verified the authoritative checkout and protocol, but its attempted
cross-task message failed with `MCP tool call requires approval, but approval
policy is never`. The delegated heartbeat therefore cannot provide a reliable
direct coordinator message in this environment.

The repaired design keeps Luna on the persistent monitor task for Kaggle
polling, retrieval, and no-inference acceptance. It atomically sets
`handoff_ready=true` under the authoritative artifact record. A second
heartbeat attached to the existing coordinator task reads only that local
marker, performs scientific analysis when ready, and pauses both automations.
It creates no new task/chat and avoids repeated remote work.

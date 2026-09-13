# Terminal handoff incident — 2026-09-13

The kernel reached `COMPLETE`, but the active heartbeat produced no coordinator
message and remained active. Inspection showed that its persistent task still
used a detached, stale `.codex/worktrees` checkout that did not contain this
experiment protocol. The prompt also allowed a heartbeat turn to end without a
durable delivery stage or a verified `send_message_to_thread` acknowledgement.

The coordinator manually retrieved and accepted the terminal artifacts, then
paused automation `molgap-k1-v4-kaggle2-monitor`. Its replacement contract uses
the authoritative `C:\Users\Adminn\Documents\molgap` checkout, an atomic staged
handoff marker, and acknowledgement-gated delivery. A failed or missing message
acknowledgement must retain the active heartbeat and retry message delivery
only; retrieval and acceptance must not repeat.


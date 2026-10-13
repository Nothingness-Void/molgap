# Shared-Session Execution Speed

One bounded synthetic execution question, not a model candidate or training
release. Read [protocol](protocol.md), frozen prospective [decision](decision.md)
and eventual `terminal_decision.md` before interpreting timings.

`prepare.py` freezes actual historical/current Git source snapshots and uses the
RML planner. `run.py` delegates to the shared diagnostic, which calls the existing
family loop/kernel/loader/session/output owners. `close.py` rechecks equality and
reuses V5/RML finalization. Source ZIPs are frozen replay oracles, not a live slow
training option. `results/` contains raw timings and ignored scratch model files.
Do not read synthetic metrics as measured molecular errors or strict replay.

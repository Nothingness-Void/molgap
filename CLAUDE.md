# Claude Code Entry Point

Read `AGENTS.md`. It is the single source for the repository reading protocol,
hard constraints, and document map. This file intentionally duplicates none of
those facts.

## Delegation

Delegate broad repository searches and whole pytest groups to a Haiku 5.5 subagent
so the main context stays small: `search` maps files, `grep` finds call sites,
`audit` checks a committed diff in its own worktree. For a test run, launch a
general subagent with the Haiku model and the exact pytest command. Subagents
report paths, counts and failing node IDs, not raw output. Read a known file or
run a single targeted check directly; a subagent costs more than it saves there.

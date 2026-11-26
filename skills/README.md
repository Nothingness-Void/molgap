# Project skills

These five skills belong to the project. The copies here are the authority;
edit them here and review the change like code.

| Skill | Use it to |
|---|---|
| `kaggle-molgap-workloads` | Submit, retrieve and accept a Kaggle run |
| `ims-molgap-workloads` | Submit, retrieve and reconcile IMS CPU/A100 jobs |
| `scnet-bw-dcu-molgap` | Stage and recover SCNet Slurm jobs |
| `molgap-experiment-reuse` | Find the existing owner before changing infrastructure |
| `molgap-obsidian-vault` | Maintain the Obsidian vault (see [VAULT](../docs/operations/VAULT.md)) |

The user skill directories (`~/.agents/skills/`, `~/.claude/skills/`, `~/.codex/skills/`,
and `~/.gemini/config/skills/`) are linked directly to this folder via junctions, so edits
here immediately apply to every agent tool. Relative paths inside each `SKILL.md`
(for example `references/...`) stay inside its own folder.

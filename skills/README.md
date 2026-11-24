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

The agent tools read them from the user directory (`~/.agents/skills/`, linked
from `~/.claude/skills/` and `~/.codex/skills/`). After this branch merges, the
user-directory copies are to be replaced by links to this directory; until then
they are separate copies, so edit here and copy out. Relative paths inside each
`SKILL.md` (for example `references/...`) stay inside its own folder.

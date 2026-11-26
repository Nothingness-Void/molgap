"""Full-feature MolGap Agent Infrastructure Benchmark & Improvement Gate.

Compares a baseline Git ref (--base) against a candidate ref or dirty WORKTREE
(--candidate) across 12 read-only tasks covering every MolGap subsystem:
  - Core (T1-T6): RML CLI/levers, Kaggle resume/closure, RML contracts & precedence,
    hard scientific/Windows constraints, live catalog, V5 & rule conflicts.
  - Full (T1-T12): adds codebase ownership (T7), new-experiment E2E planning (T8),
    IMS cluster & dashboard (T9), SCNet/DCU workflows (T10), Track A production
    inference & CURRENT_STATE (T11), local infra_budget / testing / Obsidian vault (T12).

Gate contract (exit 0 = IMPROVED, exit 1 = REGRESSION or NO_IMPROVEMENT, exit 2 = error):
  - Zero regression on pass rate, assertion accuracy, tokens (+5% tol), and tool calls (+5% tol).
  - Strictly improved accuracy/pass rate OR >= min-gain-pct (default 5%) efficiency gain.

Usage examples:
  .venv\\Scripts\\python.exe platforms/local/agent_infra_bench.py --list-tasks
  .venv\\Scripts\\python.exe platforms/local/agent_infra_bench.py --base HEAD --candidate WORKTREE --model haiku --suite full
  .venv\\Scripts\\python.exe platforms/local/agent_infra_bench.py --base d39902d08 --candidate HEAD --model haiku --suite full
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from molgap.constants import REPO_ROOT


WORKTREE_BASE = Path(os.environ.get("MOLGAP_BENCH_WORKTREE_BASE", r"D:\molgap-worktrees"))
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(sys.executable)

EXPECTED_FAILURE_MODES = {
    "infra_failure", "gate_or_cost_stop", "lost_at_scale", "early_gain_fades",
    "endpoint_regression", "below_gate", "confounded_change",
    "comparator_missing", "discriminator_missing", "boundary_pass",
}
EXPECTED_CLOSURE_KEYS = {"format", "failure_mode", "search_terms", "effect_meV", "ci_meV", "source"}
EXPECTED_PROTECTED_ROLES = {"official_validation", "test_dev", "test_challenge"}


COMMON_PREAMBLE = (
    "You are running inside a read-only MolGap repository checkout (current working directory '.').\n"
    "Rules for this task:\n"
    "- Stay strictly inside '.' — do NOT read files outside '.' and do NOT run git log/diff/show.\n"
    "- Do NOT create, edit, or delete any file, and do NOT contact remote services.\n"
    f"- When running Python commands, use `{PYTHON_EXE.as_posix()}`.\n"
    "- Return ONLY a single valid JSON object inside a ```json ... ``` code fence as your final message.\n\n"
)

TASKS: list[dict] = [
    {
        "id": "T1_cli_discovery_levers",
        "suite": "core",
        "name": "RML CLI发现+枚举+实操levers",
        "checks_count": 5,
        "prompt": COMMON_PREAMBLE + (
            "Task T1:\n"
            "1. Determine the worktree-safe local command prefix to invoke `molgap.research_memory`.\n"
            "2. List ALL valid enum values for `lookup --failure-mode`.\n"
            "3. Actually execute the `levers` subcommand for `--family 'pcqm-k1-*-100k'` and read the exact `effect_meV` "
            "values for `pcqm-k1-batch-renorm-100k` and `pcqm-k1-extremal-readout-100k`.\n\n"
            "Return JSON with exact keys:\n"
            '{"rml_command_prefix": "<string>", "failure_modes": ["..."], '
            '"batch_renorm_effect_meV": <number>, "extremal_readout_effect_meV": <number>}'
        ),
    },
    {
        "id": "T2_kaggle_resume_closure",
        "suite": "core",
        "name": "Kaggle续跑闭环命令+防踩坑",
        "checks_count": 5,
        "prompt": COMMON_PREAMBLE + (
            "Task T2 (read docs/skills only, do not execute remote or write commands):\n"
            "1. Where must `draft-closure --output-dir` be located, and what error code happens if placed at repository root?\n"
            "2. Which CLI flag (exact flag name only, e.g. `--foo`) passes the bound continuation directory to retrieval, diagnosis, `draft-closure`, and acceptance?\n"
            "3. After a successful closure via `accept_run.py --execute`, should you run `rebuild` again? (true/false)\n"
            "4. If the closure's derived-index update fails, what is the exact worktree-safe command to recover it?\n\n"
            "Return JSON with exact keys:\n"
            '{"output_dir_parent_rule": "<string>", "root_dir_error_code": "<string>", '
            '"continuation_flag": "<string>", "run_rebuild_after_successful_closure": <bool>, '
            '"terminal_rebuild_command": "<string>"}'
        ),
    },
    {
        "id": "T3_rml_contracts_precedence",
        "suite": "core",
        "name": "RML数据契约+冲突裁决",
        "checks_count": 5,
        "prompt": COMMON_PREAMBLE + (
            "Task T3 (read docs/code only):\n"
            "1. List all allowed top-level JSON keys for `molgap-closure-facts-v1`.\n"
            "2. What terminal outcome must the reference arm have closed with for `effect_meV` to be non-null?\n"
            "3. When `validate` passes (with a narrative erratum in `known_gaps`) while `accept_run.py --strict` returns `BLOCKED`, "
            "does `validate` unblock strict replay/promotion? (true/false) And which verifier governs replay/comparison/promotion?\n"
            "4. List the 3 protected role names refused by `analyze --role-id`.\n\n"
            "Return JSON with exact keys:\n"
            '{"closure_facts_keys": ["..."], "required_reference_outcome": "<string>", '
            '"validate_unblocks_replay": <bool>, "governing_verifier": "<string>", '
            '"refused_analyze_roles": ["..."]}'
        ),
    },
    {
        "id": "T4_hard_constraints_gotchas",
        "suite": "core",
        "name": "科研与Windows硬红线速查",
        "checks_count": 7,
        "prompt": COMMON_PREAMBLE + (
            "Task T4 (read AGENTS.md and pointers):\n"
            "1. How many rows are permanently reserved in `tail_holdout_v1`, and is tail MAE directly comparable to 50K internal dev MAE? (true/false)\n"
            "2. What ASCII Windows junction path must `.pth` hold, and what `mklink` command rebuilds it if `pip install -e .` overwrites it?\n"
            "3. May a new experiment edit shared trainers or tests directly? (true/false) And what is required if test code growth exceeds feature code growth?\n"
            "4. What file must be read before ANY IMS command, and what `/lustre/...` path prefix must all IMS operations stay under?\n\n"
            "Return JSON with exact keys:\n"
            '{"tail_holdout_rows": <int>, "tail_comparable_to_50k_dev": <bool>, '
            '"pth_junction_path": "<string>", "mklink_command": "<string>", '
            '"new_exp_may_edit_shared_trainers": <bool>, "test_growth_exceeds_feature_action": "<string>", '
            '"ims_required_doc": "<string>", "ims_allowed_prefix": "<string>"}'
        ),
    },
    {
        "id": "T5_live_catalog_extension",
        "suite": "core",
        "name": "实操catalog能力核验",
        "checks_count": 5,
        "prompt": COMMON_PREAMBLE + (
            "Task T5:\n"
            "1. Actually execute `catalog --compact` in this checkout.\n"
            "2. Return the exact list of `<name>/<version>` for all `family` entries with `runnable=true`.\n"
            "3. Count how many `addon` entries have `runnable=true` and how many have `runnable=false`.\n"
            "4. Does `runnable=false` forbid implementing that idea? (true/false)\n\n"
            "Return JSON with exact keys:\n"
            '{"runnable_families": ["..."], "runnable_addons_count": <int>, '
            '"unrunnable_addons_count": <int>, "runnable_false_forbids_work": <bool>}'
        ),
    },
    {
        "id": "T6_v5_and_rule_conflicts",
        "suite": "core",
        "name": "规则冲突检测+V5离线红线",
        "checks_count": 3,
        "prompt": COMMON_PREAMBLE + (
            "Task T6:\n"
            "Read `AGENTS.md` and `.claude/CLAUDE.md` in this checkout.\n"
            "1. List any specific model names mentioned in `AGENTS.md` for subagent/subtask delegation (return `[]` if none).\n"
            "2. Do `AGENTS.md` and `.claude/CLAUDE.md` prescribe conflicting subagent delegation models? (true/false)\n"
            "3. Under Desktop Offline / V5, is it permitted to add desktop-to-server monitor handoff, automatic takeover, "
            "or a shared cross-machine live SQLite DB? (true/false)\n\n"
            "Return JSON with exact keys:\n"
            '{"agents_md_delegation_models": ["..."], "has_delegation_conflict": <bool>, '
            '"allow_desktop_server_takeover": <bool>}'
        ),
    },
    {
        "id": "T7_codebase_ownership_map",
        "suite": "full",
        "name": "代码库8大模块归属",
        "checks_count": 8,
        "prompt": COMMON_PREAMBLE + (
            "Task T7 (Repo Architecture & Ownership Understanding):\n"
            "Using the repository docs (starting from AGENTS.md / ARCHITECTURE.md), identify the exact repository-relative "
            "file path (forward slashes) that owns each of these 8 components:\n"
            "1. Public inference API (`public_inference_file`)\n"
            "2. GPTrans-T node/pair propagation core (`gptrans_core_file`)\n"
            "3. Registered true-bond residual addon on GPTrans (`gptrans_bond_local_file`)\n"
            "4. Family/addon registry (`addon_catalog_file`)\n"
            "5. RML schemas / `TRAJECTORY_OUTCOMES` / `FAILURE_MODES` (`rml_schemas_file`)\n"
            "6. Worktree-safe selected-checkout launcher (`worktree_launcher_file`)\n"
            "7. Kaggle status/quota web dashboard script (`kaggle_dashboard_file`)\n"
            "8. IMS queue collector & dashboard script (`ims_dashboard_file`)\n\n"
            "Return JSON with those exact 8 keys."
        ),
    },
    {
        "id": "T8_new_experiment_end_to_end",
        "suite": "full",
        "name": "端到端新实验规划",
        "checks_count": 7,
        "prompt": COMMON_PREAMBLE + (
            "Task T8 (End-to-End New Kaggle Experiment Planning, read docs only):\n"
            "Suppose you need to propose and launch a brand-new desktop GPTrans-T 100K mechanism experiment on Kaggle. Extract:\n"
            "1. Required branch naming pattern (`branch_pattern`) and base branch (`base_branch`) for a new desktop question.\n"
            "2. The 3 human-authored files required in `experiments/<q>/` before `freeze-input` (`required_author_files`, list of filenames).\n"
            "3. The pre-submit reviewer record filename (`pre_submit_review_file`) and the exact 2 verdicts accepted by the Kaggle submitter "
            "(`accepted_review_verdicts`, list of strings).\n"
            "4. Repository-relative path of the Kaggle submission script (`kaggle_submit_script`) and the infrastructure budget checker script "
            "(`infra_budget_script`).\n\n"
            "Return JSON with exact keys:\n"
            '{"branch_pattern": "<string>", "base_branch": "<string>", "required_author_files": ["..."], '
            '"pre_submit_review_file": "<string>", "accepted_review_verdicts": ["..."], '
            '"kaggle_submit_script": "<string>", "infra_budget_script": "<string>"}'
        ),
    },
    {
        "id": "T9_ims_workloads_and_dashboard",
        "suite": "full",
        "name": "IMS集群提交与看板规范",
        "checks_count": 6,
        "prompt": COMMON_PREAMBLE + (
            "Task T9 (IMS Cluster & Dashboard, read docs/skills only):\n"
            "1. Repository-relative path of the Windows V4 IMS submission script (`ims_submit_script`).\n"
            "2. GPU memory per IMS A100 card in GB (`a100_vram_gb`, int) and host RAM per CPU core in GB (`host_ram_gb_per_core`, float).\n"
            "3. Live queue inspection command on the IMS remote session (`ims_live_queue_cmd`).\n"
            "4. Local bind port of the always-on IMS dashboard (`ims_dashboard_port`, int) and stale-data threshold in seconds (`ims_dashboard_stale_seconds`, int).\n\n"
            "Return JSON with exact keys:\n"
            '{"ims_submit_script": "<string>", "a100_vram_gb": <int>, "host_ram_gb_per_core": <float>, '
            '"ims_live_queue_cmd": "<string>", "ims_dashboard_port": <int>, "ims_dashboard_stale_seconds": <int>}'
        ),
    },
    {
        "id": "T10_scnet_dcu_workloads",
        "suite": "full",
        "name": "SCNet超算DCU与CPU分工",
        "checks_count": 5,
        "prompt": COMMON_PREAMBLE + (
            "Task T10 (SCNet BW-1 CPU/DCU Workloads, read docs/skills only):\n"
            "1. Is there a single common SCNet submission frontend? (`has_common_scnet_frontend`, bool)\n"
            "2. Repository-relative path of the DCU environment checker script (`dcu_check_script`).\n"
            "3. Does `--allow-cpu` on that checker qualify DCU execution? (`allow_cpu_qualifies_dcu`, bool)\n"
            "4. On which device class (`CPU` or `DCU`) must new ETKDG/MMFF graph caches be constructed before DCU training? (`graph_cache_device`)\n"
            "5. Which Slurm command inspects currently queued/running jobs on the login node? (`slurm_queue_cmd`)\n\n"
            "Return JSON with exact keys:\n"
            '{"has_common_scnet_frontend": <bool>, "dcu_check_script": "<string>", '
            '"allow_cpu_qualifies_dcu": <bool>, "graph_cache_device": "<string>", "slurm_queue_cmd": "<string>"}'
        ),
    },
    {
        "id": "T11_track_a_inference_and_state",
        "suite": "full",
        "name": "TrackA生产推理API与真相源",
        "checks_count": 6,
        "prompt": COMMON_PREAMBLE + (
            "Task T11 (Track A Production Inference & CURRENT_STATE, read docs/code only):\n"
            "1. From `CURRENT_STATE.md`, what is the recommended Track A registry key (`track_a_registry_key`) and public loader function name (`track_a_loader_fn`)?\n"
            "2. From `README.md`, what is the corresponding batch prediction function name (`track_a_batch_predict_fn`)?\n"
            "3. In what exact order are the 3 target columns returned (`output_target_order`, list of 3 lowercase strings)?\n"
            "4. How should callers align predictions back to input SMILES when some graphs fail to build (`alignment_key`, e.g. variable name)?\n"
            "5. Are predictions experimental solid-state IP/EA values? (`is_experimental_solid_state`, bool)\n\n"
            "Return JSON with exact keys:\n"
            '{"track_a_registry_key": "<string>", "track_a_loader_fn": "<string>", '
            '"track_a_batch_predict_fn": "<string>", "output_target_order": ["...", "...", "..."], '
            '"alignment_key": "<string>", "is_experimental_solid_state": <bool>}'
        ),
    },
    {
        "id": "T12_local_verification_budget_vault",
        "suite": "full",
        "name": "实操基建预算+测试范围+Vault边界",
        "checks_count": 6,
        "prompt": COMMON_PREAMBLE + (
            "Task T12 (Live infra_budget + Local Testing + Obsidian Vault):\n"
            "1. Actually execute `platforms/local/infra_budget.py` in this checkout and report the live measured integers "
            "`src_py_lines` and `docs_operations_lines`.\n"
            "2. From `docs/operations/EXPERIMENT_CLI.md`, which 2 `--scope` names are recommended for `molgap.local_testing`? (`local_testing_scopes`, list of strings)\n"
            "3. From `skills/molgap-obsidian-vault/SKILL.md` (and `AGENTS.md`), is the Obsidian vault an authoritative source of scientific truth? (`vault_is_authority`, bool)\n"
            "4. List the 2 mandatory finish-gate Python script filenames under the vault's `_scripts/` directory (`vault_gate_scripts`, list of 2 filenames).\n\n"
            "Return JSON with exact keys:\n"
            '{"src_py_lines": <int>, "docs_operations_lines": <int>, '
            '"local_testing_scopes": ["..."], "vault_is_authority": <bool>, "vault_gate_scripts": ["...", "..."]}'
        ),
    },
]


def find_claude_exe() -> str:
    for candidate in (
        r"C:\nvm4w\nodejs\node_modules\@anthropic-ai\claude-code\bin\claude.exe",
        shutil.which("claude.exe"),
        shutil.which("claude.cmd"),
        shutil.which("claude"),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("claude CLI executable not found on PATH")


def _remove_worktree(wt: Path) -> None:
    if wt.exists():
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True, text=True)
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)


def _dirty_relative_paths() -> tuple[list[str], list[str]]:
    """Return (modified_or_added_paths, deleted_paths) in REPO_ROOT working tree."""
    out = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uall"],
        text=True, encoding="utf-8",
    )
    copy_paths: list[str] = []
    del_paths: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        code, rel = line[:2], line[3:].strip().strip('"')
        if " -> " in rel:
            old_p, new_p = rel.split(" -> ", 1)
            del_paths.append(old_p.strip('"'))
            copy_paths.append(new_p.strip('"'))
        elif "D" in code:
            del_paths.append(rel)
        else:
            copy_paths.append(rel)
    return copy_paths, del_paths


def setup_isolated_worktrees(base_ref: str, cand_ref: str, tag: str) -> tuple[Path, Path]:
    WORKTREE_BASE.mkdir(parents=True, exist_ok=True)
    base_wt = WORKTREE_BASE / f"_agent_bench_{tag}_base"
    cand_wt = WORKTREE_BASE / f"_agent_bench_{tag}_cand"
    for wt in (base_wt, cand_wt):
        _remove_worktree(wt)
    subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "prune"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(base_wt), base_ref],
                   check=True, capture_output=True, text=True)
    if cand_ref.upper() == "WORKTREE":
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(cand_wt), "HEAD"],
                       check=True, capture_output=True, text=True)
        copy_paths, del_paths = _dirty_relative_paths()
        for rel in del_paths:
            target = cand_wt / rel
            if target.exists():
                target.unlink()
        for rel in copy_paths:
            src = REPO_ROOT / rel
            if src.is_file():
                dst = cand_wt / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    else:
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(cand_wt), cand_ref],
                       check=True, capture_output=True, text=True)
    return base_wt, cand_wt


def cleanup_worktrees(base_wt: Path, cand_wt: Path) -> None:
    _remove_worktree(base_wt)
    _remove_worktree(cand_wt)
    subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "prune"], capture_output=True)


def collect_arm_truth(arm_dir: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(arm_dir / "src")
    env["PYTHONIOENCODING"] = "utf-8"

    cat = subprocess.run(
        [str(PYTHON_EXE), "platforms/local/run.py", "--", "catalog", "--compact"],
        cwd=str(arm_dir), env=env, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    fams_true, add_true, add_false = set(), 0, 0
    for line in cat.stdout.splitlines():
        s = line.strip()
        if s.startswith("family ") and "runnable=true" in s:
            fams_true.add(s.split()[1])
        elif s.startswith("addon "):
            if "runnable=true" in s:
                add_true += 1
            elif "runnable=false" in s:
                add_false += 1

    bud = subprocess.run(
        [str(PYTHON_EXE), "platforms/local/infra_budget.py"],
        cwd=str(arm_dir), env=env, capture_output=True, text=True, encoding="utf-8",
    )
    metrics = {}
    for line in bud.stdout.splitlines():
        m = re.match(r"^\s*([a-z0-9_]+)\s+(\d+)\s*/", line)
        if m:
            metrics[m.group(1)] = int(m.group(2))

    agents_text = (arm_dir / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    has_old_delegation = ("gpt-6-luna" in agents_text and "gpt-6.1-sol" in agents_text)

    return {
        "runnable_families": fams_true,
        "runnable_addons_count": add_true,
        "unrunnable_addons_count": add_false,
        "src_py_lines": metrics.get("src_py_lines"),
        "docs_operations_lines": metrics.get("docs_operations_lines"),
        "has_old_delegation": has_old_delegation,
    }


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = list(reversed(fenced))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                return val
        except Exception:
            continue
    return None


def score_task(task: dict, ans: dict | None, tool_events: list[dict], truth: dict) -> tuple[int, int, dict]:
    tid = task["id"]
    if not isinstance(ans, dict):
        return 0, task["checks_count"], {"valid_json": False}

    checks: dict[str, bool] = {}
    shell_cmds = " ".join(
        str(e.get("input", {}).get("command", ""))
        for e in tool_events
        if e.get("name") in ("Bash", "PowerShell", "run_command")
    )
    norm = lambda k: str(ans.get(k, "")).replace("\\", "/").lower()

    if tid == "T1_cli_discovery_levers":
        prefix = str(ans.get("rml_command_prefix", ""))
        checks["worktree_safe_prefix"] = ("run.py" in prefix and "molgap.research_memory" in prefix)
        checks["all_10_failure_modes"] = ({str(x).strip() for x in (ans.get("failure_modes") or [])} == EXPECTED_FAILURE_MODES)
        try:
            checks["batch_renorm_1.65"] = abs(float(ans.get("batch_renorm_effect_meV")) - 1.65) < 1e-3
        except Exception:
            checks["batch_renorm_1.65"] = False
        try:
            checks["extremal_readout_3.33"] = abs(float(ans.get("extremal_readout_effect_meV")) - 3.33) < 1e-3
        except Exception:
            checks["extremal_readout_3.33"] = False
        checks["executed_levers_cli"] = ("levers" in shell_cmds)

    elif tid == "T2_kaggle_resume_closure":
        checks["output_dir_under_experiments"] = ("experiments/" in norm("output_dir_parent_rule"))
        checks["root_error_finalization_failed"] = ("FINALIZATION_FAILED" in str(ans.get("root_dir_error_code", "")).upper())
        cont = str(ans.get("continuation_flag", "")).strip().split()[0] if str(ans.get("continuation_flag", "")).strip() else ""
        checks["continuation_flag"] = (cont == "--resume-prepared")
        checks["no_rebuild_after_closure"] = (ans.get("run_rebuild_after_successful_closure") is False)
        rec = str(ans.get("terminal_rebuild_command", ""))
        checks["worktree_safe_terminal_rebuild"] = (
            "run.py" in rec and "molgap.research_memory" in rec
            and "terminal-rebuild" in rec and "--trajectory" in rec
        )

    elif tid == "T3_rml_contracts_precedence":
        checks["closure_facts_6_keys"] = ({str(x).strip() for x in (ans.get("closure_facts_keys") or [])} == EXPECTED_CLOSURE_KEYS)
        checks["ref_outcome_closed"] = (str(ans.get("required_reference_outcome", "")).strip().upper() == "CLOSED")
        checks["validate_does_not_unblock"] = (ans.get("validate_unblocks_replay") is False)
        gov = norm("governing_verifier")
        checks["strict_governs"] = ("strict" in gov or "accept_run" in gov)
        checks["refused_3_roles"] = ({str(x).strip() for x in (ans.get("refused_analyze_roles") or [])} == EXPECTED_PROTECTED_ROLES)

    elif tid == "T4_hard_constraints_gotchas":
        checks["tail_rows_128606"] = (ans.get("tail_holdout_rows") == 128606)
        checks["tail_not_comparable"] = (ans.get("tail_comparable_to_50k_dev") is False)
        checks["pth_junction_c_molgap_src"] = ("c:/molgap-src" in norm("pth_junction_path"))
        mk = norm("mklink_command")
        checks["mklink_j_repair"] = ("mklink" in mk and "/j" in mk and "c:/molgap-src" in mk)
        checks["no_shared_trainer_edit"] = (ans.get("new_exp_may_edit_shared_trainers") is False)
        act = norm("test_growth_exceeds_feature_action")
        checks["rework_on_test_bloat"] = ("rework" in act or "返工" in act)
        checks["ims_doc_and_prefix"] = ("remote_handoff.md" in norm("ims_required_doc") and "/lustre/home/users/sm2/chou" in norm("ims_allowed_prefix"))

    elif tid == "T5_live_catalog_extension":
        checks["runnable_families_exact"] = ({str(x).strip() for x in (ans.get("runnable_families") or [])} == truth["runnable_families"])
        checks["runnable_addons_count"] = (ans.get("runnable_addons_count") == truth["runnable_addons_count"])
        checks["unrunnable_addons_count"] = (ans.get("unrunnable_addons_count") == truth["unrunnable_addons_count"])
        checks["false_does_not_forbid"] = (ans.get("runnable_false_forbids_work") is False)
        checks["executed_catalog_cli"] = ("catalog" in shell_cmds)

    elif tid == "T6_v5_and_rule_conflicts":
        models = [str(x).lower() for x in (ans.get("agents_md_delegation_models") or [])]
        if truth["has_old_delegation"]:
            checks["delegation_models_accurate"] = (any("gpt-6-luna" in m for m in models) and any("gpt-6.1-sol" in m for m in models))
            checks["conflict_detected_accurately"] = (ans.get("has_delegation_conflict") is True)
        else:
            checks["delegation_models_accurate"] = (len(models) == 0)
            checks["conflict_detected_accurately"] = (ans.get("has_delegation_conflict") is False)
        checks["v5_forbids_takeover"] = (ans.get("allow_desktop_server_takeover") is False)

    elif tid == "T7_codebase_ownership_map":
        checks["public_inference"] = ("src/molgap/inference.py" in norm("public_inference_file"))
        checks["gptrans_core"] = ("src/molgap/gptrans.py" in norm("gptrans_core_file"))
        checks["gptrans_bond_local"] = ("src/molgap/gptrans_bond_local.py" in norm("gptrans_bond_local_file"))
        checks["addon_catalog"] = ("src/molgap/experiment_catalog.py" in norm("addon_catalog_file"))
        checks["rml_schemas"] = ("src/molgap/research_memory/schemas.py" in norm("rml_schemas_file"))
        checks["worktree_launcher"] = ("platforms/local/run.py" in norm("worktree_launcher_file"))
        checks["kaggle_dashboard"] = ("platforms/kaggle/dashboard.py" in norm("kaggle_dashboard_file"))
        checks["ims_dashboard"] = ("platforms/ims/check_queue.py" in norm("ims_dashboard_file"))

    elif tid == "T8_new_experiment_end_to_end":
        checks["branch_pattern"] = ("codex/exp/" in norm("branch_pattern"))
        checks["base_branch"] = ("molgap-desktop" in norm("base_branch"))
        req_files = {str(x).strip().lower() for x in (ans.get("required_author_files") or [])}
        checks["three_author_files"] = ({"experiment.json", "protocol.md", "decision.md"} <= req_files)
        checks["review_file"] = ("pre_submit_review.json" in norm("pre_submit_review_file"))
        verdicts = {str(x).strip().upper() for x in (ans.get("accepted_review_verdicts") or [])}
        checks["review_verdicts"] = (verdicts == {"APPROVE", "APPROVE_WITH_CHANGES"})
        checks["submit_script"] = ("platforms/kaggle/submit_workflow.py" in norm("kaggle_submit_script"))
        checks["infra_budget_script"] = ("platforms/local/infra_budget.py" in norm("infra_budget_script"))

    elif tid == "T9_ims_workloads_and_dashboard":
        checks["ims_submit_script"] = ("platforms/ims/v4_submit/submit_from_windows.py" in norm("ims_submit_script"))
        checks["a100_vram_80"] = (ans.get("a100_vram_gb") == 80)
        try:
            checks["host_ram_1.875"] = abs(float(ans.get("host_ram_gb_per_core")) - 1.875) < 1e-3
        except Exception:
            checks["host_ram_1.875"] = False
        checks["ims_live_queue_cmd"] = ("jobinfo" in norm("ims_live_queue_cmd"))
        checks["ims_port_8765"] = (ans.get("ims_dashboard_port") == 8765)
        checks["ims_stale_240"] = (ans.get("ims_dashboard_stale_seconds") == 240)

    elif tid == "T10_scnet_dcu_workloads":
        checks["no_common_frontend"] = (ans.get("has_common_scnet_frontend") is False)
        checks["dcu_check_script"] = ("platforms/scnet/check_dcu_environment.py" in norm("dcu_check_script"))
        checks["allow_cpu_no_qualify"] = (ans.get("allow_cpu_qualifies_dcu") is False)
        checks["graph_cache_cpu"] = ("cpu" in norm("graph_cache_device"))
        checks["slurm_squeue"] = ("squeue" in norm("slurm_queue_cmd"))

    elif tid == "T11_track_a_inference_and_state":
        checks["registry_key"] = (str(ans.get("track_a_registry_key", "")).strip() == "repaired_2m_dense_2d")
        checks["loader_fn"] = ("load_repaired_2m_2d" in str(ans.get("track_a_loader_fn", "")))
        checks["batch_predict_fn"] = ("predict_smiles_batch_repaired_2m_2d" in str(ans.get("track_a_batch_predict_fn", "")))
        order = [str(x).strip().lower() for x in (ans.get("output_target_order") or [])]
        checks["target_order"] = (order == ["homo", "lumo", "gap"])
        checks["alignment_valid_idx"] = ("valid_idx" in norm("alignment_key"))
        checks["not_experimental_solid_state"] = (ans.get("is_experimental_solid_state") is False)

    elif tid == "T12_local_verification_budget_vault":
        checks["live_src_py_lines"] = (ans.get("src_py_lines") == truth["src_py_lines"])
        checks["live_docs_ops_lines"] = (ans.get("docs_operations_lines") == truth["docs_operations_lines"])
        scopes = {str(x).strip().lower() for x in (ans.get("local_testing_scopes") or [])}
        checks["testing_scopes"] = ({"prepare", "accept"} <= scopes)
        checks["vault_not_authority"] = (ans.get("vault_is_authority") is False)
        vscripts = {Path(str(x).strip()).name.lower() for x in (ans.get("vault_gate_scripts") or [])}
        checks["vault_gate_scripts"] = ({"build_vault.py", "validate_vault.py"} <= vscripts)
        checks["executed_infra_budget"] = ("infra_budget.py" in shell_cmds)

    passed = sum(1 for v in checks.values() if v)
    return passed, len(checks), checks


def run_claude_one(claude_exe: str, model: str, arm: str, arm_dir: Path, task: dict, truth: dict,
                   repeat_idx: int, max_turns: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(arm_dir / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        claude_exe, "-p", task["prompt"],
        "--model", model,
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--setting-sources", "project",
        "--strict-mcp-config",
        "--permission-mode", "bypassPermissions",
        "--allowedTools", "Read,Glob,Grep,Bash",
        "--max-turns", str(max_turns),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(arm_dir), env=env,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=240,
    )
    wall_s = time.perf_counter() - t0

    tool_events: list[dict] = []
    result_event: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        if evt.get("type") == "assistant":
            for block in (evt.get("message", {}).get("content", []) or []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_events.append({"name": block.get("name"), "input": block.get("input", {})})
        elif evt.get("type") == "result":
            result_event = evt

    parsed = extract_json(str(result_event.get("result", "")))
    passed, total, checks = score_task(task, parsed, tool_events, truth)
    usage = result_event.get("usage", {}) or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    cc_tok = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cr_tok = int(usage.get("cache_read_input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)

    return {
        "arm": arm, "repeat": repeat_idx, "task_id": task["id"], "task_name": task["name"],
        "returncode": proc.returncode,
        "task_passed": (passed == total and total > 0),
        "checks_passed": passed, "checks_total": total,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "checks": checks,
        "tool_calls": len(tool_events),
        "num_turns": int(result_event.get("num_turns", 0) or 0),
        "wall_s": round(wall_s, 2),
        "input_uncached_plus_create": in_tok + cc_tok,
        "cache_read_tokens": cr_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + cc_tok + cr_tok + out_tok,
        "cost_usd": round(float(result_event.get("total_cost_usd", 0.0) or 0.0), 6),
        "parsed_answer": parsed,
    }


def summarize_arm(rows: list[dict]) -> dict:
    tot_c = sum(r["checks_total"] for r in rows)
    pas_c = sum(r["checks_passed"] for r in rows)
    n = len(rows)
    return {
        "runs": n,
        "tasks_passed": sum(1 for r in rows if r["task_passed"]),
        "pass_rate": round(sum(1 for r in rows if r["task_passed"]) / n, 4) if n else 0.0,
        "checks_passed": pas_c,
        "checks_total": tot_c,
        "accuracy": round(pas_c / tot_c, 4) if tot_c else 0.0,
        "total_tool_calls": sum(r["tool_calls"] for r in rows),
        "total_turns": sum(r["num_turns"] for r in rows),
        "total_wall_s": round(sum(r["wall_s"] for r in rows), 2),
        "total_input_uncached_plus_create": sum(r["input_uncached_plus_create"] for r in rows),
        "total_cache_read_tokens": sum(r["cache_read_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
    }


def evaluate_gate(base: dict, cand: dict, *, min_gain_pct: float = 5.0, allow_tie: bool = False) -> dict:
    """Enforce zero regression and strict measurable improvement."""
    regressions = []
    if cand["pass_rate"] < base["pass_rate"]:
        regressions.append(f"pass_rate regressed: {cand['pass_rate']:.4f} < {base['pass_rate']:.4f}")
    if cand["accuracy"] < base["accuracy"]:
        regressions.append(f"accuracy regressed: {cand['accuracy']:.4f} < {base['accuracy']:.4f}")
    if base["total_tokens"] > 0 and cand["total_tokens"] > base["total_tokens"] * 1.05:
        regressions.append(f"total_tokens regressed >5%: {cand['total_tokens']} > {base['total_tokens']}")
    if base["total_tool_calls"] > 0 and cand["total_tool_calls"] > base["total_tool_calls"] * 1.05:
        regressions.append(f"total_tool_calls regressed >5%: {cand['total_tool_calls']} > {base['total_tool_calls']}")

    improvements = []
    if cand["pass_rate"] > base["pass_rate"]:
        improvements.append(f"pass_rate +{cand['pass_rate'] - base['pass_rate']:.4f}")
    if cand["accuracy"] > base["accuracy"]:
        improvements.append(f"accuracy +{cand['accuracy'] - base['accuracy']:.4f}")

    def _pct_drop(b_val: float, c_val: float) -> float:
        return ((b_val - c_val) / b_val * 100.0) if b_val > 0 else 0.0

    for label, key, thresh in (
        ("total_tokens", "total_tokens", min_gain_pct),
        ("tool_calls", "total_tool_calls", min_gain_pct),
        ("cost_usd", "total_cost_usd", min_gain_pct),
        ("wall_s", "total_wall_s", max(10.0, min_gain_pct)),
    ):
        drop = _pct_drop(base[key], cand[key])
        if drop >= thresh:
            improvements.append(f"{label} -{drop:.1f}%")

    if regressions:
        verdict, exit_code = "REGRESSION", 1
    elif improvements:
        verdict, exit_code = "IMPROVED", 0
    elif allow_tie:
        verdict, exit_code = "NO_REGRESSION_TIE", 0
    else:
        verdict, exit_code = "NO_IMPROVEMENT", 1

    return {
        "verdict": verdict,
        "exit_code": exit_code,
        "min_gain_pct": min_gain_pct,
        "regressions": regressions,
        "improvements": improvements,
    }


def select_tasks(suite: str, task_ids: list[str] | None) -> list[dict]:
    if task_ids:
        wanted = set(task_ids)
        chosen = [t for t in TASKS if t["id"] in wanted or t["id"].split("_", 1)[0] in wanted]
        if not chosen:
            raise ValueError(f"No matching tasks for {task_ids}")
        return chosen
    if suite == "core":
        return [t for t in TASKS if t["suite"] == "core"]
    return list(TASKS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full-feature MolGap agent infrastructure A/B benchmark & improvement gate.",
        epilog="Gate exits 0 only on IMPROVED (zero regression + measurable quality or efficiency gain).",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List all benchmark tasks and check counts, then exit")
    parser.add_argument("--suite", choices=("core", "full"), default="full", help="Task suite: core (T1-T6) or full (T1-T12)")
    parser.add_argument("--tasks", nargs="*", help="Optional task IDs or prefixes (e.g. T1 T7 T12)")
    parser.add_argument("--base", default="HEAD", help="Baseline git ref (default: HEAD)")
    parser.add_argument("--candidate", default="WORKTREE", help="Candidate git ref or WORKTREE for current dirty changes")
    parser.add_argument("--model", default="haiku", help="Claude CLI model alias/id (default: haiku)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of paired repeats per task (default: 1)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count (default: 4)")
    parser.add_argument("--max-turns", type=int, default=16, help="Max agent turns per task (default: 16)")
    parser.add_argument("--min-gain-pct", type=float, default=5.0, help="Minimum %% efficiency gain when accuracy ties (default: 5.0)")
    parser.add_argument("--allow-tie", action="store_true", help="Exit 0 on zero-regression tie without strict improvement")
    parser.add_argument("--out", type=Path, help="Output JSON report path (default: temp file)")
    parser.add_argument("--export-prompts", type=Path, help="Export task prompts JSON to this path for external/Swarm runners and exit")
    args = parser.parse_args(argv)

    selected = select_tasks(args.suite, args.tasks)
    if args.list_tasks:
        total_checks = sum(t["checks_count"] for t in selected)
        print(f"Suite={args.suite} | tasks={len(selected)} | total_checks={total_checks}")
        for t in selected:
            print(f"  {t['id']:38s} [{t['suite']:4s}] checks={t['checks_count']:2d} | {t['name']}")
        return 0

    if args.export_prompts:
        args.export_prompts.parent.mkdir(parents=True, exist_ok=True)
        args.export_prompts.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Exported {len(selected)} tasks to {args.export_prompts}")
        return 0

    claude_exe = find_claude_exe()
    tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_wt, cand_wt = setup_isolated_worktrees(args.base, args.candidate, tag)
    try:
        truth_base = collect_arm_truth(base_wt)
        truth_cand = collect_arm_truth(cand_wt)
        jobs = []
        for rep in range(1, args.repeats + 1):
            for task in selected:
                jobs.append(("base", base_wt, task, truth_base, rep))
                jobs.append(("cand", cand_wt, task, truth_cand, rep))

        print(f"[agent_infra_bench] model={args.model} suite={args.suite} tasks={len(selected)} "
              f"repeats={args.repeats} total_runs={len(jobs)} base={args.base} cand={args.candidate}", flush=True)

        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(run_claude_one, claude_exe, args.model, arm, wt, task, truth, rep, args.max_turns): (arm, task["id"], rep)
                for arm, wt, task, truth, rep in jobs
            }
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                print(
                    f"  [{row['arm']:4s} r{row['repeat']}] {row['task_id']:36s} "
                    f"pass={str(row['task_passed']):5s} acc={row['checks_passed']}/{row['checks_total']} "
                    f"tools={row['tool_calls']:2d} time={row['wall_s']:5.1f}s "
                    f"tok={row['total_tokens']:7d} cost=${row['cost_usd']:.4f}",
                    flush=True,
                )

        rows.sort(key=lambda r: (r["task_id"], r["repeat"], 0 if r["arm"] == "base" else 1))
        base_sum = summarize_arm([r for r in rows if r["arm"] == "base"])
        cand_sum = summarize_arm([r for r in rows if r["arm"] == "cand"])
        gate = evaluate_gate(base_sum, cand_sum, min_gain_pct=args.min_gain_pct, allow_tie=args.allow_tie)

        out_path = args.out or (Path(tempfile.gettempdir()) / f"molgap_agent_infra_bench_{tag}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model, "suite": args.suite, "repeats": args.repeats,
            "base": args.base, "candidate": args.candidate,
            "summary": {"base": base_sum, "cand": cand_sum},
            "gate": gate,
            "runs": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n=== GATE VERDICT ===")
        print(json.dumps({"summary": payload["summary"], "gate": gate, "report": str(out_path)}, indent=2, ensure_ascii=False))
        return int(gate["exit_code"])
    finally:
        cleanup_worktrees(base_wt, cand_wt)


if __name__ == "__main__":
    raise SystemExit(main())

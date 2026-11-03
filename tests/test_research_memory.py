from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from molgap.research_memory.backtest import build_screening_backtest
from molgap.research_memory.compiler import compile_research_memory
from molgap.research_memory.cost import build_cost_ledger
from molgap.research_memory.derived import EXPECTED_FORMATS, validate_derived_outputs
from molgap.research_memory.discovery import DiscoveredRecords
from molgap.research_memory.ready import build_ready_package
from molgap.research_memory.schemas import validate_id, validate_trajectory
from molgap.research_memory.summary import render_summary_markdown
from molgap.research_memory.validate import validate_repository_records


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v5_conversion_inventory_points_to_valid_trajectories():
    audit = json.loads(
        (REPO_ROOT / "research_memory/v5_trajectory_conversion_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["included_trajectory_count"] == len(audit["included"])
    for pointer in audit["included"]:
        path = REPO_ROOT / pointer
        assert path.is_file(), pointer
        validate_trajectory(json.loads(path.read_text(encoding="utf-8")))

    for excluded in audit["excluded_nonterminal"]:
        assert not (REPO_ROOT / excluded["experiment"] / "trajectory.json").exists()


def _cost_event(
    event_id: str,
    device_status: str,
    device_value: float | None,
    *,
    cpu_status: str = "measurement_missing",
    cpu_value: float | None = None,
) -> dict:
    return {
        "cost_event_id": event_id,
        "trajectory_id": "TB-cost",
        "category": "training",
        "hardware": "T4",
        "measurement": {
            "device_hours": {"value": device_value, "status": device_status},
            "cpu_hours": {"value": cpu_value, "status": cpu_status},
            "wall_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "measurement_missing"},
        },
    }


def _ready_fixture() -> tuple[dict, dict, dict]:
    candidate_id = "ev-candidate"
    reference_id = "ev-reference"
    trajectory = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": "TB-ready-test",
        "record_mode": "prospective",
        "track": "B",
        "owner": "server",
        "family_id": "family-ready",
        "question": "Should the qualified candidate be transferred?",
        "hypothesis": {
            "hypothesis_id": "hyp-ready",
            "supporting_evidence_ids": [candidate_id],
            "alternative_explanations": ["run noise"],
            "related_closed_family_ids": ["closed-family"],
            "observed_deficiency": "The reference remains above the target MAE.",
            "changed_mechanism": "One frozen mechanism.",
            "cheapest_falsifier": "The frozen paired comparison.",
            "expected_native_cost_ref": "cost-ready",
            "decision_changed_if_positive": "Transfer.",
            "decision_changed_if_negative": "Close.",
        },
        "state_at_start": {
            "source_commit": "a" * 40,
            "source_config_identity": "source-config-v1",
            "contract_refs": ["contract.json"],
            "reference_ids": [reference_id],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": [reference_id],
            "role_snapshot_refs": ["roles.json"],
            "budget_snapshot_ref": "budget.json",
        },
        "actions": [
            {
                "action_id": "action-ready",
                "type": "TRAIN",
                "source_commit": "a" * 40,
                "run_ids": ["run-1"],
                "attempt_ids": ["attempt-1"],
                "evidence_refs": ["comparison.json"],
                "cost_event_ids": ["cost-ready"],
            }
        ],
        "result": {
            "evidence_ids": [candidate_id],
            "evidence_refs": ["comparison.json"],
        },
        "decision": {
            "outcome": "READY_FOR_DESKTOP",
            "final": True,
            "decision_ref": "decision.md",
            "next_allowed_actions": ["TRANSFER"],
            "reopen_conditions": [],
        },
        "readiness": {
            "candidate_evidence_id": candidate_id,
            "qualification_100k_evidence_ids": [candidate_id],
            "qualification_500k_evidence_ids": [candidate_id],
            "funnel_requires_500k": True,
            "compatible_reference_id": reference_id,
            "candidate_contract_identity": "contract-v5",
            "reference_contract_identity": "contract-v5",
            "source_config_identity": "source-config-v1",
            "paired_comparison_ref": "comparison.json",
            "role_history_refs": ["roles.json"],
            "native_cost_event_ids": ["cost-ready"],
            "known_limitations": ["single seed"],
            "full_scale_question": "Does this exact candidate retain its gain at full scale?",
        },
    }
    evidence = {
        candidate_id: {
            "evidence_id": candidate_id,
            "legacy_contract": "contract-v5",
            "outcome": {
                "transfer_status": "qualified",
                "comparison_status": "complete",
            },
            "artifacts": [
                {
                    "name": "checkpoint",
                    "locator": "external://checkpoint",
                    "sha256": "b" * 64,
                    "availability": "durable_remote_verified",
                }
            ],
        },
        reference_id: {
            "evidence_id": reference_id,
            "legacy_contract": "contract-v5",
            "outcome": {},
            "artifacts": [],
        },
    }
    costs = {
        "cost-ready": {
            "cost_event_id": "cost-ready",
            "measurement": {
                "device_hours": {"value": 2.0, "status": "measured"},
                "cpu_hours": {"value": None, "status": "not_applicable"},
            },
        }
    }
    return trajectory, evidence, costs


def test_dangling_evidence_is_rejected_with_source_and_field(tmp_path, monkeypatch):
    pointer = tmp_path / "pointer.txt"
    pointer.write_text("fixture", encoding="utf-8")
    evidence_path = tmp_path / "v5_evidence.json"
    evidence_path.write_text(json.dumps({"evidence_id": "ev-known"}), encoding="utf-8")
    trajectory_path = tmp_path / "trajectory.json"
    trajectory = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": "TB-dangling",
        "record_mode": "retrospective_partial",
        "track": "B",
        "owner": "historical",
        "family_id": "family",
        "question": "Historical record.",
        "hypothesis": {
            "hypothesis_id": "hyp-dangling",
            "supporting_evidence_ids": [],
            "alternative_explanations": [],
            "related_closed_family_ids": [],
        },
        "state_at_start": {
            "source_commit": "unknown_retrospective",
            "contract_refs": [],
            "reference_ids": [],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": [],
            "role_snapshot_refs": [],
        },
        "actions": [],
        "result": {
            "evidence_ids": ["ev-missing"],
            "evidence_refs": [],
        },
        "decision": {
            "outcome": "CLOSED",
            "decision_ref": pointer.name,
            "next_allowed_actions": [],
            "reopen_conditions": [],
        },
    }
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    discovered = DiscoveredRecords(
        evidence=(evidence_path,),
        trajectories=(trajectory_path,),
        costs=(),
        roles=(),
        traces=(),
        ready=(),
    )
    monkeypatch.setattr(
        "molgap.research_memory.validate.discover_records", lambda _: discovered
    )
    monkeypatch.setattr(
        "molgap.research_memory.validate.validate_v5_evidence_envelope",
        lambda record, repo_root: record,
    )
    with pytest.raises(ValueError, match=r"trajectory\.json:result\.evidence_ids.*ev-missing"):
        validate_repository_records(tmp_path)


def test_ready_rejects_retrospective():
    trajectory, evidence, costs = _ready_fixture()
    trajectory["record_mode"] = "retrospective_partial"
    with pytest.raises(ValueError, match="retrospective"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_missing_reference():
    trajectory, evidence, costs = _ready_fixture()
    evidence.pop("ev-reference")
    with pytest.raises(ValueError, match="dangling evidence"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_empty_role_history():
    trajectory, evidence, costs = _ready_fixture()
    trajectory["readiness"]["role_history_refs"] = []
    with pytest.raises(ValueError, match="role history"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_missing_artifact_sha():
    trajectory, evidence, costs = _ready_fixture()
    evidence["ev-candidate"]["artifacts"][0]["sha256"] = None
    with pytest.raises(ValueError, match="lacks SHA256"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_unknown_native_cost():
    trajectory, evidence, costs = _ready_fixture()
    costs["cost-ready"]["measurement"]["device_hours"] = {
        "value": None,
        "status": "measurement_missing",
    }
    with pytest.raises(ValueError, match="measured native"):
        build_ready_package(trajectory, evidence, costs)


def test_cost_ledger_separates_measured_estimated_missing_and_not_applicable():
    ledger = build_cost_ledger(
        [
            _cost_event(
                "cost-m",
                "measured",
                2.0,
                cpu_status="not_applicable",
            ),
            _cost_event(
                "cost-e",
                "estimated",
                3.0,
                cpu_status="not_applicable",
            ),
        ]
    )
    device = ledger["totals_by_hardware"]["T4"]["device_hours"]
    assert device["measured"] == {"known_total": 2.0, "known_records": 1}
    assert device["estimated"] == {"known_total": 3.0, "known_records": 1}
    assert "total" not in device
    assert ledger["totals_by_hardware"]["T4"]["cpu_hours"]["not_applicable_records"] == 2
    assert ledger["totals_by_hardware"]["T4"]["wall_hours"]["unknown_records"] == 2
    assert ledger["cross_hardware_total"] is None


def test_repository_cost_completeness_is_internally_consistent():
    payloads = compile_research_memory(REPO_ROOT)
    report = json.loads(payloads["completeness_report.json"])
    cost = report["cost_completeness"]
    assert cost["trajectories_total"] >= 1
    assert cost["with_cost_event"] + cost["without_cost_event"] == cost["trajectories_total"]
    assert (
        cost["with_complete_native_measurement"]
        + cost["with_incomplete_native_measurement"]
        == cost["with_cost_event"]
    )
    issues = {item["code"]: item["count"] for item in report["issues"]}
    assert issues.get("missing_cost_event", 0) == cost["without_cost_event"]
    assert issues.get("incomplete_native_cost_measurement", 0) == cost[
        "with_incomplete_native_measurement"
    ]
    assert "missing_native_cost" not in issues


def test_empty_numeric_basis_serializes_as_null():
    ledger = build_cost_ledger(
        [_cost_event("cost-missing", "measurement_missing", None)]
    )
    measured = ledger["totals_by_hardware"]["T4"]["device_hours"]["measured"]
    assert measured == {"known_total": None, "known_records": 0}


def test_measured_zero_remains_real_zero():
    ledger = build_cost_ledger([_cost_event("cost-zero", "measured", 0.0)])
    measured = ledger["totals_by_hardware"]["T4"]["device_hours"]["measured"]
    assert measured == {"known_total": 0.0, "known_records": 1}


def test_not_applicable_only_is_not_zero_or_unknown():
    ledger = build_cost_ledger(
        [
            _cost_event(
                "cost-na",
                "not_applicable",
                None,
                cpu_status="not_applicable",
            )
        ]
    )
    device = ledger["totals_by_hardware"]["T4"]["device_hours"]
    assert device["measured"]["known_total"] is None
    assert device["estimated"]["known_total"] is None
    assert device["unknown_records"] == 0
    assert device["not_applicable_records"] == 1


def test_estimated_only_does_not_create_measured_zero():
    ledger = build_cost_ledger([_cost_event("cost-est", "estimated", 1.25)])
    device = ledger["totals_by_hardware"]["T4"]["device_hours"]
    assert device["measured"] == {"known_total": None, "known_records": 0}
    assert device["estimated"] == {"known_total": 1.25, "known_records": 1}


def _summary_with_bucket(bucket: dict) -> dict:
    return {
        "evidence": {"validated": 0, "without_trajectory": 0},
        "trajectories": {"outcomes": {}},
        "transfer": {"ready_valid": 0, "ready_blocked": 0, "missing_references": 0},
        "roles": {"explicit_events": 0, "coarse_historical": 0, "repeatedly_selected": 0},
        "cost": {
            "totals_by_hardware": {"T4": {"device_hours": bucket}},
            "unknown_events": 1,
            "completeness": {
                "trajectories_total": 1,
                "with_cost_event": 1,
                "without_cost_event": 0,
                "with_complete_native_measurement": 0,
                "with_incomplete_native_measurement": 1,
            },
        },
        "backtest": {"status": "insufficient_evidence", "eligible": 0, "excluded": 0},
        "memory_gaps": [],
    }


def test_markdown_renders_unknown_without_false_zero():
    bucket = build_cost_ledger(
        [_cost_event("cost-missing", "measurement_missing", None)]
    )["totals_by_hardware"]["T4"]["device_hours"]
    markdown = render_summary_markdown(_summary_with_bucket(bucket))
    assert "measured=unknown (0 measured records)" in markdown
    assert "0.000000 (0 records)" not in markdown


def test_markdown_renders_real_measured_zero():
    bucket = build_cost_ledger(
        [_cost_event("cost-zero", "measured", 0.0)]
    )["totals_by_hardware"]["T4"]["device_hours"]
    markdown = render_summary_markdown(_summary_with_bucket(bucket))
    assert "measured=0.000000 (1 measured record)" in markdown


def test_all_derived_json_schemas_match_generated_top_level_contracts():
    payloads = compile_research_memory(REPO_ROOT)
    for output_name in EXPECTED_FORMATS:
        output = json.loads(payloads[output_name])
        schema_name = output_name.replace("_", "-").replace(
            ".json", "-v1.schema.json"
        )
        schema = json.loads(
            (REPO_ROOT / "research_memory/schemas" / schema_name).read_text(
                encoding="utf-8"
            )
        )
        assert set(schema["required"]) == set(output), output_name
        assert set(schema["properties"]) == set(output), output_name
        assert schema["additionalProperties"] is False

    ledger_schema = json.loads(
        (
            REPO_ROOT / "research_memory/schemas/cost-ledger-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert "estimated_event_count" in ledger_schema["required"]
    assert set(ledger_schema["$defs"]["basis"]["properties"]["known_total"]["type"]) == {
        "number",
        "null",
    }


def test_prospective_missing_hypothesis_field_is_rejected():
    trajectory, _, _ = _ready_fixture()
    del trajectory["hypothesis"]["cheapest_falsifier"]
    with pytest.raises(ValueError, match="cheapest_falsifier"):
        validate_trajectory(trajectory)


@pytest.mark.parametrize("value", ["bad id", "../x", "a/b", "a\\b", "x\ny", " C:"])
def test_malformed_ids_are_rejected(value):
    with pytest.raises(ValueError):
        validate_id(value, "fixture.id")


def _trace(run_id, role, optimizer):
    return {
        "trajectory_id": f"TB-{run_id}",
        "run_id": run_id,
        "reference_id": "ev-reference",
        "comparison_role": role,
        "x_axis": "optimizer_steps",
        "backtest_eligibility": {"eligible": True, "exclusion_reasons": []},
        "exposure": {"optimizer_steps": 100, "sample_presentations": 12800},
        "comparability_identity": {
            "scientific_contract": "v5",
            "dataset_identity": "100k",
            "row_split_identity": "rows-v1",
            "architecture_identity": "arch",
            "optimizer_identity": optimizer,
            "lr_schedule_identity": "cosine",
            "target_transform_identity": "direct-gap",
            "precision_identity": "fp32",
            "ema_semantics": "none",
            "evaluation_role_identity": "development",
            "selection_role_identity": "development",
            "x_axis_semantics": "optimizer_steps",
            "terminal_endpoint_identity": "step-100",
            "matched_architecture_required": False,
        },
    }


def test_incompatible_traces_remain_insufficient():
    result = build_screening_backtest(
        [_trace("candidate", "candidate", "adamw"), _trace("reference", "reference", "sgd")]
    )
    assert result["status"] == "insufficient_evidence"
    assert result["recommended_early_stop"] is None
    assert result["false_stop_count"] is None
    assert result["native_cost_saved"] is None
    assert result["policy_activated"] is False


def test_invalid_derived_output_is_rejected():
    payloads = compile_research_memory(REPO_ROOT)
    outputs = {
        name: json.loads(payload)
        for name, payload in payloads.items()
        if name.endswith(".json")
    }
    outputs["cost_ledger.json"]["cross_hardware_total"] = 0
    with pytest.raises(ValueError, match="must not aggregate across hardware"):
        validate_derived_outputs(outputs)


def test_cards_filter_by_outcome_and_term(tmp_path):
    from molgap.research_memory.cards import _keywords, select

    assert _keywords("training mean of all two atom pair norm", "pcqm-k1-sam-100k") == ["norm", "pair", "sam"]
    cards = [
        {"outcome": "NEGATIVE_UNDER_CONTRACT", "mechanism": "sharpness-aware step", "keywords": ["sharpness"]},
        {"outcome": "POSITIVE_UNDER_CONTRACT", "mechanism": "pair norm", "keywords": ["pair"]},
    ]
    assert select(cards, outcome="NEGATIVE_UNDER_CONTRACT", term="SHARPNESS") == cards[:1]
    assert select(cards, term="sharp") == []  # whole words only: "SAM" must not match "same"/"sample"
    assert select(cards, term="PAIR") == cards[1:]
    assert select(cards, term="pair NORM") == select(cards, term=["pair", "norm"]) == cards[1:]  # every word
    assert select(cards, term="pair sharpness") == [] and select(cards, term="pair nor") == []
    assert select(cards) == cards


def test_card_effect_is_own_run_against_closed_reference(tmp_path, capsys):
    from molgap.research_memory.cards import build_cards

    def put(rel, value):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(json.dumps(value), encoding="utf-8")
        return rel

    rows = []
    # Two runs share the arm name "pretrained"; run "lever" has two lever arms (reference closes NEGATIVE).
    for run, delta, ref_outcome in (("a", 0.001, "CLOSED"), ("b", 0.002, "CLOSED"), ("lever", -0.0008, "NEGATIVE_UNDER_CONTRACT")):
        pa = put(f"experiments/x/{run}/pretrained/paired_analysis.json",
                 {"mean_delta_eV": delta, "lower_95_eV": delta - 1e-4, "upper_95_eV": delta + 1e-4})
        ready = put(f"experiments/x/{run}/ready.json", {"reference_id": f"ev-{run}-ref",
                    "candidate_artifact_bindings": {"paired_analysis": {"ref": pa}}})
        for arm, outcome, record in (("ref", ref_outcome, {}), ("pretrained", "NEGATIVE_UNDER_CONTRACT",
                                                             {"comparison_readiness_ref": ready})):
            rows.append({"trajectory_id": f"T-{run}-{arm}", "family_id": "f", "outcome": outcome,
                         "result_evidence_ids": [f"ev-{run}-{arm}"],
                         "record_path": put(f"experiments/x/{run}/{arm}.json", record)})
    put("research_memory/derived/trajectory_index.json", {"trajectories": rows})
    effects = {c["trajectory_id"]: c["effect_meV"] for c in build_cards(tmp_path)}
    assert effects == {"T-a-ref": None, "T-a-pretrained": pytest.approx(1.0), "T-b-ref": None,
                       "T-b-pretrained": pytest.approx(2.0), "T-lever-ref": None, "T-lever-pretrained": None}

    # --ref: another revision's committed cards join the local ones, each naming its source; local wins an id.
    from molgap.research_memory.cli import main

    def cards(*extra):
        main(["--repo-root", str(tmp_path), "cards", *extra])
        return [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    before = cards()
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    other = [rows[0], {**rows[3], "outcome": "INCONCLUSIVE"}, {**rows[1], "trajectory_id": "T-other"}]
    put("research_memory/derived/trajectory_index.json", {"trajectories": other})
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "other"], ["tag", "other"]):
        subprocess.run(git + args, check=True)
    put("research_memory/derived/trajectory_index.json", {"trajectories": rows})
    assert cards() == before and all("source" not in c for c in before)
    merged = {c["trajectory_id"]: c for c in cards("--ref", "other")}
    assert len(merged) == len(rows) + 1 and merged["T-b-pretrained"]["outcome"] == "NEGATIVE_UNDER_CONTRACT"
    assert merged["T-b-pretrained"]["source"]["ref"] == "local"
    assert merged["T-other"]["source"]["ref"] == "other" and merged["T-other"]["source"]["last_commit"]
    assert merged["T-other"]["source"]["record_path"] == rows[1]["record_path"]
    assert merged["T-other"]["effect_meV"] == pytest.approx(1.0)  # same projection, read from git objects
    with pytest.raises(SystemExit) as failed:
        cards("--ref", "missing")
    assert failed.value.code not in (0, None)


def test_lookup_and_gaps_are_read_only_views():
    from molgap.research_memory.cards import _search_terms, gaps, lookup

    def card(fam, outcome, effect=None):
        return {"trajectory_id": fam, "family_id": fam, "outcome": outcome, "mechanism": "norm", "keywords": ["norm"],
                "failure_mode": None,
                "falsifier": "f", "reopen_conditions": None, "effect_meV": effect, "search_terms": []}

    cards = [card("a-100k", "NEGATIVE_UNDER_CONTRACT", 1.0), card("b-100k", "NEGATIVE_UNDER_CONTRACT"),
             card("b-500k", "POSITIVE_UNDER_CONTRACT")]
    found = lookup(cards, mechanism="norm")
    assert [u["family_id"] for u in found["unfalsified"]] == ["b-100k", "b-500k"]
    assert lookup(cards, family="a-")["matches"] == cards[:1]
    assert lookup(cards, mechanism="B-5")["matches"] == cards[2:]  # also the family id, case-insensitive
    assert lookup(cards, mechanism="NOR")["matches"] == cards  # case-insensitive substring
    cards[0]["failure_mode"] = "below_gate"
    assert lookup(cards, mechanism="below")["matches"] == cards[:1]  # and the failure_mode
    assert lookup(cards, mechanism="ensemble")["matches"] == []
    assert _search_terms({"mechanism": "lower-peak-lr", "family_id": "pcqm-k1-sam-100k", "failure_mode": "below_gate",
                          "search_terms": ["sharpness"]}) == ["sharpness", "below_gate", "lower", "peak", "sam"]
    many = "-".join(f"w{c}{c}" for c in "abcdefghi")
    assert _search_terms({"mechanism": "Free prose norm.", "family_id": many, "failure_mode": None,
                          "search_terms": []}) == many.split("-")[:8]  # prose is not tokenized; capped at 8
    assert lookup(cards, term="1.0", mechanism="norm")["matches"] == cards[:1]
    assert lookup(cards, mechanism="nor B-")["matches"] == cards[1:]  # several substrings, all required
    assert lookup(cards, mechanism="norm ensemble")["matches"] == []
    flagged = gaps(cards)["contradictions"]  # same arm "b" across scales; "a" has no 500K partner
    assert [(f["key"], f["a"]["scale_k"], f["b"]["scale_k"]) for f in flagged] == [("b", 100, 500)]
    assert flagged[0]["kind"] == "outcome_differs_across_scale_needs_review"
    from molgap.research_memory.cards import compact_line, levers
    cards[1]["decision_ref"] = "experiments/b/terminal_decision.md"
    assert compact_line(cards[0]) == "a-100k | NEGATIVE_UNDER_CONTRACT | a-100k | norm | +1.00 | below_gate"
    assert compact_line(cards[1]).endswith("| null | -")
    assert levers(cards, "b-100k")[1:] == [
        "b-100k | b-100k | NEGATIVE_UNDER_CONTRACT | null (see experiments/b/terminal_decision.md) | - | norm"]
    assert [row.split(" | ")[0] for row in levers(cards, "*-100k")[1:]] == ["a-100k", "b-100k"]
    from molgap.research_memory.cards import gaps_compact
    assert gaps_compact(gaps(cards)) == [
        "sparse | a-100k | n=1 with_effect=1", "sparse | b-100k | n=1 with_effect=0",
        "sparse | b-500k | n=1 with_effect=0",
        "contradiction | b | b-100k 100k NEGATIVE_UNDER_CONTRACT null vs b-500k 500k POSITIVE_UNDER_CONTRACT null"]


def test_erratum_scope_comes_from_the_bound_file_and_its_experiment(tmp_path):
    from molgap.research_memory.validate import _load_errata, _verify_with_errata

    home = tmp_path / "experiments" / "exp_a"
    (home / "errata").mkdir(parents=True)
    real = "experiments/exp_a/terminal_decision.md"
    (tmp_path / real).write_bytes(b"observed\n")
    recorded, observed = "1" * 64, hashlib.sha256(b"observed\n").hexdigest()
    base = {"format": "molgap-erratum-v1", "artifact_class": "narrative", "effect": "narrative_only",
            "target_locator": real, "recorded_sha256": recorded, "observed_sha256": observed}
    (home / "errata" / "e.json").write_text(json.dumps(base), encoding="utf-8")
    errata = _load_errata(tmp_path)
    assert list(errata) == [(real, recorded)]
    gaps = []
    _verify_with_errata(tmp_path, errata, gaps, real, recorded)
    assert gaps == [real]
    for pointer, digest, pair in [(real, "2" * 64, base),  # recorded digest not in the erratum
                                  (real, recorded, {**base, "observed_sha256": "3" * 64})]:  # observed differs
        with pytest.raises(ValueError, match="SHA mismatch"):
            _verify_with_errata(tmp_path, {(real, recorded): pair}, [], pointer, digest)
    (home / "errata" / "e.json").unlink()
    run = "experiments/exp_a/runs/x/"
    for where, change in [("exp_a", {"target_locator": run + "paired_analysis.json"}),  # result, not .md
                          ("exp_a", {"target_locator": run + "cost_records.json"}),  # contract, not .md
                          ("exp_b", {}),  # erratum filed in another experiment's dir
                          ("exp_a", {"target_locator": real.replace("/", "\\")}),
                          ("exp_a", {"artifact_class": "result"}),
                          ("exp_a", {"effect": "replay"})]:
        errata_dir = tmp_path / "experiments" / where / "errata"
        errata_dir.mkdir(parents=True, exist_ok=True)
        (errata_dir / "e.json").write_text(json.dumps({**base, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match="erratum may only excuse"):
            _load_errata(tmp_path)
        (errata_dir / "e.json").unlink()


def test_crlf_tolerance_needs_head_roles_normalize_and_closure_numbers_are_finite(tmp_path):
    from molgap.research_memory.paths import verify_bound_artifact
    from molgap.research_memory.saved_analysis import run
    from molgap.research_memory.schemas import validate_closure_facts

    (tmp_path / "note.md").write_bytes(b"a\r\nb\r\n")
    lf_sha = hashlib.sha256(b"a\nb\n").hexdigest()
    for args in (["init", "-q"], ["add", "note.md"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match="SHA mismatch"):  # staged only, not in HEAD
        verify_bound_artifact(tmp_path, "note.md", lf_sha)
    subprocess.run(["git", "-c", "user.email=t@e.invalid", "-c", "user.name=t", "commit", "-qm", "x"], cwd=tmp_path, check=True)
    verify_bound_artifact(tmp_path, "note.md", lf_sha)
    for role in ("Official_Validation", "official_validation ", "test-dev", "test dev", " Test \t Challenge"):
        with pytest.raises(ValueError, match="protected"):
            run("paired", [], role_id=role)
    facts = {"format": "molgap-closure-facts-v1"}
    for bad in ({"effect_meV": float("nan")}, {"effect_meV": float("inf")}, {"ci_meV": [0.0, float("-inf")]}):
        with pytest.raises(ValueError, match="closure facts"):
            validate_closure_facts({**facts, **bad})
    assert validate_closure_facts({**facts, "effect_meV": -1.5, "ci_meV": [-2, 0.5]})["effect_meV"] == -1.5


def test_research_memory_analyze_ensemble_and_refusals(tmp_path):
    torch = pytest.importorskip("torch")
    from molgap.research_memory.saved_analysis import run, write_output
    ids, target = torch.arange(100, 108), torch.linspace(-1, 1, 8)
    paths = []
    for i, offset in enumerate((.1, -.1, .1)):
        path = tmp_path / f"p{i}.pt"
        torch.save({"source_idx": ids, "target_eV": target, "prediction_eV": target + offset}, path)
        paths.append(path)
    result = run("ensemble-curve", paths, role_id="internal_development")
    assert result["posthoc"] is True and result["row_count"] == 8
    assert result["results"]["mae_by_n_members_meV"]["1"] == pytest.approx(100.)
    assert result["results"]["mae_by_n_members_meV"]["3"] == pytest.approx(100. / 3)
    named = {f"m{i}": path for i, path in enumerate(paths)}
    summary = run("summary", named, role_id="internal_development", pairs=[("m1", "m0")])["results"]
    assert summary["D_meV"] == pytest.approx(0.) and summary["ensemble_gain_meV"] == pytest.approx(200. / 3)
    assert summary["fit"] is not None and summary["m_meV"] == pytest.approx(0.)
    assert run("floor", named, role_id="internal_development")["results"]["members_used"] == 2
    with pytest.raises(ValueError, match="protected"):
        run("paired", paths, role_id="test_dev")
    torch.save({"source_idx": ids + 1, "target_eV": target, "prediction_eV": target}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="differ"):
        run("paired", [paths[0], tmp_path / "x.pt"], role_id="internal_development")
    with pytest.raises(ValueError, match="scratch"):
        write_output(result, tmp_path / "research_memory" / "derived" / "o.json", tmp_path)

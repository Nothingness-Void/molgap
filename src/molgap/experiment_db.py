"""Build and query the normalized MolGap model experiment database."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TARGETS = ("homo", "lumo", "gap", "average")

MODEL_NAME_TO_ID = {
    "routed_v4_500k": "M01",
    "fusion_1m": "M02",
    "repair_v2_1m_2d": "M04",
    "additive_1p5m_2d": "M05",
    "broad_1p098m_2d": "M06",
    "ensemble_two_1m_2d": "M07",
    "coverage_2m_2d": "M08",
    "ensemble_three_2m_2d": "M09",
    "distilled_2m_2d": "M10",
    "retention_b_2m_gps7": "M21",
    "retention_d_2m_gps7_seed42": "M25",
    "uniform_exact2m_gps7": "M26",
}

MODEL_ENRICHMENT = {
    "M00": (2.0, "component_only", "Reusable only through its registered loader."),
    "M01": (3.0, "production", "Frozen production baseline; fine-tune a copy only."),
    "M02": (3.0, "head_or_full", "Reusable 1M reference; preserve its ETKDG alignment."),
    "M04": (2.0, "expert_or_warm_start", "Useful as a complementary repair-data expert."),
    "M05": (2.0, "specialist_only", "Reuse only for OOD-focused controlled experiments."),
    "M06": (2.0, "specialist_only", "Reuse only on matched broad-residual chemistry."),
    "M07": (4.0, "teacher", "Good distillation teacher; too costly as the default."),
    "M08": (2.0, "ood_expert", "Useful as an OOD/coverage teacher or warm start."),
    "M09": (6.0, "teacher_upper_bound", "Use as an oracle/teacher, not direct deployment."),
    "M10": (1.0, "specialist_only", "Single-pass checkpoint, but weak general retention."),
    "M11": (4.0, "closed", "Do not reuse without matched 2D/3D data and split."),
    "M12": (3.0, "closed", "Representations are not drop-in compatible with 500K fusion."),
    "M13": (3.0, "always_dual_reference", "Reuse only with a newly calibrated route."),
    "M14": (None, "labels_and_oracle", "Reuse labels and oracle analysis, not the Router."),
    "M15": (None, "bounded_baseline", "Reuse as a cheap post-hoc control."),
    "M16": (3.0, "pilot_only", "Architecture evidence only; do not warm-start production."),
    "M17": (2.0, "pilot_only", "Static dual-2D control only."),
    "M18": (None, "label_audit", "Reuse the physical-consistency audit and loss code."),
    "M19": (None, "mechanism_audit", "Reuse the negative feature-coverage evidence."),
    "M20": (None, "archive_only", "Reproduction only."),
    "M21": (1.0, "general_warm_start", "Strong general warm start; PCQM needs a separate branch."),
    "M22": (1.0, "code_and_cache", "No quality result; reuse implementation and target cache."),
    "M23": (1.0, "specialist_candidate", "Gap-only specialist; never replace HOMO/LUMO branch."),
    "M24": (6.8, "optional_inference", "Reusable for flagged high-value rows only."),
    "M25": (1.0, "general_candidate", "Fine-tune/replay compatible after the three-seed gate."),
    "M26": (1.0, "control_only", "Valid warm-start control; external ranking is incomplete."),
}

CAUSES = {
    "F1": ("targeted_data_dilution", "The targeted 500K gradient share fell as data grew."),
    "F2": ("missing_replay", "Planned old:new replay was absent in completed scale-up runs."),
    "F3": ("nonstationary_validation", "Random validation followed a changing data mixture."),
    "F4": ("opposing_specialists", "Specialists improved different domains in opposite directions."),
    "F5": ("route_nontransfer", "The old route encoded a representation-specific relationship."),
    "F6": ("2d_3d_misalignment", "2D and 3D encoders were trained on different mixtures."),
    "F7": ("distillation_target_mismatch", "Teacher matching did not preserve external retention."),
    "F8": ("inference_cost", "Accuracy-positive ensembles exceeded the pass budget."),
    "F9": ("overcoupled_objective", "General B3LYP and PCQM Gap need separate contracts."),
}

MODEL_CAUSES = {
    "M02": ("F1", "F3", "F5"),
    "M03": ("F1", "F5"),
    "M04": ("F1", "F3", "F4"),
    "M05": ("F1", "F3", "F4"),
    "M06": ("F4",),
    "M07": ("F8", "F9"),
    "M08": ("F1", "F3", "F4"),
    "M09": ("F4", "F8", "F9"),
    "M10": ("F7",),
    "M11": ("F6",),
    "M12": ("F5", "F6"),
    "M13": ("F5",),
    "M14": ("F5",),
    "M16": ("F4",),
    "M17": ("F4",),
    "M21": ("F2", "F9"),
    "M25": ("F1", "F2", "F9"),
}


@dataclass(frozen=True)
class BuildPaths:
    repo: Path
    inventory_csv: Path
    unified_common_csv: Path
    unified_pcqm_csv: Path
    retention_b_common_json: Path
    retention_b_pcqm_json: Path
    retention_d_json: Path
    uniform_a_internal_json: Path
    retention_d_internal_json: Path

    @classmethod
    def from_repo(cls, repo: Path) -> "BuildPaths":
        phase8 = repo / "results" / "phase8"
        return cls(
            repo=repo,
            inventory_csv=phase8 / "model_inventory_audit" / "model_inventory.csv",
            unified_common_csv=phase8 / "scaleup_full_analysis" / "unified_common_metrics.csv",
            unified_pcqm_csv=phase8 / "scaleup_full_analysis" / "unified_pcqm_metrics.csv",
            retention_b_common_json=phase8
            / "retention_2m_external_eval"
            / "replay"
            / "common_metrics.json",
            retention_b_pcqm_json=phase8
            / "retention_2m_external_eval"
            / "replay"
            / "pcqm_metrics.json",
            retention_d_json=phase8
            / "repaired_2m"
            / "retention_d_seed42_comparison.json",
            uniform_a_internal_json=phase8
            / "multi2d_2m_scnet"
            / "gps7"
            / "metrics.json",
            retention_d_internal_json=phase8
            / "repaired_2m"
            / "retention_d_seed42_raw"
            / "train_metrics.json",
        )


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    training_data TEXT,
    architecture TEXT,
    status TEXT,
    primary_reason TEXT,
    reusable_value TEXT,
    approximate_encoder_passes REAL,
    reuse_mode TEXT,
    reuse_constraints TEXT
);
CREATE TABLE protocols (
    protocol_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    description TEXT NOT NULL,
    comparable_across_models INTEGER NOT NULL
);
CREATE TABLE evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    protocol_id TEXT NOT NULL REFERENCES protocols(protocol_id),
    scope TEXT NOT NULL,
    target TEXT NOT NULL,
    n INTEGER,
    mae_ev REAL,
    r2 REAL,
    baseline_model_id TEXT REFERENCES models(model_id),
    delta_mae_ev REAL,
    ci95_low_ev REAL,
    ci95_high_ev REAL,
    source_path TEXT NOT NULL,
    UNIQUE(model_id, protocol_id, scope, target)
);
CREATE TABLE artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT REFERENCES models(model_id),
    path TEXT NOT NULL UNIQUE,
    bytes INTEGER NOT NULL,
    sha256 TEXT,
    artifact_kind TEXT NOT NULL
);
CREATE TABLE causes (
    cause_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE model_causes (
    model_id TEXT NOT NULL REFERENCES models(model_id),
    cause_id TEXT NOT NULL REFERENCES causes(cause_id),
    PRIMARY KEY(model_id, cause_id)
);
CREATE VIEW comparable_external AS
SELECT e.*, m.family, m.status, m.approximate_encoder_passes
FROM evaluations e JOIN models m USING(model_id)
WHERE e.protocol_id = 'fixed_external_1977_v1';
"""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _relative(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_model(
    connection: sqlite3.Connection,
    row: dict[str, str],
) -> None:
    model_id = row["id"]
    passes, reuse_mode, constraints = MODEL_ENRICHMENT.get(
        model_id, (None, "unclassified", "Review before reuse.")
    )
    connection.execute(
        """
        INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            row["family"],
            row["data"],
            row["architecture"],
            row["status"],
            row["primary_reason"],
            row["reusable_value"],
            passes,
            reuse_mode,
            constraints,
        ),
    )


def _insert_evaluation(
    connection: sqlite3.Connection,
    *,
    model_id: str,
    protocol_id: str,
    scope: str,
    target: str,
    n: int | None,
    mae_ev: float | None,
    source_path: str,
    r2: float | None = None,
    baseline_model_id: str | None = None,
    delta_mae_ev: float | None = None,
    ci95_low_ev: float | None = None,
    ci95_high_ev: float | None = None,
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO evaluations (
            model_id, protocol_id, scope, target, n, mae_ev, r2,
            baseline_model_id, delta_mae_ev, ci95_low_ev, ci95_high_ev,
            source_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            protocol_id,
            scope,
            target,
            n,
            mae_ev,
            r2,
            baseline_model_id,
            delta_mae_ev,
            ci95_low_ev,
            ci95_high_ev,
            source_path,
        ),
    )


def _load_inventory(connection: sqlite3.Connection, paths: BuildPaths) -> None:
    with paths.inventory_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            _add_model(connection, row)
    for model_id, family, data, architecture, status, reason, reuse in (
        (
            "M25",
            "Retention-D repaired-2M seed42",
            "Repaired 2M with 50 percent targeted replay",
            "Single GPS7",
            "candidate_pending_seed_repeats",
            "Seed42 improves common OOD and P8-hard; PCQM remains separate",
            "Best one-pass general candidate pending seeds 43 and 44",
        ),
        (
            "M26",
            "Uniform exact-2M GPS7 control A",
            "Exact mixed 2M with uniform sampling",
            "Single GPS7",
            "control",
            "Internal split exists; fixed external single-GPS evaluation is absent",
            "Controlled scale-up and warm-start reference",
        ),
    ):
        if connection.execute(
            "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
        ).fetchone():
            continue
        _add_model(
            connection,
            {
                "id": model_id,
                "family": family,
                "data": data,
                "architecture": architecture,
                "status": status,
                "primary_reason": reason,
                "reusable_value": reuse,
            },
        )


def _load_unified_external(connection: sqlite3.Connection, paths: BuildPaths) -> None:
    source = _relative(paths.unified_common_csv, paths.repo)
    with paths.unified_common_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = MODEL_NAME_TO_ID.get(row["model"])
            if model_id is None:
                continue
            _insert_evaluation(
                connection,
                model_id=model_id,
                protocol_id="fixed_external_1977_v1",
                scope=row["scope"],
                target=row["target"],
                n=int(row["n"]),
                mae_ev=float(row["mae_eV"]),
                baseline_model_id="M01",
                delta_mae_ev=float(row["delta_vs_routed_v4_500k_eV"]),
                ci95_low_ev=float(row["paired_normal_ci95_low_eV"]),
                ci95_high_ev=float(row["paired_normal_ci95_high_eV"]),
                source_path=source,
            )
    source = _relative(paths.unified_pcqm_csv, paths.repo)
    with paths.unified_pcqm_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = MODEL_NAME_TO_ID.get(row["model"])
            if model_id is None or not row.get("mae_eV"):
                continue
            _insert_evaluation(
                connection,
                model_id=model_id,
                protocol_id="pcqm_valid_4981_v1",
                scope="pcqm_valid",
                target="gap",
                n=int(row["n"]),
                mae_ev=float(row["mae_eV"]),
                baseline_model_id="M01",
                delta_mae_ev=float(row["delta_vs_routed_v4_500k_eV"]),
                ci95_low_ev=(
                    float(row["paired_normal_ci95_low_eV"])
                    if row.get("paired_normal_ci95_low_eV")
                    else None
                ),
                ci95_high_ev=(
                    float(row["paired_normal_ci95_high_eV"])
                    if row.get("paired_normal_ci95_high_eV")
                    else None
                ),
                source_path=source,
            )


def _load_retention_b(connection: sqlite3.Connection, paths: BuildPaths) -> None:
    data = _read_json(paths.retention_b_common_json)
    source = _relative(paths.retention_b_common_json, paths.repo)
    for scope, block in data["scopes"].items():
        metrics = block["retention_2m_replay_gps7"]
        delta = block["candidate_minus_routed_v4_500k"]
        for target in TARGETS:
            _insert_evaluation(
                connection,
                model_id="M21",
                protocol_id="fixed_external_1977_v1",
                scope=scope,
                target=target,
                n=block["n"],
                mae_ev=metrics[target]["mae_eV"],
                r2=metrics[target].get("r2"),
                baseline_model_id="M01",
                delta_mae_ev=delta[target]["mae_delta_eV"],
                ci95_low_ev=delta[target]["normal_ci95_eV"][0],
                ci95_high_ev=delta[target]["normal_ci95_eV"][1],
                source_path=source,
            )
    pcqm = _read_json(paths.retention_b_pcqm_json)
    _insert_evaluation(
        connection,
        model_id="M21",
        protocol_id="pcqm_valid_4981_v1",
        scope="pcqm_valid",
        target="gap",
        n=pcqm["n_valid"],
        mae_ev=pcqm["retention_2m_replay_gps7_gap_mae_eV"],
        baseline_model_id="M01",
        delta_mae_ev=pcqm["candidate_minus_routed_v4_500k_gap"]["mae_delta_eV"],
        ci95_low_ev=pcqm["candidate_minus_routed_v4_500k_gap"]["normal_ci95_eV"][0],
        ci95_high_ev=pcqm["candidate_minus_routed_v4_500k_gap"]["normal_ci95_eV"][1],
        source_path=_relative(paths.retention_b_pcqm_json, paths.repo),
    )


def _load_retention_d(connection: sqlite3.Connection, paths: BuildPaths) -> None:
    data = _read_json(paths.retention_d_json)
    source = _relative(paths.retention_d_json, paths.repo)
    for scope, block in data["scopes"].items():
        for target in TARGETS:
            value = block[target]
            _insert_evaluation(
                connection,
                model_id="M25",
                protocol_id="fixed_external_1977_v1",
                scope=scope,
                target=target,
                n=1977 if scope == "all" else 999 if scope == "ood1000" else 978,
                mae_ev=value["retention_d_mae_eV"],
                baseline_model_id="M21",
                delta_mae_ev=value["delta_d_minus_b_eV"],
                source_path=source,
            )
    pcqm = data["pcqm"]
    _insert_evaluation(
        connection,
        model_id="M25",
        protocol_id="pcqm_valid_4981_v1",
        scope="pcqm_valid",
        target="gap",
        n=4981,
        mae_ev=pcqm["retention_d_gap_mae_eV"],
        baseline_model_id="M21",
        delta_mae_ev=pcqm["delta_d_minus_b_eV"],
        source_path=source,
    )


def _load_internal(
    connection: sqlite3.Connection,
    paths: BuildPaths,
    model_id: str,
    path: Path,
) -> None:
    data = _read_json(path)
    metrics = data["test_metrics"]
    n = round(data.get("n_graphs", 2_000_000) * 0.1)
    for source_target, target in (
        ("HOMO", "homo"),
        ("LUMO", "lumo"),
        ("Gap", "gap"),
        ("average", "average"),
    ):
        values = metrics[source_target]
        _insert_evaluation(
            connection,
            model_id=model_id,
            protocol_id="internal_random_model_specific",
            scope="test",
            target=target,
            n=n,
            mae_ev=values.get("mae", values.get("mae_eV")),
            r2=values.get("r2"),
            source_path=_relative(path, paths.repo),
        )


def _artifact_model_id(path: Path) -> str | None:
    name = path.name.lower()
    rules = (
        ("repaired_2m_d", "M25"),
        ("retention", "M21"),
        ("coverage_2m_gps7", "M26"),
        ("expansion_500k", "M00"),
        ("routed", "M01"),
        ("expansion_1m", "M02"),
        ("repair_v2", "M04"),
        ("repair_v3_1p5m", "M05"),
        ("broad", "M06"),
        ("distill", "M10"),
    )
    return next((model_id for token, model_id in rules if token in name), None)


def _load_artifacts(
    connection: sqlite3.Connection,
    paths: BuildPaths,
    *,
    hash_artifacts: bool,
) -> None:
    for path in sorted((paths.repo / "models").rglob("*")):
        if not path.is_file() or path.name.lower() == "readme.md":
            continue
        suffix = path.suffix.lower()
        kind = "checkpoint" if suffix in {".pt", ".pth", ".ckpt"} else "model_support"
        connection.execute(
            """
            INSERT INTO artifacts(model_id, path, bytes, sha256, artifact_kind)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                _artifact_model_id(path),
                _relative(path, paths.repo),
                path.stat().st_size,
                _sha256(path) if hash_artifacts else None,
                kind,
            ),
        )


def build_database(
    repo: Path,
    output: Path,
    *,
    hash_artifacts: bool = False,
) -> None:
    """Build a new database atomically from immutable result artifacts."""
    paths = BuildPaths.from_repo(repo)
    required = (
        paths.inventory_csv,
        paths.unified_common_csv,
        paths.unified_pcqm_csv,
        paths.retention_b_common_json,
        paths.retention_b_pcqm_json,
        paths.retention_d_json,
        paths.uniform_a_internal_json,
        paths.retention_d_internal_json,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing database inputs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA)
        _load_inventory(connection, paths)
        connection.executemany(
            "INSERT INTO protocols VALUES (?, ?, ?, ?)",
            (
                (
                    "fixed_external_1977_v1",
                    "external",
                    "Common 1,977; OOD 999; P8-hard 978, aligned by identity.",
                    1,
                ),
                (
                    "pcqm_valid_4981_v1",
                    "external",
                    "Paired ETKDG-valid PCQM4Mv2 valid proxy, 4,981 rows.",
                    1,
                ),
                (
                    "internal_random_model_specific",
                    "internal",
                    "Model-specific random split; never use for cross-model promotion.",
                    0,
                ),
            ),
        )
        _load_unified_external(connection, paths)
        _load_retention_b(connection, paths)
        _load_retention_d(connection, paths)
        _load_internal(connection, paths, "M26", paths.uniform_a_internal_json)
        _load_internal(connection, paths, "M25", paths.retention_d_internal_json)
        connection.executemany(
            "INSERT INTO causes VALUES (?, ?, ?)",
            ((key, name, description) for key, (name, description) in CAUSES.items()),
        )
        connection.executemany(
            "INSERT INTO model_causes VALUES (?, ?)",
            (
                (model_id, cause_id)
                for model_id, causes in MODEL_CAUSES.items()
                for cause_id in causes
            ),
        )
        _load_artifacts(connection, paths, hash_artifacts=hash_artifacts)
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    finally:
        connection.close()
    output.unlink(missing_ok=True)
    temporary.replace(output)


def export_query(
    database: Path,
    sql: str,
    output_csv: Path,
    parameters: Iterable[Any] = (),
) -> None:
    """Export a query with stable column order."""
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(sql, tuple(parameters))
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

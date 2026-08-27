"""No-runtime acceptance for the downloaded official-PCQM 100K cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def accept(root: Path, expected_source_commit: str | None = None) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = root / manifest["split_file"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    failures_path = root / manifest["failures_file"]
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    ledger_path = root / manifest["replacement_ledger_file"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(manifest.get("format") == "molgap-pcqm-gap100k-cache-v2", "format")
    require(manifest.get("complete") is True, "complete")
    require(manifest.get("source_dataset") == "piero0/pcqm4mv2", "source dataset")
    if expected_source_commit is not None:
        require(manifest.get("source_commit") == expected_source_commit, "source commit")
    require(manifest.get("official_train_rows_read") == 3_378_606, "train rows")
    require(manifest.get("official_validation_role_read") is False, "official validation read")
    require(manifest.get("test_dev_role_read") is False, "test-dev read")
    require(manifest.get("gpu_used") is False, "GPU used")
    require(manifest.get("unresolved_graphs") == 0, "unresolved graphs")
    require(
        isinstance(manifest.get("bondless_graphs"), int)
        and manifest.get("bondless_graphs") >= 0,
        "bondless graph count",
    )
    require(manifest.get("train_graphs") == 100_000, "train graph count")
    require(manifest.get("validation_graphs") == 10_000, "validation graph count")
    require(manifest.get("atom_feature_dim") == 9, "atom feature dim")
    require(manifest.get("bond_feature_dim") == 3, "bond feature dim")
    require(manifest.get("rwse_dim") == 16, "RWSE dim")
    feature_ranges = manifest.get("feature_ranges", {})
    for key, width in (
        ("atom_feature_min", 9),
        ("atom_feature_max", 9),
        ("bond_feature_min", 3),
        ("bond_feature_max", 3),
    ):
        values = feature_ranges.get(key, [])
        require(
            len(values) == width
            and all(isinstance(value, int) for value in values),
            f"feature range {key}",
        )
    require(sha256_file(split_path) == manifest.get("split_file_sha256"), "split SHA")
    require(
        sha256_file(failures_path) == manifest.get("failures_file_sha256"),
        "failures SHA",
    )
    require(
        sha256_file(ledger_path) == manifest.get("replacement_ledger_sha256"),
        "replacement ledger SHA",
    )
    require(split.get("format") == "molgap-pcqm-gap100k-split-v2", "split format")
    require(
        failures.get("format") == "molgap-pcqm-gap100k-failures-v2",
        "failures format",
    )
    require(
        ledger.get("format") == "molgap-pcqm-gap100k-replacements-v1",
        "replacement ledger format",
    )
    initial_train = split.get("initial_train", [])
    initial_validation = split.get("initial_validation", [])
    reserve = split.get("reserve", [])
    train = split.get("train", [])
    validation = split.get("validation", [])
    require(len(initial_train) == 100_000, "initial train split count")
    require(len(initial_validation) == 10_000, "initial validation split count")
    require(len(reserve) == 1_024, "reserve count")
    require(len(train) == 100_000, "train split count")
    require(len(validation) == 10_000, "validation split count")
    require(set(initial_train).isdisjoint(initial_validation), "initial split overlap")
    require(
        set(reserve).isdisjoint(initial_train)
        and set(reserve).isdisjoint(initial_validation),
        "reserve overlap",
    )
    require(set(train).isdisjoint(validation), "split overlap")
    all_indices = initial_train + initial_validation + reserve + train + validation
    require(min(all_indices, default=-1) >= 0, "negative index")
    require(max(all_indices, default=3_378_606) < 3_378_606, "non-train index")
    require(
        index_sha256(initial_train) == manifest.get("initial_train_index_sha256"),
        "initial train index SHA",
    )
    require(
        index_sha256(initial_validation)
        == manifest.get("initial_validation_index_sha256"),
        "initial validation index SHA",
    )
    require(
        index_sha256(reserve) == manifest.get("reserve_index_sha256"),
        "reserve index SHA",
    )
    require(index_sha256(train) == manifest.get("train_index_sha256"), "train index SHA")
    require(
        index_sha256(validation) == manifest.get("validation_index_sha256"),
        "validation index SHA",
    )
    replacements = ledger.get("replacements", [])
    attempts = failures.get("attempts", [])
    require(
        manifest.get("replacement_policy")
        == "seed42-reserve-in-priority-order",
        "replacement policy",
    )
    require(
        ledger.get("policy") == "seed42-reserve-in-priority-order",
        "replacement ledger policy",
    )
    require(
        manifest.get("replacement_count") == len(replacements),
        "replacement count",
    )
    require(
        manifest.get("failed_graph_attempts") == len(attempts),
        "failed attempt count",
    )
    require(len(attempts) >= len(replacements), "replacement failure evidence")
    expected_effective = {
        "train": list(initial_train),
        "validation": list(initial_validation),
    }
    used_replacements = set()
    successful_reserve_ranks = set()
    initial_failures = {
        (item.get("role"), item.get("slot"), item.get("row_index"))
        for item in attempts
        if item.get("attempt_kind") == "initial"
    }
    require(len(initial_failures) == len(replacements), "initial failure count")
    for item in replacements:
        role = item.get("role")
        slot = item.get("slot")
        original = item.get("original_row_index")
        replacement = item.get("replacement_row_index")
        reserve_rank = item.get("reserve_rank")
        require(role in expected_effective, f"replacement role {role}")
        if role not in expected_effective:
            continue
        require(isinstance(slot, int) and 0 <= slot < len(expected_effective[role]), "replacement slot")
        if not isinstance(slot, int) or not 0 <= slot < len(expected_effective[role]):
            continue
        require(expected_effective[role][slot] == original, "replacement original")
        require(replacement in reserve, "replacement outside reserve")
        require(
            isinstance(reserve_rank, int)
            and 0 <= reserve_rank < len(reserve)
            and reserve[reserve_rank] == replacement,
            "replacement reserve rank",
        )
        require(replacement not in used_replacements, "replacement reused")
        require((role, slot, original) in initial_failures, "missing initial failure")
        expected_effective[role][slot] = replacement
        used_replacements.add(replacement)
        if isinstance(reserve_rank, int):
            successful_reserve_ranks.add(reserve_rank)
    require(expected_effective["train"] == train, "effective train ledger")
    require(expected_effective["validation"] == validation, "effective validation ledger")
    failed_reserve_rows = {
        item.get("row_index")
        for item in attempts
        if item.get("attempt_kind") == "reserve"
    }
    failed_reserve_ranks = set()
    for item in attempts:
        if item.get("attempt_kind") != "reserve":
            continue
        rank = item.get("reserve_rank")
        row_index = item.get("row_index")
        require(
            isinstance(rank, int)
            and 0 <= rank < len(reserve)
            and reserve[rank] == row_index,
            "failed reserve rank",
        )
        if isinstance(rank, int):
            failed_reserve_ranks.add(rank)
    consumed = manifest.get("reserve_rows_consumed")
    require(isinstance(consumed, int) and 0 <= consumed <= len(reserve), "reserve consumed")
    if isinstance(consumed, int) and 0 <= consumed <= len(reserve):
        require(
            successful_reserve_ranks | failed_reserve_ranks == set(range(consumed)),
            "reserve priority sequence",
        )
        require(
            successful_reserve_ranks.isdisjoint(failed_reserve_ranks),
            "reserve outcome overlap",
        )
    require(
        failed_reserve_rows.isdisjoint(train)
        and failed_reserve_rows.isdisjoint(validation),
        "failed reserve row selected",
    )
    aggregate = hashlib.sha256()
    shard_graphs = {"train": 0, "validation": 0}
    for shard in manifest.get("shards", []):
        path = root / shard["file"]
        require(path.is_file(), f"missing shard {path.name}")
        if path.is_file():
            require(sha256_file(path) == shard.get("sha256"), f"shard SHA {path.name}")
        role = shard.get("role")
        require(role in shard_graphs, f"shard role {role}")
        if role in shard_graphs:
            shard_graphs[role] += int(shard.get("graph_count", 0))
        aggregate.update(
            f"{role}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    require(shard_graphs == {"train": 100_000, "validation": 10_000}, "shard graph totals")
    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate SHA")
    result = {
        "format": "molgap-pcqm-gap100k-cache-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": manifest.get("source_commit"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "train_index_sha256": manifest.get("train_index_sha256"),
        "validation_index_sha256": manifest.get("validation_index_sha256"),
        "replacement_count": manifest.get("replacement_count"),
        "reserve_rows_consumed": manifest.get("reserve_rows_consumed"),
        "failed_graph_attempts": manifest.get("failed_graph_attempts"),
        "bondless_graphs": manifest.get("bondless_graphs"),
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

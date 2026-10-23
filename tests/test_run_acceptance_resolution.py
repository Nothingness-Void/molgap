"""Frozen acceptance lookup and retrieval maps; no model or remote execution."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from molgap.experiment_family_workflow import RunContext
from molgap.experiment_run_acceptance import record_run_retrieval, resolve_run_acceptance_inputs
from test_experiment_package import repo
from run_acceptance_fixtures import frozen_acceptance_run, write


@pytest.fixture
def acceptance_run(tmp_path, repo):
    return frozen_acceptance_run(tmp_path, repo)


def test_resolves_all_frozen_arms_after_source_records_disappear(acceptance_run):
    case = acceptance_run
    root = case.paths.artifact_root / "retrieved"
    # Mutable checkout records are deliberately absent. The frozen archive owns them.
    assert not (case.repo / "plans/acceptance.json").exists()
    result = resolve_run_acceptance_inputs(*case.original, retrieved_root=root)
    assert result["spec"].identity == case.spec.identity
    assert set(result["outputs"]) == {"k1_candidate", "gptrans_candidate"}
    for arm, item in result["outputs"].items():
        assert item == {"output_dir": str(root / arm), "expected": case.expected}
        assert result["contexts"][arm].platform_version == "7"
        location = result["locations"][arm]
        assert location["run_id"] == case.spec.to_dict()["logical_run_id"] + ":" + arm + ":downstream"
        assert location["terminal"].endswith("/terminal.json")
        assert location["trace"].endswith("/trace_manifest.json")
        binding = result["target_identities"][arm]
        digest, provenance = binding.digest(torch.zeros(3, dtype=torch.float32),
            context=result["contexts"][arm], expected=item["expected"])
        assert digest == case.expected["target_sha256"]
        assert provenance["encoding"] == "little-endian-float32-contiguous-raw-bytes"
    assert not (case.run / "operations/retrieved_outputs").exists()


@pytest.mark.parametrize("fault", ["missing_arm", "expected", "outside", "frozen", "traversal"])
def test_explicit_outputs_cannot_change_identity_or_escape_storage(acceptance_run, fault):
    case = acceptance_run
    outputs = {a["arm_id"]: {"output_dir": str(case.paths.artifact_root / "retrieved" / a["arm_id"])}
               for a in case.spec.to_dict()["arms"]}
    first = next(iter(outputs))
    if fault == "missing_arm": del outputs[first]
    elif fault == "expected": outputs[first]["expected"] = {**case.expected, "optimizer_steps": 9}
    elif fault == "outside": outputs[first]["output_dir"] = str(case.paths.data_root / "output")
    elif fault == "frozen": outputs[first]["output_dir"] = str(case.run / "prepared")
    elif fault == "traversal": outputs[first]["output_dir"] = str(case.paths.artifact_root / ".." / "escape")
    with pytest.raises(ValueError):
        resolve_run_acceptance_inputs(*case.original, supplied=outputs)


def test_legacy_repository_retrieved_root_and_explicit_locations(acceptance_run):
    case = acceptance_run
    root = case.repo / "experiments/retrieved"
    resolved = resolve_run_acceptance_inputs(*case.original, retrieved_root=root)
    where = deepcopy(resolved["locations"])
    first = next(iter(where))
    where[first]["terminal"] = "experiments/alternate/terminal.json"
    assert resolve_run_acceptance_inputs(*case.original, retrieved_root=root,
        locations=where)["locations"] == where
    where[first]["run_id"] = "wrong-run"
    with pytest.raises(ValueError):
        resolve_run_acceptance_inputs(*case.original, retrieved_root=root, locations=where)


def retain(case, arm, *, receipt=None):
    context = RunContext.from_launch(case.spec, receipt or case.run / "operations/launch_receipt.json",
        case.package, expected_package_identity=case.manifest["package_identity"], arm_id=arm)
    destination = case.paths.artifact_root / "retained" / (context.platform_version if receipt else "original") / arm
    trace = b"synthetic retained trace"
    destination.mkdir(parents=True)
    (destination / "trace.json").write_bytes(trace)
    manifest = {"format": "molgap-family-output-v1", "context": context.to_dict(),
                "artifacts": {"trace": {"path": "trace.json", "sha256": hashlib.sha256(trace).hexdigest()}}}
    write(destination / "output_manifest.json", manifest)
    pin = hashlib.sha256((destination / "output_manifest.json").read_bytes()).hexdigest()
    return context, destination, pin


@pytest.mark.parametrize("continuation", [False, True])
def test_maps_bind_exact_context_and_manifest_without_copying(acceptance_run, continuation):
    case = acceptance_run
    receipt = None
    original_mappings = {}
    if continuation:
        from test_experiment_launch import build
        response = deepcopy(case.receipt["response"])
        response["platform_version"]["value"] = "8"
        for physical in response["physical_runs"]["value"]:
            physical["platform_version"]["value"] = "8"
        receipt = case.paths.artifact_root / "continuation-receipt.json"
        write(receipt, build((case.spec, case.package, case.manifest["package_identity"]), response))
        for arm in (a["arm_id"] for a in case.spec.to_dict()["arms"]):
            context, destination, pin = retain(case, arm)
            target = record_run_retrieval(*case.original, context=context,
                destination=destination, manifest_sha256=pin)
            original_mappings[target] = target.read_bytes()
    destinations = {}
    for arm in (a["arm_id"] for a in case.spec.to_dict()["arms"]):
        context, destination, pin = retain(case, arm, receipt=receipt)
        target = record_run_retrieval(*case.original, context=context,
            destination=destination, manifest_sha256=pin, continuation_receipt=receipt)
        assert target == record_run_retrieval(*case.original, context=context,
            destination=destination, manifest_sha256=pin, continuation_receipt=receipt)
        assert json.loads(target.read_bytes())["context"] == context.to_dict()
        assert target.parent.name == ("8" if continuation else "retrieved_outputs")
        destinations[arm] = destination
    result = resolve_run_acceptance_inputs(*case.original, continuation_receipt=receipt)
    assert result["original_receipt"] == case.run / "operations/launch_receipt.json"
    assert all(path.read_bytes() == raw for path, raw in original_mappings.items())
    if continuation:
        assert all("original" in item["output_dir"]
                   for item in resolve_run_acceptance_inputs(*case.original)["outputs"].values())
    assert {arm: item["output_dir"] for arm, item in result["outputs"].items()} == {
        arm: str(path) for arm, path in destinations.items()}
    assert result["raw_traces"] == {arm: str(path / "trace.json") for arm, path in destinations.items()}
    if continuation:
        from molgap.research_memory.paths import artifact_locator
        from molgap.research_memory.trace import file_digest, json_bytes

        # Retrieval maps alone never choose a successor. Only all arms' sealed
        # canonical terminal pointers may select that exact retained receipt.
        declaration = case.spec.to_dict()
        continuation_paths = []
        for index, binding in enumerate(declaration["prospective"]["arms"]):
            owner = case.repo / binding["output"]
            trajectory = {"trajectory_id": binding["trajectory_id"]}
            write(owner / "trajectory.json", trajectory)
            continued = {"attempt_id": binding["arm_id"] + "-v8"}
            pointer = owner / "continuation.json"
            origin_receipt = case.run / "operations/launch_receipt.json"
            record = {"format": "molgap-terminal-continuation-v1", "continued": continued,
                **{key: {"path": artifact_locator(case.repo, path), "sha256": file_digest(path)}
                   for key, path in (("origin_receipt", origin_receipt), ("continued_receipt", receipt))}}
            write(pointer, record)
            hashes = {item["path"]: item["sha256"] for item in
                      (record["origin_receipt"], record["continued_receipt"])}
            hashes[artifact_locator(case.repo, pointer)] = file_digest(pointer)
            terminal = {"trajectory_id": binding["trajectory_id"],
                "run_id": declaration["logical_run_id"] + ":" + binding["arm_id"] + ":downstream",
                "continuation_ref": artifact_locator(case.repo, pointer), "artifact_hashes": hashes,
                "same_run_observation": continued}
            finalized = owner / "rml_finalized"
            # This is a minimal byte-integrity fixture, not scientific closure.
            for name, value in (("trajectory.json", trajectory), ("prospective_snapshot.json", trajectory),
                                ("v5_evidence.json", {"synthetic": True}), ("terminal_input.json", terminal)):
                write(finalized / name, value)
            published = {path.name: file_digest(path) for path in finalized.iterdir()}
            write(finalized / "finalization.json", {"format": "molgap-rml-finalization-v1",
                "trajectory_id": binding["trajectory_id"], "published_hashes": published, "replacements": {},
                "source_artifact_digest": hashlib.sha256(json_bytes({
                    "trajectory_sha256": file_digest(finalized / "prospective_snapshot.json"),
                    "terminal": terminal, "trace_sha256": None})).hexdigest()})
            continuation_paths.append(pointer)
            if index == 0:
                with pytest.raises(ValueError, match="one common continuation"):
                    resolve_run_acceptance_inputs(*case.original)
        automatic = resolve_run_acceptance_inputs(*case.original)
        assert automatic["outputs"] == result["outputs"]
        assert automatic["options"]["receipt_path"] == receipt
        assert {value.platform_version for value in automatic["contexts"].values()} == {"8"}
        assert resolve_run_acceptance_inputs(*case.original)["outputs"] == automatic["outputs"]
        retained_receipt = receipt.read_bytes()
        receipt.write_bytes(b"changed accepted receipt")
        with pytest.raises(ValueError, match="SHA mismatch"):
            resolve_run_acceptance_inputs(*case.original)
        receipt.write_bytes(retained_receipt)
        retained_continuation = continuation_paths[0].read_bytes()
        continuation_paths[0].write_bytes(b"changed continuation")
        with pytest.raises(ValueError, match="SHA mismatch"):
            resolve_run_acceptance_inputs(*case.original)
        continuation_paths[0].write_bytes(retained_continuation)
    first = next(iter(destinations))
    (destinations[first] / "output_manifest.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest changed"):
        resolve_run_acceptance_inputs(*case.original, continuation_receipt=receipt)


def test_damaged_retrieval_never_publishes_mapping(acceptance_run):
    case = acceptance_run
    arm = case.spec.to_dict()["arms"][0]["arm_id"]
    context, destination, pin = retain(case, arm)
    (destination / "trace.json").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="artifact differs"):
        record_run_retrieval(*case.original, context=context,
                             destination=destination, manifest_sha256=pin)
    assert not (case.run / "operations/retrieved_outputs").exists()


def test_frozen_package_tampering_blocks_resolution(acceptance_run):
    case = acceptance_run
    (case.package / "source.tar.gz").write_bytes(b"corrupt original package")
    with pytest.raises(ValueError):
        resolve_run_acceptance_inputs(*case.original, retrieved_root=case.paths.artifact_root / "outputs")

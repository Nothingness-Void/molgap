"""K1 candidate/native execution qualification and retained proof artifacts.

The shared family runner owns orchestration and certificate assembly.  This
module owns the K1-specific shadow comparison, state/evaluation evidence, and
the strict recheck performed before a certificate is accepted.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .experiment_family_workflow import _artifact_path, tensor_digest
from .family_training_data import PassBatchSampler


def _runner_state_fingerprint(value):
    """Resolve the runner's shared state encoder without importing it eagerly."""
    from .family_training_runner import _state_fingerprint
    return _state_fingerprint(value)


def _runner_loaded_source(package_dir):
    """Resolve the runner's source-package admission check lazily."""
    from .family_training_runner import _loaded_source
    return _loaded_source(package_dir)


def _runner_file_sha(path):
    """Resolve the runner's normalized artifact hash lazily."""
    from .family_training_runner import normalized_file_sha
    return normalized_file_sha(path)


def _native_probe(recipe):
    """Build the two-step historical probe through the runner's shared helper."""
    from .family_training_runner import _probe_recipe
    return replace(_probe_recipe(recipe), k1_execution=None)


def _execution_compare_state(value):
    """Return the state used by a K1 backend shadow comparison.

    A backend choice is execution provenance, not training state.  The probe
    recipes therefore have different identities, which are retained in their
    own artifacts but must not make an otherwise identical checkpoint differ.
    The cursor identity is the sole field that may be removed from this
    comparison; all model, optimizer, scheduler, metric and RNG fields remain
    covered by the exact fingerprint.
    """
    if type(value) is not dict or type(value.get("cursor")) is not dict:
        raise ValueError("K1 execution comparison requires a family state cursor")
    cursor = value["cursor"]
    if "recipe_identity" not in cursor:
        raise ValueError("K1 execution comparison cursor lacks recipe identity")
    compared = dict(value)
    compared["cursor"] = {key: child for key, child in cursor.items()
                           if key != "recipe_identity"}
    return compared


def _execution_state_fingerprint(value):
    return _runner_state_fingerprint(_execution_compare_state(value))


def _execution_artifact(root, record, label):
    """Verify and load one CPU proof artifact named by an execution record."""
    import torch
    if (type(record) is not dict or set(record) != {"path", "sha256", "fingerprint"}
            or type(record["path"]) is not str):
        raise ValueError("Malformed K1 execution " + label + " artifact record")
    path = _artifact_path(root, record["path"])
    if _runner_file_sha(path) != record["sha256"]:
        raise ValueError("K1 execution " + label + " artifact hash changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if _execution_state_fingerprint(state) != record["fingerprint"]:
        raise ValueError("K1 execution " + label + " state fingerprint changed")
    return state


def _execution_tensor_artifact(root, record, label):
    import torch
    if (type(record) is not dict or set(record) != {"path", "sha256", "fingerprint"}
            or type(record["path"]) is not str):
        raise ValueError("Malformed K1 execution " + label + " artifact record")
    path = _artifact_path(root, record["path"])
    if _runner_file_sha(path) != record["sha256"]:
        raise ValueError("K1 execution " + label + " artifact hash changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or _runner_state_fingerprint(value) != record["fingerprint"]:
        raise ValueError("K1 execution " + label + " tensor fingerprint changed")
    return value


def _verify_execution_proof(root, probe, proof, *, required,
                            expected_candidate_sha256=None):
    """Recheck the CPU-retained candidate/native K1 equivalence evidence.

    This gate intentionally does not construct CUDA tensors or claim that a
    CPU artifact is a native accelerator qualification.  The native runtime
    gate remains the producer-side requirement; this function only verifies
    that the retained shadow artifacts were not changed after publication.
    """
    configured = getattr(probe, "k1_execution", None)
    if configured is None:
        if proof is not None:
            raise ValueError("Unconfigured K1 recipe contains execution proof")
        return None
    if proof is None:
        if required:
            raise ValueError("Configured K1 recipe lacks native execution proof")
        return None
    expected = {"format", "status", "candidate_recipe_identity", "native_recipe_identity",
                "config", "candidate_state", "native_state", "evaluation"}
    if type(proof) is not dict or set(proof) != expected:
        raise ValueError("Malformed K1 execution proof")
    native_probe = _native_probe(probe)
    if (proof["format"] != "molgap-k1-execution-proof-v1" or
            proof["status"] != "accepted" or
            proof["candidate_recipe_identity"] != probe.identity or
            proof["native_recipe_identity"] != native_probe.identity or
            proof["config"] != configured.to_dict()):
        raise ValueError("K1 execution proof binding differs from the recipe")
    candidate_record, native_record = proof["candidate_state"], proof["native_state"]
    if not isinstance(candidate_record, dict) or not isinstance(native_record, dict):
        raise ValueError("Malformed K1 execution state artifact record")
    if (expected_candidate_sha256 is not None and
            candidate_record.get("sha256") != expected_candidate_sha256):
        raise ValueError("K1 execution candidate state is not the retained repeat-b proof")
    if (candidate_record.get("path") != "repeat-b.pt" or
            native_record.get("path") != "execution-native-state.pt"):
        raise ValueError("K1 execution proof uses an unexpected state artifact")
    candidate = _execution_artifact(root, candidate_record, "candidate state")
    native = _execution_artifact(root, native_record, "native state")
    from .family_training_session import validate_session_resume
    for label, state, recipe in (("candidate", candidate, probe),
                                 ("native", native, native_probe)):
        cursor = state.get("cursor")
        if (not isinstance(cursor, dict) or cursor.get("recipe_identity") != recipe.identity):
            raise ValueError("K1 execution " + label + " cursor recipe identity differs")
        progress = validate_session_resume(recipe, state)
        if (progress.status != "complete" or progress.optimizer_step != 2 or
                len(state["rng_state"].get("cuda", [])) != 1):
            raise ValueError("K1 execution " + label + " proof is not a complete two-step CUDA state")
    if _execution_state_fingerprint(candidate) != _execution_state_fingerprint(native):
        raise ValueError("K1 execution candidate/native state differs")
    evaluation = proof["evaluation"]
    if (type(evaluation) is not dict or
            set(evaluation) != {"format", "fixture", "source_idx_sha256", "equal",
                                "source_idx", "candidate", "native"} or
            evaluation["format"] != "molgap-k1-evaluation-forward-v1" or
            evaluation["fixture"] != "first-training-batch" or
            evaluation["equal"] is not True):
        raise ValueError("Malformed K1 execution evaluation proof")
    candidate_eval = _execution_tensor_artifact(root, evaluation["candidate"], "candidate evaluation")
    native_eval = _execution_tensor_artifact(root, evaluation["native"], "native evaluation")
    source_idx = _execution_tensor_artifact(root, evaluation["source_idx"], "evaluation source")
    import torch
    if (source_idx.device.type != "cpu" or source_idx.dtype != torch.int64 or
            source_idx.ndim != 1 or source_idx.numel() != 128 or
            torch.unique(source_idx).numel() != source_idx.numel() or
            int(source_idx.min()) < 0 or int(source_idx.max()) >= probe.train_rows or
            tensor_digest(source_idx, role="source_idx") != evaluation["source_idx_sha256"] or
            probe.train_rows < 128):
        raise ValueError("K1 execution evaluation source identity changed")
    # The accepted fixed PCQM train role starts at source_idx zero.  The frozen
    # sampler, rather than a fabricated sequential range, determines batch 0.
    sampler = PassBatchSampler(probe.train_rows, probe.batch_size, probe.seed,
        algorithm=probe.sampler, pass_index=0, start_batch=0, drop_last=probe.drop_last)
    expected_source = torch.tensor(next(iter(sampler)), dtype=torch.int64)
    if not torch.equal(source_idx, expected_source):
        raise ValueError("K1 execution evaluation source is not the frozen first training batch")
    source_length = source_idx.numel()
    if (candidate_eval.device.type != "cpu" or native_eval.device.type != "cpu" or
            not candidate_eval.is_floating_point() or not native_eval.is_floating_point() or
            candidate_eval.ndim != 1 or native_eval.ndim != 1 or
            candidate_eval.numel() != source_length or native_eval.numel() != source_length or
            candidate_eval.dtype != native_eval.dtype or
            not bool(torch.isfinite(candidate_eval).all()) or
            not bool(torch.isfinite(native_eval).all()) or
            not torch.equal(candidate_eval, native_eval)):
        raise ValueError("K1 execution evaluation candidate/native output differs")
    return proof


def _run_execution_shadow(*, recipe, data, tuning, options, output):
    """Run a two-update native shadow with the candidate recipe's data/RNG."""
    import torch
    from .family_training_kernel import FamilyKernel
    from .family_training_loaders import FamilyLoaders
    from .family_training_session import FamilyTrainingSession
    from .training_reproducibility import atomic_torch_save, configure_fp32_determinism

    configure_fp32_determinism(recipe.seed)
    kernel = FamilyKernel(recipe, Path(options["initial_state_path"]), device="cuda")
    _runner_loaded_source(options["package_dir"])
    loaders = FamilyLoaders(recipe, data, tuning)
    path = Path(output) / "execution-native-state.pt"
    session = FamilyTrainingSession(kernel, loaders,
        checkpoint=lambda progress, state: atomic_torch_save(path, state),
        selected=lambda *args: None, pass_finished=lambda *args: None)
    try:
        session.run()
    finally:
        if not session.closed:
            session.close()
    return torch.load(path, map_location="cpu", weights_only=True), path


def _run_execution_evaluation(*, recipe, data, tuning, options, retained_state):
    """Evaluate the first training batch using a retained two-step model state."""
    import torch
    from .family_training_kernel import FamilyKernel
    from .family_training_loaders import FamilyLoaders
    from .training_reproducibility import configure_fp32_determinism

    configure_fp32_determinism(recipe.seed)
    kernel = FamilyKernel(recipe, Path(options["initial_state_path"]), device="cuda")
    _runner_loaded_source(options["package_dir"])
    loaders = FamilyLoaders(recipe, data, tuning)
    try:
        model_state = retained_state.get("model") if isinstance(retained_state, dict) else None
        if not isinstance(model_state, dict):
            raise ValueError("K1 execution evaluation state lacks a model state dict")
        kernel.model.load_state_dict(model_state, strict=True)
        batch = next(iter(loaders.train(0, 0)))
        source = getattr(batch, "source_idx", getattr(batch, "row_index", None))
        if source is None:
            raise ValueError("K1 execution evaluation fixture lacks source identity")
        kernel.model.eval()
        with torch.no_grad():
            value = kernel.forward(batch).detach().cpu()
        return value, source.detach().cpu().long()
    finally:
        loaders.close()
        kernel.close()


def _write_execution_proof(*, output, probe, candidate_state, candidate_path,
                           native_state, native_path, candidate_eval, candidate_eval_path,
                           native_eval, native_eval_path, source_idx, source_idx_path):
    from .training_reproducibility import atomic_json, atomic_torch_save

    # Evaluation artifacts are tensors rather than resumable states; write them
    # through the same atomic owner and retain their exact CPU hashes.
    atomic_torch_save(candidate_eval_path, candidate_eval)
    atomic_torch_save(native_eval_path, native_eval)
    atomic_torch_save(source_idx_path, source_idx)
    candidate_record = {"path": Path(candidate_path).name,
        "sha256": _runner_file_sha(candidate_path),
        "fingerprint": _execution_state_fingerprint(candidate_state)}
    native_record = {"path": Path(native_path).name,
        "sha256": _runner_file_sha(native_path),
        "fingerprint": _execution_state_fingerprint(native_state)}
    candidate_eval_record = {"path": Path(candidate_eval_path).name,
        "sha256": _runner_file_sha(candidate_eval_path), "fingerprint": _runner_state_fingerprint(candidate_eval)}
    native_eval_record = {"path": Path(native_eval_path).name,
        "sha256": _runner_file_sha(native_eval_path), "fingerprint": _runner_state_fingerprint(native_eval)}
    source_record = {"path": Path(source_idx_path).name,
        "sha256": _runner_file_sha(source_idx_path), "fingerprint": _runner_state_fingerprint(source_idx)}
    proof = {"format": "molgap-k1-execution-proof-v1", "status": "accepted",
        "candidate_recipe_identity": probe.identity,
        "native_recipe_identity": _native_probe(probe).identity,
        "config": probe.k1_execution.to_dict(), "candidate_state": candidate_record,
        "native_state": native_record,
        "evaluation": {"format": "molgap-k1-evaluation-forward-v1",
            "fixture": "first-training-batch",
            "source_idx_sha256": tensor_digest(source_idx, role="source_idx"),
            "source_idx": source_record,
            "equal": bool(candidate_eval.shape == native_eval.shape and
                            candidate_eval.dtype == native_eval.dtype and
                            torch_equal(candidate_eval, native_eval)),
            "candidate": candidate_eval_record, "native": native_eval_record}}
    if proof["evaluation"]["equal"] is not True:
        raise ValueError("K1 execution evaluation candidate/native output differs")
    atomic_json(Path(output) / "execution_proof.json", proof)
    return proof


def torch_equal(left, right):
    import torch
    return torch.equal(left, right)

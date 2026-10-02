"""Repeated qualified enrollment is one run, not ambiguous evidence."""
import inspect
import textwrap
from copy import deepcopy

from molgap.research_memory import replay


def test_verified_reference_reuse_is_exact_only():
    # Exercise the actual enrollment block with a verifier stub, not a model.
    source = inspect.getsource(replay.build_replay_pool)
    block = source[source.index("    for _, bundle"):source.index("    from .ema_replay")]
    candidate = {"comparison_role": "candidate", "reference_id": "reference"}
    reference = {"comparison_role": "reference", "reference_id": "reference",
                 "trace_artifact_sha256": "a" * 64}
    manifests = [candidate]
    calls = []

    def verifier(root, pointer, *, expected_bundle):
        calls.append(pointer)
        view = deepcopy(reference)
        if pointer == "conflicting":
            view["trace_artifact_sha256"] = "b" * 64
        return view

    bundles = [(None, {"reference_id": "reference", "candidate_reference_qualification_ref": p})
               for p in ("first", "second", "conflicting")]
    exec(textwrap.dedent(block), {"records": {"reference_bundles": bundles}, "manifests": manifests,
                 "verify_candidate_reference": verifier, "root": None})
    assert calls == ["first", "second", "conflicting"]
    assert len(manifests) == 3
    assert manifests[1] == reference
    assert manifests[2]["trace_artifact_sha256"] == "b" * 64

"""GPTrans-T native gates over shared saved-artifact execution analysis."""
from types import SimpleNamespace


def _native_validator(*, arm_id, root, context, record, certificate, identity):
    from .experiment_family_workflow import _artifact_path, _json, _check_context
    from .family_training_runner import _verify_preflight, _recipe_configuration, _verify_profile_proof
    if arm_id == "optimized":
        provenance = _json(_artifact_path(root, "runtime_provenance.json"))
        _check_context(provenance["context"], context)
        _verify_preflight(root, _recipe_configuration(record), provenance, profiling_required=True)
        return
    from . import pcqm_gptrans_v4 as owner
    from .gptrans_legacy_execution_profile import verify_profile_state_proof
    from .screen_policy import canonical_fingerprint
    preflight = _json(_artifact_path(root, "preflight.json"))
    runtime = _json(_artifact_path(root, "runtime_manifest.json"))
    certificate_id = canonical_fingerprint(certificate)
    if (preflight.get("accepted") is not True
            or preflight.get("runtime_certificate_id") != certificate_id
            or preflight.get("runtime_certificate") != certificate
            or preflight.get("runtime_manifest") != runtime
            or preflight.get("source_commit") != context.source_commit
            or preflight.get("source_archive_sha256") != context.source_archive_sha256
            or preflight.get("manifest_sha256") != record["manifest_sha256"]
            or preflight.get("initial_state_artifact_sha256") != record["seed42_initial_state_artifact_sha256"]
            or preflight.get("variant") != "reference"
            or preflight.get("parameters") != owner.EXPECTED_PARAMETERS
            or runtime.get("runtime_fingerprint") != certificate.get("runtime_fingerprint")
            or runtime.get("accelerator", {}).get("device_count_visible") != 1
            or any(preflight.get(key) is not False for key in
                   ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))):
        raise ValueError("GPTrans native preflight/source/runtime binding differs")
    owner.validate_runtime_certificate(certificate, {
        **owner._scientific_fields(), "platform_id": context.platform,
        "accelerator": runtime["accelerator"]["name"], "runtime_certificate_id": certificate_id})
    _verify_profile_proof(root, SimpleNamespace(identity=identity), certificate.get("profile_proof"), required=True)
    verify_profile_state_proof(root, record, certificate.get("profile_state_proof"), required=True)


def analyze(*, package, receipt, output_root):
    from . import gptrans_infrastructure_comparison as comparison
    from .execution_analysis import analyze_execution
    return analyze_execution(package=package, receipt=receipt, output_root=output_root,
        comparison=comparison, native_validator=_native_validator,
        scientific_subdirectories={"original": "normalized"},
        report_format="molgap-gptrans-execution-comparison-v1")

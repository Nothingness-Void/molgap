"""K1 owner gates over the shared saved-output execution comparison."""
from pathlib import Path
import json

from .execution_analysis import paired_spans, prediction_comparison, speed, trace_observations


def _native_validator(*, arm_id, root, context, record, certificate, identity):
    from .experiment_family_workflow import _artifact_path, _json, _check_context
    from .family_training_runner import _verify_preflight, _recipe_configuration, _verify_profile_proof
    provenance = _json(_artifact_path(root, "runtime_provenance.json"))
    _check_context(provenance["context"], context)
    if arm_id == "original":
        from types import SimpleNamespace
        from .k1_screen_training import validate_runtime_preflight
        validate_runtime_preflight(root, provenance)
        fingerprints = certificate.get("profiled_plain_state_fingerprints", [])
        if len(fingerprints) != 2 or len(set(fingerprints)) != 1:
            raise ValueError("Legacy profiled/plain state qualification differs")
        _verify_profile_proof(root, SimpleNamespace(identity=identity), certificate.get("profile_proof"), required=True)
    else:
        _verify_preflight(root, _recipe_configuration(record), provenance, profiling_required=True)


def analyze(*, package, receipt, output_root):
    from . import k1_infrastructure_comparison as comparison
    from .execution_analysis import analyze_execution
    return analyze_execution(package=package, receipt=receipt, output_root=output_root,
        comparison=comparison, native_validator=_native_validator,
        report_format="molgap-k1-execution-comparison-v1")

def main():
    import argparse
    from .research_memory.trace import atomic_write, json_bytes
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for field in ("package", "receipt", "output-root", "report"):
        parser.add_argument("--" + field, required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Use a fresh retained analysis report")
    result = analyze(package=args.package, receipt=args.receipt, output_root=args.output_root)
    atomic_write(args.report, json_bytes(result))
    print(json.dumps({key: result[key] for key in ("status", "parent_timed_train_workers", "numerical_comparison")}))


if __name__ == "__main__":
    main()

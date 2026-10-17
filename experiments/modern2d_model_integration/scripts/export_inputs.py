"""Thin scheduled-CPU entry for the shared selected-role export owner.

This does not submit jobs, publish datasets or run models. The caller supplies
the approved boundary and externally frozen request, then explicitly executes.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    from molgap.common.io import _safe_local
    from molgap.research_memory.paths import repo_local_path
    from molgap.data.indexed_inputs import AcceptedIndexedInputPermit, IndexedInputPermit
    from molgap.pcqm_indexed_export import export_indexed_roles
    from molgap.research_memory.trace import file_digest

    boundary = args.boundary.absolute()
    _safe_local(boundary)
    request_path = args.request.absolute()
    _safe_local(request_path)
    if not request_path.is_relative_to(boundary):
        raise ValueError("Request is outside the explicitly authorized boundary")
    request = json.loads(request_path.read_bytes())
    fields = {"format", "scope", "parent_root", "output_root", "roles", "permits",
              "authorities", "role_histories", "parent_acceptance_sha256", "max_chunk_bytes", "max_rows"}
    if (type(request) is not dict or set(request) != fields or
            request["format"] != "molgap-indexed-private-export-request-v1" or
            request["scope"] not in {"CPU_SYNTHETIC_ONLY", "ACCEPTED_INDEXED_INPUT_V1"}):
        raise ValueError("Unknown selected-role export request")
    roles = request["roles"]
    if (type(roles) is not list or len(roles) != 2 or any(type(role) is not str for role in roles) or
            set(roles) not in ({"train_100k", "common_dev"}, {"train_500k", "common_dev"})):
        raise ValueError("ROLE_BLOCKED: request exactly one training scale plus Common-Dev")
    kind = IndexedInputPermit if request["scope"] == "CPU_SYNTHETIC_ONLY" else AcceptedIndexedInputPermit
    permits = {role: kind(**value) for role, value in request["permits"].items()}
    options = {key: request[key] for key in fields - {"format", "scope", "permits"}}
    for key in ("parent_root", "output_root"):
        relative = request[key]
        if (type(relative) is not str or not relative or "\\" in relative or ":" in relative or
                any(part in {"", ".", ".."} for part in relative.split("/"))):
            raise ValueError("Export roots must be confined relative paths")
        _safe_local(boundary / relative)
        options[key] = repo_local_path(boundary, relative)
    options["permits"] = permits
    if not args.execute:
        print(json.dumps({"status": "PREPARE_ONLY_NOT_EXECUTED", "request_sha256": file_digest(request_path),
            "roles": request["roles"], "remote_submission": False, "training_released": False}))
        return 0
    result = export_indexed_roles(**options)
    receipt = options["output_root"] / "acceptance.json"
    print(json.dumps({"status": result["status"], "acceptance_path": str(receipt),
        "acceptance_sha256": file_digest(receipt), "publication_status": result["publication_status"],
        "roles": {role: value["rows"] for role, value in result["roles"].items()}, "training_released": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

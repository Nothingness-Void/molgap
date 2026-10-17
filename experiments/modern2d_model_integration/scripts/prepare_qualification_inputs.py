"""Thin CPU metadata preparation; --execute is required to open graph payloads.

The request supplies absolute export_root/output_root/boundary, independent
export/parent acceptance SHA256 pins, exact roles and accepted export inventory.
It must itself reside inside that boundary. The caller owns CPU allocation and
budget; this entry never submits work or constructs a model.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    from molgap.common.io import _safe_local
    from molgap.modern2d_input_preparation import (
        MAX_METADATA_BYTES, REQUEST_FORMAT, prepare_qualification_inputs)
    request_path = args.request.absolute()
    _safe_local(request_path)
    if request_path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Preparation request exceeds the metadata byte ceiling")
    request = json.loads(request_path.read_bytes())
    if type(request) is not dict or request.get("format") != REQUEST_FORMAT:
        raise ValueError("Unknown qualification input request")
    boundary = Path(request["boundary"])
    _safe_local(boundary)
    if not request_path.is_relative_to(boundary):
        raise ValueError("Request is outside the explicitly authorized boundary")
    result = prepare_qualification_inputs(request, execute=args.execute)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

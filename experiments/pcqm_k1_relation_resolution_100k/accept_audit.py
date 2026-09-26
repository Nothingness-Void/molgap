"""Check saved NO_TRAIN audit chunks and already accepted reference reuse."""
import argparse
from pathlib import Path
from molgap.k1_relation_audit import accept_audit
from molgap.research_memory.trace import atomic_write, json_bytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write(args.output, json_bytes(accept_audit(args.audit_root)))

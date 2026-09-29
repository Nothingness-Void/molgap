"""CLI for streaming acceptance of a retrieved GPTrans audit segment."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.gptrans_v5_audit_acceptance import accept_segment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--acceptance", required=True, type=Path)
    args = parser.parse_args()
    print(accept_segment(args.root, args.epochs, args.acceptance))


if __name__ == "__main__":
    main()

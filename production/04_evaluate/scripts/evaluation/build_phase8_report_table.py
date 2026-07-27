"""Generate the human-readable Phase 8 comparison table."""

from __future__ import annotations

from pathlib import Path

from molgap.phase8_reporting import build_comparison_rows, write_comparison


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    out_dir = repo / "results" / "phase8" / "reporting"
    rows = build_comparison_rows(repo)
    write_comparison(
        rows,
        out_dir / "model_comparison.csv",
        out_dir / "model_comparison.md",
    )
    print(out_dir / "model_comparison.md")


if __name__ == "__main__":
    main()

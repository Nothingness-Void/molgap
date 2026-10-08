"""Thin saved-trace lifecycle entry; no model or platform operations."""
import sys
from molgap.k1_saved_analysis import main

if __name__ == "__main__":
    sys.argv.append("trace_pair")
    main()

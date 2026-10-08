"""Thin saved-prediction lifecycle entry; never constructs a model."""
import sys
from molgap.k1_saved_analysis import main

if __name__ == "__main__":
    sys.argv.append("bn_rows")
    main()

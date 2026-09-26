"""Freeze source/role/budget identities through native RML planning."""
import json
from molgap.k1_relation_study_records import freeze

if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))

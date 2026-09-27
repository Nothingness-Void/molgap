"""Close the retained v1 joint-study preflight failure through shared RML."""
from __future__ import annotations

import json

from molgap.k1_joint_failure_records import close_failure_v1


def main() -> None:
    print(json.dumps(close_failure_v1(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

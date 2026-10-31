"""One-line experiment cards for scanning closed work and picking search terms.

Read-only view over the derived trajectory index and each trajectory's own
hypothesis block. Nothing is written and nothing is inferred from prose: a field
the record does not carry stays None.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_STOP = frozenset({"the", "and", "for", "with", "of", "to", "in", "a", "step", "only", "new"})


def _keywords(*texts) -> list[str]:
    words = re.split(r"[^a-z0-9]+", " ".join(t for t in texts if t).lower())
    return sorted({w for w in words if len(w) > 2 and w not in _STOP and not w.isdigit()})


def build_cards(repo_root) -> list[dict]:
    root = Path(repo_root).resolve()
    index = json.loads((root / "research_memory/derived/trajectory_index.json").read_text(encoding="utf-8"))
    cards = []
    for row in index["trajectories"]:
        record = json.loads((root / row["record_path"]).read_text(encoding="utf-8"))
        hyp = record.get("hypothesis") or {}
        mechanism = hyp.get("changed_mechanism")
        cards.append({
            "trajectory_id": row["trajectory_id"],
            "family_id": row["family_id"],
            "outcome": row["outcome"],
            "mechanism": mechanism,
            "deficiency": hyp.get("observed_deficiency"),
            "falsifier": hyp.get("cheapest_falsifier"),
            "related_families": hyp.get("related_closed_family_ids") or [],
            "keywords": _keywords(mechanism, row["family_id"]),
            "decision_ref": row.get("decision_ref"),
        })
    return cards


def select(cards, *, outcome=None, term=None) -> list[dict]:
    term = term.lower() if term else None
    return [c for c in cards
            if (outcome is None or c["outcome"] == outcome)
            and (term is None or term in json.dumps(c, ensure_ascii=False).lower())]

"""One-line experiment cards for scanning closed work and picking search terms.

Read-only view over the derived trajectory index, each trajectory's own
hypothesis block, retained paired_analysis.json files and the optional closure
facts / attribution labels. Nothing is written and nothing is inferred from
prose: a field the record does not carry stays None.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_STOP = frozenset("the and for with step only new training train mean all one two three nine atom same model change "
                  "unchanged accepted existing fixed selected before keep replace exact from into both".split())


def _keywords(*texts) -> list[str]:
    words = re.split(r"[^a-z0-9]+", " ".join(t for t in texts if t).lower())
    return sorted({w for w in words if len(w) > 2 and w not in _STOP and not w.isdigit()
                    and not re.fullmatch(r"s\d+|\d+k|pcqm|r\d", w)})


def _evidence(root: Path, row: dict, record: dict, outcomes: dict) -> dict:
    """Effect and failure mode only where a record states them; each carries its source."""
    facts = (record.get("decision") or {}).get("facts") or {}
    out = {"effect_meV": facts.get("effect_meV"), "ci_meV": facts.get("ci_meV"), "effect_source": facts.get("source"),
           "failure_mode": facts.get("failure_mode"),
           "failure_mode_source": "closure_facts" if facts.get("failure_mode") else None,
           "search_terms": facts.get("search_terms") or []}
    # Only this trajectory's own bound paired file, and only against a reference arm that closed CLOSED:
    # two lever arms differing by seed measure seed noise, not the lever, so their effect stays None.
    ready = root / (record.get("comparison_readiness_ref") or "-")
    ready = json.loads(ready.read_text(encoding="utf-8")) if ready.suffix == ".json" and ready.is_file() else {}
    pair = (ready.get("candidate_artifact_bindings") or {}).get("paired_analysis")
    d = json.loads((root / pair["ref"]).read_text(encoding="utf-8")) if pair else {}
    if outcomes.get(ready.get("reference_id")) == "CLOSED" and "mean_delta_eV" in d:
        out.update(effect_meV=d["mean_delta_eV"] * 1000, ci_meV=[d["lower_95_eV"] * 1000, d["upper_95_eV"] * 1000],
                   effect_source=pair["ref"])
    ref = row.get("decision_ref")
    label = root / Path(ref).parent / "attribution_label.json" if ref else None
    if out["failure_mode"] is None and label is not None and label.is_file():
        d = json.loads(label.read_text(encoding="utf-8"))
        if row["trajectory_id"] in d.get("trajectory_ids", []) and d.get("failure_mode"):
            out.update(failure_mode=d["failure_mode"], failure_mode_source=d.get("source", "agent_interpretation"))
    return out


def _scale(card: dict):
    m = re.search(r"-(\d+)k(?=-|$)", f"{card['trajectory_id']}-{card['family_id']}")
    return int(m.group(1)) if m else None


def build_cards(repo_root) -> list[dict]:
    root = Path(repo_root).resolve()
    index = json.loads((root / "research_memory/derived/trajectory_index.json").read_text(encoding="utf-8"))
    outcomes = {e: row["outcome"] for row in index["trajectories"] for e in row.get("result_evidence_ids") or []}
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
            "reopen_conditions": (record.get("decision") or {}).get("reopen_conditions"),
            **_evidence(root, row, record, outcomes),
        })
    return cards


def select(cards, *, outcome=None, term=None) -> list[dict]:
    """--term is one whole word (letters/digits), case-insensitive, anywhere in the card."""
    word = term and re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])")
    return [c for c in cards
            if (outcome is None or c["outcome"] == outcome)
            and (not word or word.search(json.dumps(c, ensure_ascii=False).lower()))]


def lookup(cards, *, mechanism=None, failure_mode=None, family=None, term=None) -> dict:
    """Related closed work, why it failed (with source), open directions and search terms."""
    hit = [c for c in select(cards, term=term)
           if (mechanism is None or mechanism.lower() in (c["mechanism"] or "").lower())
           and (failure_mode is None or c["failure_mode"] == failure_mode)
           and (family is None or family.lower() in c["family_id"].lower())]
    by_family = defaultdict(list)
    for c in hit:
        by_family[c["family_id"]].append(c)
    # A family stays open when none of its negative results carries a measured effect.
    unfalsified = [{"family_id": f, "reason": "no NEGATIVE_UNDER_CONTRACT with a measured effect",
                    "falsifier": cs[0]["falsifier"], "reopen_conditions": cs[0]["reopen_conditions"]}
                   for f, cs in sorted(by_family.items())
                   if not any(c["outcome"] == "NEGATIVE_UNDER_CONTRACT" and c["effect_meV"] is not None for c in cs)]
    closing = [t for c in hit for t in c["search_terms"]]
    frequent = [w for w, _ in Counter(w for c in hit for w in c["keywords"]).most_common(12)]
    return {"query": {"mechanism": mechanism, "failure_mode": failure_mode, "family": family, "term": term},
            "matches": hit, "unfalsified": unfalsified,
            "search_terms": list(dict.fromkeys(closing + frequent))}


def gaps(cards) -> dict:
    """Thin or conflicting regions. Contradictions are prompts for review, not findings."""
    by_family, by_arm = defaultdict(list), defaultdict(list)
    for c in cards:
        by_family[c["family_id"]].append(c)
        if _scale(c) and c["outcome"] in {"POSITIVE_UNDER_CONTRACT", "NEGATIVE_UNDER_CONTRACT"}:
            # Same named arm at another scale: trajectory id minus track prefix, scale and seed tokens.
            by_arm[re.sub(r"^T[A-Z]?-|-(?:s|seed)\d+(?=-|$)|-\d+k(?=-|$)", "", c["trajectory_id"])].append(c)
    sparse = [{"family_id": f, "n_trajectories": len(cs), "n_with_effect": sum(c["effect_meV"] is not None for c in cs)}
              for f, cs in sorted(by_family.items()) if len(cs) == 1]

    def side(c):
        return {"id": c["trajectory_id"], "scale_k": _scale(c), "outcome": c["outcome"], "effect_meV": c["effect_meV"]}

    contradictions = [{"key": arm, "a": side(a), "b": side(b), "kind": "outcome_differs_across_scale_needs_review"}
                      for arm, cs in sorted(by_arm.items())
                      for a in cs for b in cs if _scale(a) < _scale(b) and a["outcome"] != b["outcome"]]
    return {"sparse": sparse, "contradictions": contradictions}

"""One-line experiment cards for scanning closed work and picking search terms.

Read-only view over the derived trajectory index, each trajectory's own
hypothesis block, retained paired_analysis.json files and the optional closure
facts / attribution labels. Nothing is written and nothing is inferred from
prose: a field the record does not carry stays None.
"""

from __future__ import annotations

import json
import posixpath
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

_STOP = frozenset("the and for with step only new training train mean all one two three nine atom same model change "
                  "unchanged accepted existing fixed selected before keep replace exact from into both".split())


def _keywords(*texts) -> list[str]:
    words = re.split(r"[^a-z0-9]+", " ".join(t for t in texts if t).lower())
    return sorted({w for w in words if len(w) > 2 and w not in _STOP and not w.isdigit()
                    and not re.fullmatch(r"s\d+|\d+k|pcqm|r\d", w)})


def _search_terms(card: dict) -> list[str]:
    """Closure-fact terms, then failure_mode, mechanism slug tokens and family id tokens; at most 8."""
    slug = card["mechanism"] if re.fullmatch(r"[a-z0-9_-]+", card["mechanism"] or "") else None
    words = [w for t in (slug, card["family_id"]) if t for w in re.split(r"[^a-z0-9]+", t.lower()) if w in _keywords(t)]
    return list(dict.fromkeys([*card["search_terms"], *filter(None, [card["failure_mode"]]), *words]))[:8]


def _evidence(root: Path, row: dict, record: dict, outcomes: dict) -> dict:
    """Effect and failure mode only where a record states them; each carries its source."""
    facts = (record.get("decision") or {}).get("facts") or {}
    out = {"effect_meV": facts.get("effect_meV"), "ci_meV": facts.get("ci_meV"), "effect_source": facts.get("source"),
           "failure_mode": facts.get("failure_mode"),
           "failure_mode_source": "closure_facts" if facts.get("failure_mode") else None,
           "search_terms": list(facts.get("search_terms") or [])}
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


class _GitTree:
    """The Path subset _evidence reads (/, suffix, is_file, read_text), over one commit's git objects."""

    def __init__(self, batch, rev, rel="."):
        self.batch, self.rev, self.rel, self.suffix = batch, rev, rel, posixpath.splitext(rel)[1]

    def __truediv__(self, other):
        rel = posixpath.normpath(posixpath.join(self.rel, str(other).replace("\\", "/")))
        return _GitTree(self.batch, self.rev, rel)

    def _blob(self):
        self.batch.stdin.write(f"{self.rev}:{self.rel}\n".encode())
        self.batch.stdin.flush()
        head = self.batch.stdout.readline().split()
        if head[-1] == b"missing":
            return None
        data = self.batch.stdout.read(int(head[2]) + 1)[:-1]
        return data if head[1] == b"blob" else None

    def is_file(self):
        return self._blob() is not None

    def read_text(self, encoding):
        if (data := self._blob()) is None:
            raise ValueError(f"cannot read {self.rel} at {self.rev}")
        return data.decode(encoding)


def _git(root, *args) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if done.returncode:
        raise ValueError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def build_cards(repo_root, ref=None, *, source=None) -> list[dict]:
    """ref: read that revision's committed files, not the working tree. source: label of a 'source' field."""
    root = Path(repo_root).resolve()
    rev = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip() if ref else "HEAD"
    batch = ref and subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tree = _GitTree(batch, rev) if ref else root
    index = json.loads((tree / "research_memory/derived/trajectory_index.json").read_text(encoding="utf-8"))
    outcomes = {e: row["outcome"] for row in index["trajectories"] for e in row.get("result_evidence_ids") or []}
    last = {}  # Newest commit touching each path at rev, from one history walk.
    for line in _git(root, "log", "--format=>%h %cs", "--name-only", rev).splitlines() if source else []:
        if line.startswith(">"):
            commit = line[1:]
        elif line:
            last.setdefault(line, commit)
    cards = []
    for row in index["trajectories"]:
        record = json.loads((tree / row["record_path"]).read_text(encoding="utf-8"))
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
            **_evidence(tree, row, record, outcomes),
        })
        cards[-1]["search_terms"] = _search_terms(cards[-1])
        if source:
            cards[-1]["source"] = {"ref": source, "record_path": row["record_path"],
                                   "last_commit": last.get(row["record_path"])}
    if batch:
        batch.communicate()
    return cards


def with_ref(repo_root, ref) -> list[dict]:
    """Local cards plus those only ref has (local wins an id), read as committed; links naming their source."""
    local = build_cards(repo_root, source="local")
    ids = {c["trajectory_id"] for c in local}
    return local + [c for c in build_cards(repo_root, ref, source=ref) if c["trajectory_id"] not in ids]


def _words(value) -> list[str]:
    """None, one string, or several strings; each string may hold several space-separated words."""
    if value is None:
        return []
    return [w.lower() for v in ([value] if isinstance(value, str) else value) for w in v.split()]


def _all_words(text: str, term) -> bool:
    """Every --term word occurs as a whole word (letters/digits boundary) in text."""
    return all(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", text) for w in _words(term))


def _all_substrings(text: str, mechanism) -> bool:
    return all(w in text for w in _words(mechanism))


def select(cards, *, outcome=None, term=None) -> list[dict]:
    """--term: one or more whole words (letters/digits), case-insensitive, all anywhere in the card."""
    return [c for c in cards
            if (outcome is None or c["outcome"] == outcome)
            and _all_words(json.dumps(c, ensure_ascii=False).lower(), term)]


def lookup(cards, *, mechanism=None, failure_mode=None, family=None, term=None) -> dict:
    """Related closed work, why it failed (with source), open directions and search terms."""
    hit = [c for c in select(cards, term=term)
           if _all_substrings(" ".join(filter(None, (c["mechanism"], c["family_id"], c["failure_mode"]))).lower(),
                              mechanism)
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


def _slug(text) -> str:
    if not text or re.fullmatch(r"[a-z0-9_-]+", text):
        return text or "-"
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:6])


def _effect(card: dict, *, see_decision=False) -> str:
    if card["effect_meV"] is not None:
        return f"{card['effect_meV']:+.2f}"
    if see_decision and card["outcome"] == "NEGATIVE_UNDER_CONTRACT" and card.get("decision_ref"):
        return f"null (see {card['decision_ref']})"
    return "null"


def compact_line(card: dict) -> str:
    """One line: id | outcome | family | mechanism slug | effect_meV | failure_mode [| source ref]."""
    return " | ".join([card["trajectory_id"], card["outcome"], card["family_id"], _slug(card["mechanism"]),
                       _effect(card), card["failure_mode"] or "-", *[s["ref"] for s in [card.get("source")] if s]])


def levers(cards, family: str) -> list[str]:
    """Every tested lever in the matching families: a substring, or a glob when it holds '*'.

    Same-run reference arms (outcome CLOSED) are omitted unless a family has nothing else.
    A NEGATIVE card without a recorded effect points at its decision file; prose is not read.
    """
    from fnmatch import fnmatchcase
    pattern = family.lower()
    hit = [c for c in cards if (fnmatchcase(c["family_id"].lower(), pattern) if "*" in pattern
                                else pattern in c["family_id"].lower())]
    by_family = defaultdict(list)
    for c in hit:
        by_family[c["family_id"]].append(c)
    rows = ["family | arm | outcome | effect_meV | failure_mode | mechanism"]
    for fam, cs in sorted(by_family.items()):
        tested = [c for c in cs if c["outcome"] != "CLOSED"] or cs
        for c in tested:
            arm = c["trajectory_id"].removeprefix("T-").removeprefix("TB-").removeprefix("TC-")
            arm = arm.removeprefix(fam).lstrip("-") or arm
            rows.append(" | ".join([fam, arm, c["outcome"], _effect(c, see_decision=True),
                                    c["failure_mode"] or "-", _slug(c["mechanism"])]))
    return rows


# Older (mostly server-branch) trajectories are indexed here as table rows but have
# no desktop trajectory record, hence no card. Read-only; nothing is inferred.
HISTORICAL_INDEX = "experiments/pcqm_k1_complete_local_audit/historical_evidence.md"
_HISTORICAL_SECTION = "## Indexed trajectory and attempt coverage"


def historical_pointer(repo_root, cards, *, mechanism=None, term=None, family=None):
    """One line naming uncarded rows of HISTORICAL_INDEX that match the query, else None.

    Each --mechanism word is a substring and each --term word a whole word of the row (id, outcome,
    mechanism, locator); --family matches 'pcqm-' + the id without track prefix
    and seed/version suffixes, as levers does for family ids.
    """
    from fnmatch import fnmatchcase
    path = Path(repo_root) / HISTORICAL_INDEX
    if not path.is_file() or not (mechanism or term or family):
        return None
    section = path.read_text(encoding="utf-8").partition(_HISTORICAL_SECTION)[2]
    carded = {c["trajectory_id"] for c in cards}
    hits = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 4 or not cells[0].startswith("`") or cells[0].strip("`") in carded:
            continue
        row, ident = " ".join(cells).lower(), cells[0].strip("`")
        fam = "pcqm-" + re.sub(r"^T[A-Z]?-|(?:-(?:s|v|r)\d+)+$", "", ident).lower()
        pattern = (family or "").lower()
        if (_all_substrings(row, mechanism) and _all_words(row, term)
                and (family is None or (fnmatchcase(fam, pattern) if "*" in pattern else pattern in fam))):
            hits.append(f"{ident} {cells[1]}")
    if not hits:
        return None
    shown = ", ".join(hits[:5]) + (f", +{len(hits) - 5} more" if len(hits) > 5 else "")
    return f"# not in cards: {len(hits)} historical trajectories match in {HISTORICAL_INDEX}: {shown}"


def gaps_compact(result: dict) -> list[str]:
    """One short line per gaps item."""
    def side(s):
        return f"{s['id']} {s['scale_k']}k {s['outcome']} {_effect(s)}"
    return ([f"sparse | {s['family_id']} | n={s['n_trajectories']} with_effect={s['n_with_effect']}"
             for s in result["sparse"]]
            + [f"contradiction | {c['key']} | {side(c['a'])} vs {side(c['b'])}" for c in result["contradictions"]])


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

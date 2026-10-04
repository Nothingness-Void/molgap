"""Atomic event-list serialization and call-site regression; no model execution."""
import ast
import json
from pathlib import Path

import pytest

from molgap.gptrans_portability import save_role_events


def test_role_events_roundtrip_and_atomic_replacement(tmp_path):
    target = tmp_path/"arm/role_events.json"
    rows = []
    for role in ("original_100k", "unseen_500k"):
        for event in ("prediction_input_and_labels_read", "metric_computed"):
            rows.append(dict(role=role,event=event,timestamp=1.0))
            save_role_events(target, rows)
            assert json.loads(target.read_text()) == rows
    assert list(target.parent.iterdir()) == [target]


def test_nonfinite_event_rejected_without_replacing_retained_bytes(tmp_path):
    target = tmp_path/"events.json"
    save_role_events(target, [])
    before = target.read_bytes()
    with pytest.raises(ValueError):
        save_role_events(target, [dict(timestamp=float("nan"))])
    assert target.read_bytes() == before


def test_all_worker_event_writes_use_list_capable_atomic_io():
    source=Path(__file__).resolve().parents[1]/"src/molgap/gptrans_portability.py"
    tree=ast.parse(source.read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="worker")
    calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call)]
    assert sum(ast.unparse(n.func)=="save_role_events" for n in calls)==3
    assert not any(ast.unparse(n.func)=="atomic_json" and 'role_events.json' in ast.unparse(n) for n in calls)
    loader=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=="graphs_for")
    assert ast.unparse(loader).index('_accepted_development(')<ast.unparse(loader).index('role_events.append(')

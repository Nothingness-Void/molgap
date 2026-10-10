"""Saved trace arithmetic only; no DLL, data, model or accelerator access."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def summarize(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/pcqm_k1_execution_efficiency"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("k1_kernel_closure_test", directory / "kernel_close.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def row(name, start, end, stream=7):
    return dict(name=name, start=start, end=end, deviceId=0, streamId=stream)


def test_intervals_are_unioned_not_summed_and_groups_explicitly_overlap(summarize):
    result = summarize([row("tensorop_gemm", 10, 30), row("simt_gemm", 20, 40, 8),
                        row("elementwise", 50, 60)])
    assert result["traced_kernel_union_ns"] == 40
    assert result["trace_span_ns"] == 50
    assert result["groups"]["all"] == dict(calls=3, summed_kernel_ns=50)
    assert result["groups"]["gemm"] == dict(calls=2, summed_kernel_ns=40)
    assert result["groups"]["tensorop"] == dict(calls=1, summed_kernel_ns=20)
    assert result["stream_ids"] == [7, 8]
    assert "instruction counts" in result["limits"][1]


@pytest.mark.parametrize("records", [[], [row("", 1, 2)], [row("gemm", 0, 1)],
    [row("gemm", 2, 1)], [row("gemm", 1., 2)], [row("gemm", 1, 2**64-1)]])
def test_unavailable_or_invalid_records_are_not_zero_cost(summarize, records):
    with pytest.raises(ValueError):
        summarize(records)


def test_input_order_and_copies_do_not_change_arithmetic(summarize):
    records = [row("RadixSort", 30, 40), row("indexing_backward", 10, 20)]
    saved = [dict(r) for r in records]
    forward, reverse = summarize(records), summarize(list(reversed(records)))
    # Equal-duration ranks have no ordering authority; compare their contents.
    assert sorted(forward.pop("top_kernels"), key=lambda r:r["name"]) == sorted(
        reverse.pop("top_kernels"), key=lambda r:r["name"])
    assert forward == reverse
    assert records == saved

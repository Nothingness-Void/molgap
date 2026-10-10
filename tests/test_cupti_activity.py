"""CPU-only mock DLL and native callback ABI tests; never load CUPTI/CUDA."""
import ctypes as ct
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import molgap.cupti_activity as activity


class Function:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class FakeDll:
    def __init__(self):
        self.calls, self.failures = [], {}
        self.version, self.dropped, self.next_error = 28, 0, 0
        self.start, self.end, self.kind = 100, 200, activity.CONCURRENT_KERNEL
        self.valid_size, self.repeat = 208, False
        self.record_count = 1
        self.name = ct.create_string_buffer(b"synthetic_kernel")
        self.request = self.complete = None
        self.offset = 0
        for name in ("cuptiGetVersion", "cuptiActivityRegisterCallbacks", "cuptiActivityEnable",
                     "cuptiActivityDisable", "cuptiActivityFlushAll", "cuptiFinalize",
                     "cuptiActivityGetNextRecord", "cuptiActivityGetNumDroppedRecords"):
            setattr(self, name, Function(lambda *args, name=name: self.invoke(name, *args)))

    def invoke(self, name, *args):
        self.calls.append((name, args))
        if name in self.failures:
            return self.failures[name]
        if name == "cuptiGetVersion":
            ct.cast(args[0], ct.POINTER(ct.c_uint32))[0] = self.version
        elif name == "cuptiActivityRegisterCallbacks":
            self.request, self.complete = args
        elif name == "cuptiActivityFlushAll":
            self.deliver()
        elif name == "cuptiActivityGetNextRecord":
            if self.next_error:
                return self.next_error
            if self.offset >= self.record_count and not self.repeat:
                return activity.MAX_LIMIT_REACHED
            offset = 0 if self.repeat else self.offset * 208
            ct.cast(args[2], ct.POINTER(ct.c_void_p))[0] = ct.cast(args[0], ct.c_void_p).value + offset
            self.offset += 1
        elif name == "cuptiActivityGetNumDroppedRecords":
            ct.cast(args[2], ct.POINTER(ct.c_size_t))[0] = self.dropped
        return 0

    def deliver(self):
        buffer = activity._BYTE_PTR()
        size, maximum = ct.c_size_t(), ct.c_size_t(99)
        self.request(ct.byref(buffer), ct.byref(size), ct.byref(maximum))
        assert maximum.value == 0
        address = ct.cast(buffer, ct.c_void_p).value
        assert address % 8 == 0
        for index in range(self.record_count):
            kernel = activity.ActivityKernel9.from_address(address + index * 208)
            kernel.kind, kernel.start, kernel.end = self.kind, self.start, self.end
            kernel.deviceId, kernel.contextId, kernel.streamId, kernel.correlationId = 2, 3, 4, 5 + index
            kernel.name = ct.addressof(self.name)
        self.offset = 0
        self.complete(None, 0, buffer, size.value, self.valid_size)


@pytest.fixture
def fake(monkeypatch):
    dll = FakeDll()
    monkeypatch.setattr(activity, "_load_dll", lambda path: dll)
    monkeypatch.setattr(activity, "_LOCK", activity.threading.Lock())
    monkeypatch.setattr(activity, "_ACTIVE", None)
    return dll


def collector(**kwargs):
    return activity.CuptiActivityCollector(Path(__file__).resolve().with_suffix(".dll"),
                                         synchronize=lambda: None, isolated_process=True, **kwargs)


def test_header_abi_offsets_and_size():
    assert ct.sizeof(activity.ActivityKernel9) == 208
    for field, offset in {"kind": 0, "cacheConfig": 4, "registersPerThread": 6,
                          "start": 16, "end": 24, "deviceId": 40, "contextId": 44,
                          "streamId": 48, "correlationId": 92, "name": 104,
                          "queued": 120, "launchType": 136, "graphNodeId": 144,
                          "pAccessPolicyWindow": 160, "clusterX": 176,
                          "localMemoryTotal_v2": 192, "maxActiveClusters": 204}.items():
        assert getattr(activity.ActivityKernel9, field).offset == offset


def test_import_and_construction_do_not_load_or_import_torch(monkeypatch):
    def forbidden(path):
        raise AssertionError("DLL must not load before enter")
    monkeypatch.setattr(activity, "_load_dll", forbidden)
    before = set(sys.modules)
    trace = collector()
    assert set(sys.modules) == before
    with pytest.raises(RuntimeError, match="closed"):
        trace.report()
    assert trace.diagnostics()["apiVersion"] is None
    assert trace.diagnostics()["dropped_records"] is None


def test_records_copied_cleanup_signatures_and_only_concurrent_kind(fake):
    trace = collector(buffer_size=256)
    with trace:
        assert activity._ACTIVE is trace
        assert trace._request_cb is not None
        with pytest.raises(RuntimeError, match="Nested"):
            with collector():
                pass
    report = trace.report()
    assert report["records"] == [{"name": "synthetic_kernel", "start": 100, "end": 200,
                                  "deviceId": 2, "contextId": 3, "streamId": 4, "correlationId": 5}]
    assert report["dropped_records"] == 0
    fake.name.value = b"overwritten"
    report["records"][0]["name"] = "changed copy"
    assert trace.report()["records"][0]["name"] == "synthetic_kernel"
    assert not trace._buffers and trace._request_cb is None and trace._complete_cb is None
    assert activity._ACTIVE is None and not activity._LOCK.locked()
    names = [name for name, _ in fake.calls]
    assert names.index("cuptiActivityDisable") < names.index("cuptiActivityFlushAll") < names.index("cuptiFinalize")
    for name, args in fake.calls:
        if name in ("cuptiActivityEnable", "cuptiActivityDisable"):
            assert args == (activity.CONCURRENT_KERNEL,)
    assert fake.cuptiActivityGetNextRecord.restype is ct.c_int
    assert len(fake.cuptiActivityGetNextRecord.argtypes) == 3
    with pytest.raises(RuntimeError, match="reused"):
        with trace:
            pass
    with pytest.raises(RuntimeError, match="already exited"):
        trace.__exit__(None, None, None)


@pytest.mark.parametrize("field,value,error", [
    ("start", 0, "timestamps"), ("end", 99, "timestamps"),
    ("end", (1 << 64) - 1, "timestamps"), ("kind", 3, "kind"),
    ("valid_size", 207, "Truncated"), ("valid_size", 300, "validSize"),
    ("next_error", 7, "CUptiResult=7"), ("repeat", True, "Nonadvancing"),
    ("dropped", 3, "dropped=3"),
])
def test_bad_records_callback_errors_and_drops_fail_closed(fake, field, value, error):
    setattr(fake, field, value)
    trace = collector(buffer_size=256)
    with pytest.raises(RuntimeError, match=error):
        with trace:
            pass
    with pytest.raises(RuntimeError, match="closed"):
        trace.report()
    assert trace._request_cb is None
    assert activity._ACTIVE is None


def test_callback_exception_never_escapes_native_callback(fake, monkeypatch):
    trace = collector()
    def allocation_failure(*args):
        raise MemoryError("synthetic allocation")
    with pytest.raises(RuntimeError, match="allocation"):
        with trace:
            monkeypatch.setattr(activity.ct, "create_string_buffer", allocation_failure)
            buffer = activity._BYTE_PTR()
            size, maximum = ct.c_size_t(99), ct.c_size_t(99)
            trace._request_cb(ct.byref(buffer), ct.byref(size), ct.byref(maximum))
            assert not buffer and size.value == maximum.value == 0
            # Don't ask this deliberately broken fake to fill the null buffer.
            fake.deliver = lambda: None
    assert trace._request_cb is None


@pytest.mark.parametrize("operation", ["cuptiActivityEnable", "cuptiActivityDisable",
                                       "cuptiActivityFlushAll", "cuptiFinalize"])
def test_api_errors_restore_or_pin_callback_lifetimes(fake, operation):
    fake.failures[operation] = 7
    trace = collector()
    with pytest.raises(RuntimeError, match="CUptiResult=7"):
        with trace:
            pass
    if operation in ("cuptiActivityFlushAll", "cuptiFinalize"):
        assert activity._ACTIVE is trace and trace._request_cb is not None
        assert activity._LOCK.locked()
        with pytest.raises(RuntimeError, match="poisoned"):
            with collector():
                pass
    else:
        assert activity._ACTIVE is None and trace._request_cb is None
    with pytest.raises(RuntimeError, match="closed"):
        trace.report()


def test_body_exception_tears_down_but_no_success_report(fake):
    trace = collector()
    with pytest.raises(ValueError, match="workload"):
        with trace:
            raise ValueError("workload")
    assert activity._ACTIVE is None
    assert "workload" in str(trace.diagnostics()["errors"])
    with pytest.raises(RuntimeError, match="closed"):
        trace.report()


def test_sync_failure_skips_unsafe_finalize_and_pins_callbacks(fake):
    def failed_sync():
        raise RuntimeError("sync failure")
    trace = activity.CuptiActivityCollector("not-loaded.dll", synchronize=failed_sync,
                                          isolated_process=True)
    with pytest.raises(RuntimeError, match="sync failure"):
        with trace:
            pass
    assert "cuptiFinalize" not in [name for name, _ in fake.calls]
    assert activity._ACTIVE is trace and trace._request_cb is not None


def test_version_conflict_and_profiler_conflict_before_registration(fake, monkeypatch):
    fake.version = 27
    with pytest.raises(RuntimeError, match="version"):
        with collector():
            pass
    assert activity._ACTIVE is None
    assert "cuptiActivityRegisterCallbacks" not in [name for name, _ in fake.calls]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        autograd=SimpleNamespace(_profiler_enabled=lambda: True)))
    fake.calls.clear()
    with pytest.raises(RuntimeError, match="profiler conflict"):
        with collector():
            pass
    assert not fake.calls
    assert activity._ACTIVE is None


def test_exclusive_declaration_and_buffer_validation():
    with pytest.raises(ValueError, match="Exclusive"):
        activity.CuptiActivityCollector("relative.dll", synchronize=lambda: None)
    with pytest.raises(ValueError, match="small"):
        collector(buffer_size=16)


def test_dll_path_checks_do_not_load_libraries(monkeypatch):
    monkeypatch.setattr(activity.os, "name", "nt")
    def forbidden(*args, **kwargs):
        raise AssertionError("No library loading in synthetic tests")
    monkeypatch.setattr(activity.ct, "WinDLL", forbidden, raising=False)
    with pytest.raises(ValueError, match="absolute"):
        activity._load_dll(Path("relative.dll"))
    with pytest.raises(ValueError, match="absolute"):
        activity._load_dll(Path(__file__).resolve())


def test_failure_diagnostics_preserve_partial_evidence_and_drop_query_error(fake):
    fake.failures["cuptiFinalize"] = 7
    fake.failures["cuptiActivityGetNumDroppedRecords"] = 8
    trace = collector()
    with pytest.raises(RuntimeError, match="CUptiResult=7"):
        with trace:
            pass
    diagnosis = trace.diagnostics()
    assert diagnosis["apiVersion"] == 28
    assert diagnosis["success"] is False
    assert diagnosis["callbacks_retained"] is True
    assert diagnosis["records"][0]["name"] == "synthetic_kernel"
    assert diagnosis["dropped_records"] is None
    assert diagnosis["dropped_records_complete"] is False
    assert "CUptiResult=8" in str(diagnosis["errors"])
    diagnosis["records"].clear()
    diagnosis["errors"].clear()
    assert trace.diagnostics()["records"] and trace.diagnostics()["errors"]
    with pytest.raises(RuntimeError, match="closed"):
        trace.report()


def test_api26_explicitly_unsupported_not_device_failure(fake):
    fake.version = 26
    trace = collector()
    with pytest.raises(RuntimeError, match="Unsupported CUPTI API version 26"):
        with trace:
            pass
    diagnosis = trace.diagnostics()
    assert diagnosis["apiVersion"] == 26
    assert "Unsupported CUPTI API version 26" in str(diagnosis["errors"])
    assert diagnosis["dropped_records"] is None
    assert diagnosis["records"] == [] and not diagnosis["success"]
    assert "cuptiActivityEnable" not in [name for name, _ in fake.calls]


def test_multiple_kernel_records_iterated_with_previous_record_pointer(fake):
    fake.record_count, fake.valid_size = 2, 416
    trace = collector()
    with trace:
        pass
    assert [row["correlationId"] for row in trace.report()["records"]] == [5, 6]
    assert trace.diagnostics()["dropped_records"] == 0
    assert trace.diagnostics()["dropped_records_complete"] is True

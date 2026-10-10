"""Raw CUDA 12.9 CUPTI kernel activity in a disposable, exclusive Windows process.

Never combine with torch.profiler, Nsight, injection agents, or another CUPTI
client (including one used earlier in this process). Stop other CUDA callers and
provide a device-wide synchronize callable before exit. No tensor-core inference
is made from names. Construction/import never loads a DLL or initializes CUDA.
"""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import sys
import threading

CONCURRENT_KERNEL = 10
MAX_LIMIT_REACHED = 12
_LOCK = threading.Lock()
_ACTIVE = None  # Also pins callbacks/DLL/buffers if safe detach fails.
_CALLBACK = getattr(ct, "WINFUNCTYPE", ct.CFUNCTYPE)
_BYTE_PTR = ct.POINTER(ct.c_uint8)
_REQUEST = _CALLBACK(None, ct.POINTER(_BYTE_PTR), ct.POINTER(ct.c_size_t),
                     ct.POINTER(ct.c_size_t))
_COMPLETE = _CALLBACK(None, ct.c_void_p, ct.c_uint32, _BYTE_PTR,
                      ct.c_size_t, ct.c_size_t)


class ActivityKernel9(ct.Structure):
    """cupti_activity.h Kernel9; cupti_common.h pack(1), record alignment 8.

    The cacheConfig bitfield union occupies one byte; only its raw byte is used.
    The full 208-byte x64 record is declared, not a guessed prefix ABI.
    """
    _pack_ = 1
    _fields_ = (
        [("kind", ct.c_uint32), ("cacheConfig", ct.c_uint8),
         ("sharedMemoryConfig", ct.c_uint8), ("registersPerThread", ct.c_uint16)]
        + [(n, ct.c_uint32) for n in ("partitionedGlobalCacheRequested",
                                     "partitionedGlobalCacheExecuted")]
        + [(n, ct.c_uint64) for n in ("start", "end", "completed")]
        + [(n, ct.c_uint32) for n in ("deviceId", "contextId", "streamId")]
        + [(n, ct.c_int32) for n in ("gridX", "gridY", "gridZ", "blockX", "blockY",
                                    "blockZ", "staticSharedMemory", "dynamicSharedMemory")]
        + [(n, ct.c_uint32) for n in ("localMemoryPerThread", "localMemoryTotal", "correlationId")]
        + [("gridId", ct.c_int64), ("name", ct.c_void_p), ("reserved0", ct.c_void_p)]
        + [(n, ct.c_uint64) for n in ("queued", "submitted")]
        + [(n, ct.c_uint8) for n in ("launchType", "isSharedMemoryCarveoutRequested",
                                    "sharedMemoryCarveoutRequested", "padding")]
        + [("sharedMemoryExecuted", ct.c_uint32), ("graphNodeId", ct.c_uint64),
           ("shmemLimitConfig", ct.c_uint32), ("graphId", ct.c_uint32),
           ("pAccessPolicyWindow", ct.c_void_p)]
        + [(n, ct.c_uint32) for n in ("channelID", "channelType", "clusterX", "clusterY",
                                     "clusterZ", "clusterSchedulingPolicy")]
        + [("localMemoryTotal_v2", ct.c_uint64), ("maxPotentialClusterSize", ct.c_uint32),
           ("maxActiveClusters", ct.c_uint32)]
    )


def _load_dll(path):
    if os.name != "nt" or ct.sizeof(ct.c_void_p) != 8:
        raise RuntimeError("Only Windows x64 CUPTI 12.9 is supported")
    if not path.is_absolute() or not path.is_file() or path.suffix.lower() != ".dll":
        raise ValueError("An existing absolute CUPTI DLL path is required")
    # Search dependencies only beside this verified DLL and in System32.
    return ct.WinDLL(str(path), winmode=0x100 | 0x800)


class CuptiActivityCollector:
    """Single-use context. Read report() only after successful teardown.

    isolated_process=True asserts sole CUPTI ownership for this process lifetime;
    CUPTI has no API to restore unknown previous activity callbacks. This is not
    a general in-process profiler replacement. synchronize must cover all devices
    used by the workload. No other profiler/CUDA thread may start during the scope.
    """
    def __init__(self, dll, *, synchronize, isolated_process=False, buffer_size=1 << 20):
        if isolated_process is not True or not callable(synchronize):
            raise ValueError("Exclusive isolated process and synchronize callable required")
        if type(buffer_size) is not int or buffer_size < ct.sizeof(ActivityKernel9):
            raise ValueError("Activity buffer is too small")
        self.path, self.synchronize, self.buffer_size = Path(dll), synchronize, buffer_size
        self.records, self.errors, self.dropped_records = [], [], 0
        self.api_version = None
        self._completed_buffers = self._dropped_queries = 0
        self._buffers = {}
        self._used = self._registered = self._enabled = self._closed = self._exited = False
        self._request_cb, self._complete_cb = _REQUEST(self._request), _COMPLETE(self._complete)

    def _call(self, name, *args):
        result = getattr(self._dll, name)(*args)
        if result != 0:
            raise RuntimeError(f"{name}: CUptiResult={result}")

    def _request(self, buffer, size, maximum):
        try:
            buffer[0], size[0], maximum[0] = _BYTE_PTR(), 0, 0
            storage = ct.create_string_buffer(self.buffer_size + 7)
            address = (ct.addressof(storage) + 7) & ~7
            self._buffers[address] = storage
            buffer[0], size[0] = ct.cast(address, _BYTE_PTR), self.buffer_size
        except BaseException as exc:
            self.errors.append(f"buffer request: {exc!r}")

    def _complete(self, context, stream, buffer, size, valid):
        address = ct.cast(buffer, ct.c_void_p).value
        self._completed_buffers += 1
        try:
            if address not in self._buffers or size != self.buffer_size or valid > size:
                raise RuntimeError("Unknown buffer or invalid validSize")
            record = ct.c_void_p()
            last = address - 1
            while valid:
                result = self._dll.cuptiActivityGetNextRecord(buffer, valid, ct.byref(record))
                if result == MAX_LIMIT_REACHED:
                    break
                if result != 0:
                    raise RuntimeError(f"cuptiActivityGetNextRecord: CUptiResult={result}")
                if not record.value or not address <= record.value <= address + valid - 4:
                    raise RuntimeError("Record pointer outside valid buffer")
                if ct.c_uint32.from_address(record.value).value != CONCURRENT_KERNEL:
                    raise RuntimeError("Unexpected activity kind")
                if record.value % 8 or record.value + ct.sizeof(ActivityKernel9) > address + valid:
                    raise RuntimeError("Truncated or unaligned Kernel9 record")
                if record.value <= last:
                    raise RuntimeError("Nonadvancing record iterator")
                last = record.value
                kernel = ActivityKernel9.from_address(record.value)
                if not 0 < kernel.start <= kernel.end < (1 << 64) - 1 or not kernel.name:
                    raise RuntimeError("Missing/nonfinite timestamps or kernel name")
                self.records.append({"name": ct.string_at(kernel.name).decode("utf-8", "strict"),
                                     **{n: int(getattr(kernel, n)) for n in
                                        ("start", "end", "deviceId", "contextId", "streamId", "correlationId")}})
        except BaseException as exc:
            self.errors.append(f"buffer complete: {exc!r}")
        finally:
            try:
                dropped = ct.c_size_t()
                self._call("cuptiActivityGetNumDroppedRecords", context, stream, ct.byref(dropped))
                self.dropped_records += dropped.value
                self._dropped_queries += 1
            except BaseException as exc:
                self.errors.append(f"dropped records query: {exc!r}")
            self._buffers.pop(address, None)

    def __enter__(self):
        global _ACTIVE
        if self._used or not _LOCK.acquire(blocking=False):
            raise RuntimeError("Nested, concurrent, reused or poisoned CUPTI collector")
        self._used, _ACTIVE = True, self
        try:
            torch = sys.modules.get("torch")
            if torch is not None and torch.autograd._profiler_enabled():
                raise RuntimeError("torch.profiler conflict")
            if ct.sizeof(ActivityKernel9) != 208 or ActivityKernel9.name.offset != 104:
                raise RuntimeError("Unsupported Kernel9 ABI")
            self._dll = _load_dll(self.path)
            signatures = {
                "cuptiGetVersion": [ct.POINTER(ct.c_uint32)],
                "cuptiActivityRegisterCallbacks": [_REQUEST, _COMPLETE],
                "cuptiActivityEnable": [ct.c_uint32], "cuptiActivityDisable": [ct.c_uint32],
                "cuptiActivityFlushAll": [ct.c_uint32], "cuptiFinalize": [],
                "cuptiActivityGetNextRecord": [_BYTE_PTR, ct.c_size_t, ct.POINTER(ct.c_void_p)],
                "cuptiActivityGetNumDroppedRecords": [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_size_t)],
            }
            for name, args in signatures.items():
                function = getattr(self._dll, name)
                function.argtypes, function.restype = args, ct.c_int
            version = ct.c_uint32()
            self._call("cuptiGetVersion", ct.byref(version))
            self.api_version = version.value
            if version.value != 28:
                raise RuntimeError(f"Unsupported CUPTI API version {version.value}, expected 28")
            self._call("cuptiActivityRegisterCallbacks", self._request_cb, self._complete_cb)
            self._registered = True
            self._enabled = True
            self._call("cuptiActivityEnable", CONCURRENT_KERNEL)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, traceback):
        global _ACTIVE
        if not self._used or self._exited:
            raise RuntimeError("CUPTI scope is not active or already exited")
        self._exited = True
        safe = not self._registered
        if self._registered:
            synced = flushed = False
            for operation in ("synchronize", "cuptiActivityDisable", "cuptiActivityFlushAll"):
                try:
                    if operation == "synchronize":
                        self.synchronize()
                        synced = True
                    elif operation == "cuptiActivityDisable":
                        if self._enabled:
                            self._call(operation, CONCURRENT_KERNEL)
                    else:
                        self._call(operation, 1)  # Forced flush includes incomplete records: rejected above.
                        flushed = True
                except BaseException as error:
                    self.errors.append(f"teardown {operation}: {error!r}")
            if synced and flushed:
                try:
                    self._call("cuptiFinalize")
                    safe = True
                except BaseException as error:
                    self.errors.append(f"teardown cuptiFinalize: {error!r}")
        if safe:
            self._buffers.clear()
            self._request_cb = self._complete_cb = None
            _ACTIVE = None
            _LOCK.release()
        failed = bool(self.errors or self.dropped_records)
        self._closed = safe and exc_type is None and not failed
        if exc is not None:
            self.errors.append(f"scope: {exc!r}")
        if failed:
            raise RuntimeError(f"CUPTI trace failed: dropped={self.dropped_records}; errors={self.errors}") from exc
        return False

    def report(self):
        if not self._closed or self.errors or self.dropped_records:
            raise RuntimeError("CUPTI trace is not successfully closed")
        return {"activity_kind": CONCURRENT_KERNEL, "cupti_api_version": self.api_version,
                "records": [dict(record) for record in self.records],
                "dropped_records": self.dropped_records}

    def diagnostics(self):
        """Partial raw evidence, never a successful trace or an unavailable==0 claim."""
        return {"apiVersion": self.api_version, "errors": list(self.errors),
                "dropped_records": self.dropped_records if self._dropped_queries else None,
                "dropped_records_complete": (self._closed and
                    self._dropped_queries == self._completed_buffers),
                "records": [dict(record) for record in self.records],
                "success": self._closed, "callbacks_retained": _ACTIVE is self}

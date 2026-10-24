"""Lossless storage of native proof bytes; tensor qualification stays with its owner."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import zipfile
import zlib

from .common.io import _safe_local
from .common.paths import validate_relative_path
from .research_memory.paths import repo_local_path
from .research_memory.trace import file_digest, json_bytes, sync_directory

ARCHIVE = "native_proofs.zip"
RESUME_ARCHIVE = "actual_resume_proofs.zip"
FORMAT = "molgap-native-proof-storage-v1"


def _proof_path(root, name):
    validate_relative_path(name)
    path = Path(root).absolute() / name
    _safe_local(path)
    return repo_local_path(root, name)


def proof_bindings(certificate):
    """Return every certificate-bound ``.pt`` member and its original digest."""
    result = {}

    def walk(value):
        if isinstance(value, dict):
            if str(value.get("path", "")).endswith(".pt") and "sha256" in value:
                name, digest = value["path"], value["sha256"]
                if name in result and result[name] != digest:
                    raise ValueError("Conflicting native proof bindings")
                result[name] = digest
            for key, child in value.items():
                if key != "proof_storage":
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(certificate)
    return result


def _storage_index(path):
    _safe_local(Path(path).absolute())
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 100000:
            raise ValueError("Ambiguous or excessive native proof archive members")
        info = archive.getinfo("index.json")
        if info.file_size > 16 * 1024 * 1024:
            raise ValueError("Native proof index exceeds metadata limit")
        index = json.loads(archive.read(info))
    if index.get("format") != FORMAT or not isinstance(index.get("files"), dict):
        raise ValueError("Unsupported native proof storage")
    return index


def storage_index(path):
    try:
        return _storage_index(path)
    except (zipfile.BadZipFile, KeyError, TypeError, AttributeError, UnicodeError,
            EOFError, OSError, zlib.error) as exc:
        raise ValueError("Malformed native proof storage index") from exc


def _segments(path):
    """Yield byte ranges that preserve the original ZIP representation exactly."""
    # Preserve ZIP headers, descriptors and central directory verbatim. Only
    # payload boundaries are shared; no tensor or pickle is reserialized.
    size = path.stat().st_size
    cuts = {0, size}
    with zipfile.ZipFile(path) as archive, path.open("rb") as source:
        for member in archive.infolist():
            source.seek(member.header_offset)
            header = source.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid native proof ZIP header")
            name_size, extra_size = struct.unpack_from("<HH", header, 26)
            start = member.header_offset + 30 + name_size + extra_size
            end = start + member.compress_size
            if not 0 <= start <= end <= size:
                raise ValueError("Native proof payload is outside its file")
            cuts.update((start, end))
    ordered = sorted(cuts)
    for start, end in zip(ordered, ordered[1:]):
        for offset in range(start, end, 8 * 1024 * 1024):
            yield offset, min(8 * 1024 * 1024, end - offset)


def pack_proofs(root, bindings, *, output=None):
    """Pack pinned originals without deleting or modifying them.

    The archive is a lossless byte store: it stores chunks of the original
    proof files and an index describing their order. Shared chunks are written
    once, so repeated proof state does not multiply retained bytes.
    """
    root = Path(root).absolute()
    _safe_local(root)
    output = Path(output).absolute() if output is not None else root / ARCHIVE
    _safe_local(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    index = {"format": FORMAT, "files": {}}
    seen = set()
    fd, temporary = tempfile.mkstemp(prefix=".native-proof-", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            def write(name, data):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data, compresslevel=1)

            for name, expected in sorted(bindings.items()):
                path = _proof_path(root, name)
                digest, chunks, size = hashlib.sha256(), [], 0
                with path.open("rb") as source:
                    for offset, length in _segments(path):
                        source.seek(offset)
                        data = source.read(length)
                        if len(data) != length:
                            raise ValueError("Native proof changed while packing")
                        digest.update(data)
                        pin = hashlib.sha256(data).hexdigest()
                        chunks.append([pin, length])
                        size += length
                        if pin not in seen:
                            write("chunks/" + pin, data)
                            seen.add(pin)
                if digest.hexdigest() != expected:
                    raise ValueError("Native proof source hash mismatch")
                index["files"][name] = {"sha256": expected, "size_bytes": size, "chunks": chunks}
            write("index.json", json_bytes(index))
        with open(temporary, "r+b") as published:
            published.flush()
            os.fsync(published.fileno())
        digest = file_digest(temporary)
        _safe_local(output)
        try:
            os.link(temporary, output)
        except FileExistsError:
            if file_digest(output) != digest:
                raise ValueError("Retained native proof archive differs")
        sync_directory(output.parent)
        return {"path": output.name, "sha256": digest, "size_bytes": output.stat().st_size,
                "members": {name: {k: value[k] for k in ("sha256", "size_bytes")}
                            for name, value in index["files"].items()}}
    finally:
        Path(temporary).unlink(missing_ok=True)


def _proof_stream(root, name, expected, *, packed=False):
    """Recover exact original bytes, checking the original certificate SHA."""
    root = Path(root).absolute()
    path = _proof_path(root, name)
    if not packed and path.is_file():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Native proof artifact hash mismatch")
        return io.BytesIO(data)
    archive_path = root / (RESUME_ARCHIVE if name.startswith("actual-resume-") else ARCHIVE)
    index = storage_index(archive_path)
    item = index["files"].get(name)
    if item is None:
        raise ValueError("Malformed native proof storage payload: requested member is absent")
    if item["sha256"] != expected:
        raise ValueError("Native proof storage does not bind the requested original")
    if (type(item.get("size_bytes")) is not int or not 0 < item["size_bytes"] <= 8 * 1024**3
            or len(item.get("chunks", [])) > 100000
            or sum(chunk[1] for chunk in item["chunks"]) != item["size_bytes"]):
        raise ValueError("Invalid native proof reconstruction size")
    recovered, digest = io.BytesIO(), hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        for pin, length in item["chunks"]:
            if type(length) is not int or not 0 < length <= 8 * 1024 * 1024:
                raise ValueError("Invalid native proof storage chunk size")
            info = archive.getinfo("chunks/" + pin)
            if info.file_size != length:
                raise ValueError("Native proof storage chunk size mismatch")
            data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != pin:
                raise ValueError("Native proof storage chunk hash mismatch")
            recovered.write(data)
            digest.update(data)
    if recovered.tell() != item["size_bytes"] or digest.hexdigest() != expected:
        raise ValueError("Reconstructed native proof hash mismatch")
    recovered.seek(0)
    return recovered


def proof_stream(root, name, expected, *, packed=False):
    try:
        return _proof_stream(root, name, expected, packed=packed)
    except (zipfile.BadZipFile, KeyError, TypeError, AttributeError, UnicodeError,
            EOFError, OSError, zlib.error) as exc:
        raise ValueError("Malformed native proof storage payload") from exc


def load_proof(root, binding):
    import torch
    if (type(binding) is not dict or set(binding) not in
            ({"path", "sha256"}, {"path", "sha256", "state_fingerprint"},
             {"path", "sha256", "fingerprint"})
            or type(binding["path"]) is not str or type(binding["sha256"]) is not str):
        raise ValueError("Malformed native proof artifact binding")
    with proof_stream(root, binding["path"], binding["sha256"]) as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def compact_certificate(root, certificate, *, actual_resume=False):
    """Publish storage after exact-byte recovery, before source retirement."""
    root = Path(root).absolute()
    _safe_local(root)
    bindings = proof_bindings(certificate)
    filename = RESUME_ARCHIVE if actual_resume else ARCHIVE
    storage = pack_proofs(root, bindings, output=root / filename)
    # Check packed representation itself, not the still-present originals.
    for name, pin in bindings.items():
        with proof_stream(root, name, pin, packed=True):
            pass
    certificate["proof_storage"] = storage
    return certificate


def retire_proof_sources(root, certificate):
    """Retire originals only after the atomically published certificate is intact."""
    storage = certificate.get("proof_storage", {})
    filename = storage.get("path")
    if filename not in {ARCHIVE, RESUME_ARCHIVE}:
        raise ValueError("Proof retirement requires published packed storage")
    certificate_name = "actual_resume_certificate.json" if filename == RESUME_ARCHIVE else "runtime_certificate.json"
    published = _proof_path(root, certificate_name)
    if json.loads(published.read_bytes()) != certificate:
        raise ValueError("Proof retirement requires the atomically published certificate")
    if file_digest(_proof_path(root, filename)) != storage.get("sha256"):
        raise ValueError("Proof retirement requires intact published storage")
    bindings = proof_bindings(certificate)
    if set(storage.get("members", {})) != set(bindings) or any(
            storage["members"][name]["sha256"] != pin for name, pin in bindings.items()):
        raise ValueError("Proof retirement storage bindings differ")
    sources = [_proof_path(root, name) for name in bindings]
    for path, pin in zip(sources, bindings.values()):
        if path.exists() and file_digest(path) != pin:
            raise ValueError("Native proof source changed before retirement")
    for path in sources:
        _safe_local(path)
        path.unlink(missing_ok=True)


def _retained_storage(root, certificate, *, actual_resume=False):
    """Resolve physical dependencies without rewriting an old certificate."""
    root = Path(root).absolute()
    _safe_local(root)
    filename = RESUME_ARCHIVE if actual_resume else ARCHIVE
    bindings = proof_bindings(certificate)
    storage = certificate.get("proof_storage")
    if storage is None:
        if not _proof_path(root, filename).is_file() or all(_proof_path(root, name).is_file() for name in bindings):
            return None
        index = storage_index(root / filename)
        storage = {"path": filename, "sha256": None, "size_bytes": (root / filename).stat().st_size,
                   "members": index["files"]}
    elif (type(storage) is not dict or set(storage) != {"path", "sha256", "size_bytes", "members"}
            or type(storage.get("sha256")) is not str
            or len(storage["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in storage["sha256"])
            or type(storage.get("size_bytes")) is not int or storage["size_bytes"] <= 0
            or type(storage.get("members")) is not dict
            or any(type(item) is not dict or set(item) != {"sha256", "size_bytes"}
                or type(item["sha256"]) is not str or len(item["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in item["sha256"])
                or type(item["size_bytes"]) is not int or item["size_bytes"] <= 0
                for item in storage["members"].values())):
        raise ValueError("Malformed native proof storage binding")
    if (storage.get("path") != filename or set(storage.get("members", {})) != set(bindings)
            or any(storage["members"][name]["sha256"] != pin for name, pin in bindings.items())):
        raise ValueError("Native proof storage differs from original certificate bindings")
    return storage


def retained_storage(root, certificate, *, actual_resume=False):
    try:
        return _retained_storage(root, certificate, actual_resume=actual_resume)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed native proof storage binding") from exc

"""Resolve an explicitly named Kaggle dataset without traversing its payloads.

This stdlib-only module may be embedded byte-for-byte in a prepared bootstrap
before the source package is available. Locations are transport provenance;
callers must still verify their frozen launch, source and data identities.
"""
from pathlib import Path
import re


def _component(value, label):
    if type(value) is not str or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", value) is None:
        raise ValueError("Invalid dataset " + label)
    return value


def _no_link(path):
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError("Dataset mount links are forbidden")


def resolve_dataset_mount(input_root, dataset_source, *, mount_name=None,
                          marker=None, report=None):
    """Select exactly one flat or owner-namespaced dataset root.

    Only two declared locations and an optional direct marker are inspected.
    No directory is enumerated and no file content is read. Duplicate roots
    are rejected even if just one has the marker, rather than selecting a
    convenient identity. ``report`` is populated on failure as well as success.
    """
    if report is not None and type(report) is not dict:
        raise ValueError("Mount discovery report must be a dictionary")
    observation = {} if report is None else report
    observation.clear()
    observation.update(format="molgap-kaggle-dataset-location-v1",
        dataset_source=dataset_source if type(dataset_source) is str else None,
        candidates=[], candidate_count=0, selected_root=None, status="checking",
        recursive_scan=False, file_contents_read=False)
    try:
        if type(dataset_source) is not str or dataset_source.count("/") != 1:
            raise ValueError("Expected explicit owner/dataset source")
        owner, slug = dataset_source.split("/")
        _component(owner, "owner")
        _component(slug, "slug")
        if mount_name is not None and mount_name != slug:
            raise ValueError("Mount name must equal the declared dataset slug")
        if marker is not None and (type(marker) is not str or not marker
                or len(marker) > 128 or marker in {".", ".."}
                or marker.endswith((".", " "))
                or any(ord(c) < 32 or c in '/\\:<>"|?*' for c in marker)):
            raise ValueError("Expected a direct metadata marker filename")
        root = Path(input_root).absolute()
        for ancestor in reversed((root, *root.parents)):
            _no_link(ancestor)
        if not root.is_dir():
            raise ValueError("Dataset input root is absent")
        found = []
        for layout, parts in (("flat", (slug,)),
                              ("owner_namespaced", ("datasets", owner, slug))):
            candidate = root
            present = True
            item = dict(layout=layout, path="/".join(parts), exists=False,
                        marker_present=None)
            observation["candidates"].append(item)
            for part in parts:
                candidate = candidate / part
                _no_link(candidate)
                if not candidate.exists():
                    present = False
                    break
                if not candidate.is_dir():
                    raise ValueError("Dataset location component is not a directory")
            if not present:
                continue
            if not candidate.resolve().is_relative_to(root.resolve()):
                raise ValueError("Dataset mount escaped the input root")
            item["exists"] = True
            found.append((candidate, item))
            observation["candidate_count"] = len(found)
            if marker is not None:
                metadata = candidate / marker
                _no_link(metadata)
                if metadata.exists() and not metadata.is_file():
                    raise ValueError("Dataset metadata marker is not a regular file")
                item["marker_present"] = metadata.is_file()
        if len(found) != 1:
            raise ValueError("Expected exactly one dataset mount; found " + str(len(found)))
        selected, item = found[0]
        if marker is not None and not item["marker_present"]:
            raise ValueError("Declared dataset mount is missing its metadata marker")
        observation.update(status="selected", selected_root=item["path"])
        return selected
    except (OSError, ValueError) as exc:
        observation.update(status="rejected", error_type=type(exc).__name__, error=str(exc))
        raise

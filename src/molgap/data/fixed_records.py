"""Read existing fixed-PCQM record layouts without rebuilding or relabeling data."""
FIXED_KAGGLE_MANIFESTS = frozenset({
    "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
    "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"})


def graph_records(manifest, manifest_sha256):
    """Geometry-packed legacy bytes are permitted only by their exact accepted pin.

    Callers use categorical graph tensors only. Coordinates are incidentally
    deserialized with the trusted file, never selected as model inputs.
    """
    if manifest.get("status") != "complete":
        raise ValueError("Incomplete fixed graph manifest")
    kind = manifest.get("format")
    if kind == "molgap-pcqm4mv2-kaggle-fixed-subset-v1":
        if manifest_sha256 not in FIXED_KAGGLE_MANIFESTS:
            raise ValueError("Legacy packed graphs require an accepted exact manifest pin")
        for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
            if manifest.get(role) is not False:
                raise ValueError("Protected source role was consumed")
        records = manifest.get("geometry_shards")
    elif kind == "molgap-pcqm4mv2-fixed-subset-v1":
        records = manifest.get("assets", {}).get("topology")
    else:
        raise ValueError("Unknown fixed graph manifest format")
    if type(records) is not list or not records:
        raise ValueError("Fixed graph inventory is absent")
    return records


graph_records.__module__ = "molgap.pcqm_fixed_records"

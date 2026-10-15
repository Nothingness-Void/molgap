"""Future-family evidence interface only; no model implementation or registration."""
from __future__ import annotations

from dataclasses import dataclass

from .cross_scale_plan import digest, exact, text, number


@dataclass(frozen=True)
class ModelFidelityEvidence:
    family: str
    revision: str
    paper_version: str
    upstream_commit: str
    license: str
    preprocessing_sha256: str
    graph_sha256: str
    layers_sha256: str
    masks_sha256: str
    positional_encoding_sha256: str
    readout_sha256: str
    fixture_sha256: str
    upstream_output_sha256: str
    local_output_sha256: str
    parameters: int
    atol: float
    rtol: float
    max_absolute_error: float
    max_relative_error: float
    pure_2d: bool
    upstream_executed: bool
    permutation_invariant: bool
    batch_isolated: bool
    backward_finite: bool

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if key.endswith("sha256"):
                digest(value, key)
        for key in ("family", "revision", "paper_version", "license"):
            text(getattr(self, key), key)
        if len(self.upstream_commit) != 40 or any(c not in "0123456789abcdef" for c in self.upstream_commit):
            raise ValueError("Pinned upstream commit required")
        number(self.parameters, "parameters", integer=True, minimum=1)
        for key in ("atol", "rtol", "max_absolute_error", "max_relative_error"):
            number(getattr(self, key), key)
        for key in ("pure_2d", "upstream_executed", "permutation_invariant", "batch_isolated", "backward_finite"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("Fidelity checks must be explicit booleans")

    @classmethod
    def from_dict(cls, value):
        return cls(**exact(value, cls.__dataclass_fields__, "fidelity evidence"))

    def assessment(self):
        checks = {key: getattr(self, key) for key in
                  ("pure_2d", "upstream_executed", "permutation_invariant", "batch_isolated", "backward_finite")}
        checks["numerical_parity"] = self.max_absolute_error <= self.atol and self.max_relative_error <= self.rtol
        return {"status": "FIDELITY_REVIEW_REQUIRED" if all(checks.values()) else "IMPLEMENTATION_UNVERIFIED",
                "checks": checks, "training_authorized": False, "family_registered": False,
                "scientific_status": "NOT_EVALUATED", "native_qualified": False}

"""Raw-state field layout. Transport width is not the network input width."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

SCHEMA_LEGACY33 = "legacy33"
SCHEMA_ANCHOR35 = "anchor35"
FIELD_HASH_LEGACY33 = "goi-state-v1-33"
FIELD_HASH_ANCHOR35 = "goi-state-v1-35"

PLAYER_X, PLAYER_Y = 0, 1
CURSOR_X, CURSOR_Y = 29, 30
ANCHOR_X, ANCHOR_Y = 33, 34
LEGACY_DIM = 33
ANCHOR_DIM = 35


@dataclass(frozen=True)
class StateSchema:
    schema_id: str
    raw_dim: int
    field_hash: str
    has_reference_anchor: bool
    calibration_available: bool = False
    calibration_sha256: str | None = None
    anchor_id: str | None = None

    def assert_raw(self, raw) -> None:
        import numpy as np
        arr = np.asarray(raw)
        if arr.shape[-1] != self.raw_dim:
            raise ValueError(f"raw last axis {arr.shape[-1]} != schema {self.schema_id} dim {self.raw_dim}")


def schema_legacy33() -> StateSchema:
    return StateSchema(SCHEMA_LEGACY33, LEGACY_DIM, FIELD_HASH_LEGACY33, False)


def schema_anchor35(*, calibration_available: bool, calibration_sha256: str | None,
                    anchor_id: str | None) -> StateSchema:
    return StateSchema(
        SCHEMA_ANCHOR35, ANCHOR_DIM, FIELD_HASH_ANCHOR35, True,
        calibration_available=calibration_available,
        calibration_sha256=calibration_sha256,
        anchor_id=anchor_id,
    )


def schema_from_raw_width(width: int) -> StateSchema:
    if width == LEGACY_DIM:
        return schema_legacy33()
    if width == ANCHOR_DIM:
        return schema_anchor35(calibration_available=False, calibration_sha256=None, anchor_id=None)
    raise ValueError(f"Unsupported raw state width {width}; cannot infer schema from column count alone")


def recognize_legacy_file(payload: dict) -> bool:
    """Old files without metadata are legacy only when the recorded format is known."""
    if payload.get("state_schema") or payload.get("reference_fingerprint"):
        return False
    width = payload.get("state_dim")
    return width in (None, LEGACY_DIM)


def fingerprint_payload(values: dict) -> str:
    blob = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

"""Route-comparison point. Scoring only; does not change observation features."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .state_schema import (
    ANCHOR_X, ANCHOR_Y, SCHEMA_ANCHOR35, SCHEMA_LEGACY33, StateSchema, schema_legacy33,
)

ANCHOR_PLAYER_ORIGIN = "player_origin"
ANCHOR_POT_CALIBRATED = "pot_visible_center_v1"
PROFILE_STATUS_UNVERIFIED = "unverified"
PROFILE_STATUS_VERIFIED = "verified"
PROFILE_STATUS_FAILED = "failed"
# Canonical hash of the C# payload without the sha256 field. Must match
# GoiData/anchor_calibration.json and HELLO calibration_sha256.
VERIFIED_CALIBRATION_SHA256 = "94426ea3c49b32ff1b508a1026ba555ba7b2221174fa36da7b251c98dce72459"
VERIFIED_PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "runs/tas_anchor_calib_20260926/anchor_profile_verified.json"
)
VIDEO_REFERENCE_DIR = (
    Path(__file__).resolve().parents[1] / "Data/video_reference/37258923076-1-192"
)
# Terrain-fit camera draft. Kept as the unmodified source.
LEGACY_TERRAIN_FIT_REFERENCE = VIDEO_REFERENCE_DIR / "reference_trajectory.json"
# Intermediate: TAS pixels unprojected with the live spawn camera.
LIVE_CAMERA_REFERENCE = VIDEO_REFERENCE_DIR / "reference_trajectory_live_camera.json"
# Training default: same polyline translated so spawn equals Unity TransformPoint(b).
DEFAULT_REFERENCE_TRAJECTORY = VIDEO_REFERENCE_DIR / "reference_trajectory_game_world.json"


@dataclass(frozen=True)
class ReferenceMetadata:
    anchor_id: str = ANCHOR_PLAYER_ORIGIN
    calibration_sha256: str | None = None
    calibration_verified: bool = False
    source_kind: str = "native"
    route_sha256: str | None = None
    corridor_radius: float | None = None


def load_anchor_profile(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("anchor_id", "local_xy", "body_frame", "units", "status")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Anchor profile missing {missing}")
    if data.get("units") != "game_world_m":
        raise ValueError("Anchor local_xy must be in game_world_m")
    local = data["local_xy"]
    if len(local) != 2 or not all(np.isfinite(float(v)) for v in local):
        raise ValueError("Anchor local_xy must be two finite numbers")
    return data


def profile_is_verified(profile: dict) -> bool:
    return profile.get("status") == PROFILE_STATUS_VERIFIED and bool(profile.get("independent_validation"))


def reference_position(
    raw_states: np.ndarray,
    state_schema: StateSchema | None,
    reference_metadata: ReferenceMetadata | None,
) -> np.ndarray:
    """Return (..., 2) world points used by corridor scoring.

    Old 33D arrays are never silently reconstructed here.
    """
    raw = np.asarray(raw_states, dtype=np.float64)
    if raw.shape[-1] < 2:
        raise ValueError("raw state must contain at least player x/y")
    schema = state_schema or schema_legacy33()
    meta = reference_metadata or ReferenceMetadata()
    schema.assert_raw(raw)
    if meta.anchor_id == ANCHOR_PLAYER_ORIGIN:
        if schema.schema_id not in (SCHEMA_LEGACY33, SCHEMA_ANCHOR35):
            raise ValueError(f"Unsupported schema for player origin: {schema.schema_id}")
        return np.asarray(raw[..., :2], dtype=np.float64)
    if meta.anchor_id == ANCHOR_POT_CALIBRATED:
        if schema.schema_id != SCHEMA_ANCHOR35 or not schema.has_reference_anchor:
            raise ValueError("Calibrated pot anchor requires the anchor35 schema")
        if not meta.calibration_verified:
            raise ValueError("Calibrated pot anchor is not verified; refuse silent fallback")
        if not schema.calibration_available or schema.calibration_sha256 is None:
            raise ValueError("Server did not advertise a usable calibration")
        if meta.calibration_sha256 != schema.calibration_sha256:
            raise ValueError("Reference calibration hash does not match the live schema")
        if schema.anchor_id not in (None, meta.anchor_id):
            raise ValueError(f"Live anchor_id {schema.anchor_id} != {meta.anchor_id}")
        point = np.asarray(raw[..., ANCHOR_X:ANCHOR_Y + 1], dtype=np.float64)
        if not np.isfinite(point).all():
            raise ValueError("Reference anchor coordinates are not finite")
        return point
    raise ValueError(f"Unknown reference anchor_id: {meta.anchor_id}")


def scoring_context(raw, config=None, env=None, route=None):
    from .state_schema import schema_anchor35, schema_from_raw_width
    if env is not None and getattr(env, "state_schema", None) is not None:
        schema = env.state_schema
    else:
        schema = schema_from_raw_width(np.asarray(raw).shape[-1])
    sha = None
    if route is not None:
        sha = (getattr(route, "metadata", None) or {}).get("sha256")
    meta = metadata_from_config(config, sha) if config is not None else ReferenceMetadata()
    # Offline 35D arrays have no HELLO packet. A verified config may advertise
    # the same calibration the live server already applied when the rows were stored.
    if (
        env is None
        and schema.has_reference_anchor
        and meta.anchor_id == ANCHOR_POT_CALIBRATED
        and meta.calibration_verified
        and meta.calibration_sha256
    ):
        schema = schema_anchor35(
            calibration_available=True,
            calibration_sha256=meta.calibration_sha256,
            anchor_id=meta.anchor_id,
        )
    return schema, meta


def metadata_from_config(config, route_sha256: str | None = None) -> ReferenceMetadata:
    return ReferenceMetadata(
        anchor_id=getattr(config, "reference_anchor_id", ANCHOR_PLAYER_ORIGIN),
        calibration_sha256=getattr(config, "reference_anchor_calibration_sha256", None),
        calibration_verified=bool(getattr(config, "reference_anchor_verified", False)),
        source_kind=getattr(config, "reference_anchor_source_kind", "native"),
        route_sha256=route_sha256,
        corridor_radius=getattr(config, "reference_corridor_radius", None),
    )

"""Reference-task fingerprint. Route file hash alone does not identify credited."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state_schema import SCHEMA_LEGACY33, fingerprint_payload, recognize_legacy_file


CREDITED_ALGORITHM = "corridor_tracker_v1"


def reference_fingerprint(
    config,
    *,
    route_meta: dict[str, Any] | None = None,
    schema_id: str | None = None,
    raw_dim: int | None = None,
) -> dict[str, Any]:
    values = {
        "state_schema": schema_id or SCHEMA_LEGACY33,
        "raw_dim": int(raw_dim if raw_dim is not None else getattr(config, "state_dim", 33)),
        "anchor_id": getattr(config, "reference_anchor_id", "player_origin"),
        "calibration_sha256": getattr(config, "reference_anchor_calibration_sha256", None),
        "calibration_verified": bool(getattr(config, "reference_anchor_verified", False)),
        "source_kind": getattr(config, "reference_anchor_source_kind", "native"),
        "route_sha256": (route_meta or {}).get("sha256"),
        "reference_simplify_tolerance": getattr(config, "reference_simplify_tolerance", None),
        "reference_corridor_radius": getattr(config, "reference_corridor_radius", None),
        "reference_progress_weight": getattr(config, "reference_progress_weight", None),
        "reference_deviation_weight": getattr(config, "reference_deviation_weight", None),
        "reference_projection_window": getattr(config, "reference_projection_window", None),
        "reference_max_step": getattr(config, "reference_max_step", None),
        "reference_interpolated_weight": getattr(config, "reference_interpolated_weight", None),
        "credited_algorithm": CREDITED_ALGORITHM,
    }
    payload = dict(values)
    payload["fingerprint"] = fingerprint_payload(values)
    return payload


def sidecar_path(archive_path: str | Path) -> Path:
    path = Path(archive_path)
    return path.with_name(path.name + ".fingerprint.json")


def write_fingerprint_sidecar(archive_path: str | Path, fingerprint: dict[str, Any]) -> Path:
    dest = sidecar_path(archive_path)
    dest.write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def load_fingerprint_sidecar(archive_path: str | Path) -> dict[str, Any] | None:
    dest = sidecar_path(archive_path)
    if not dest.is_file():
        return None
    return json.loads(dest.read_text(encoding="utf-8"))


def recognize_legacy_archive(payload: dict[str, Any] | None, array_width: int | None = None) -> bool:
    if payload and not recognize_legacy_file(payload):
        return False
    if payload is None and array_width == 35:
        return False
    if payload is None:
        return True
    return recognize_legacy_file(payload)


def assert_fingerprint_compatible(expected: dict[str, Any], actual: dict[str, Any] | None, *, context: str) -> None:
    if actual is None:
        if expected.get("anchor_id", "player_origin") != "player_origin":
            raise ValueError(f"{context}: archive has no fingerprint; cannot mix into a calibrated-anchor task")
        return
    if actual.get("fingerprint") != expected.get("fingerprint"):
        raise ValueError(
            f"{context}: reference fingerprint mismatch "
            f"({actual.get('fingerprint')} != {expected.get('fingerprint')})"
        )

"""Re-unproject TAS pot pixels with the live Unity spawn camera.

Keeps the original terrain-fit JSON. Writes a sibling live-camera file.
Pixels stay the TAS detections; only the camera used to unproject changes.
Later frames keep TAS relative camera motion, locked to the live spawn pose.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_video_reference import DEFAULT_OUT, load_map
from video_path_completion import TerrainPenetration, complete_short_gaps, interpolation_holdout

from training.anchor_projection import body_transform_point
from training.reference_anchor import (
    DEFAULT_REFERENCE_TRAJECTORY,
    LIVE_CAMERA_REFERENCE,
    LEGACY_TERRAIN_FIT_REFERENCE,
    VERIFIED_PROFILE_PATH,
    load_anchor_profile,
)

SAMPLES = ROOT / "runs/tas_anchor_calib_20260926/samples.json"
LIVE_POSE_ID = "spawn_upright"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detected_world(sample: dict) -> np.ndarray:
    origin = np.asarray(sample["unity_body_img"], dtype=np.float64)
    axis_x = np.asarray(sample["unity_axis_x_img"], dtype=np.float64) - origin
    axis_y = np.asarray(sample["unity_axis_y_img"], dtype=np.float64) - origin
    local = np.linalg.solve(
        np.column_stack([axis_x, axis_y]),
        np.asarray(sample["detected_px"], dtype=np.float64) - origin,
    )
    return body_transform_point(
        local, sample["position_xy"], sample["rotation_z_deg"], sample["scale_xy"],
    )


def _live_camera() -> dict:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))["samples"]
    upright = next(s for s in samples if s["pose_id"] == LIVE_POSE_ID)
    width = float(upright["pixel_width"])
    height = float(upright["pixel_height"])
    ortho = float(upright["orthographic_size"])
    if (width, height) != (1280.0, 720.0):
        raise ValueError("Live camera lock expects the 1280x720 TAS / live frame size")
    scale = height / (2.0 * ortho)
    return {
        "pose_id": LIVE_POSE_ID,
        "camera_x": float(upright["camera_xy"][0]),
        "camera_y": float(upright["camera_xy"][1]),
        "orthographic_size": ortho,
        "scale_px_per_m": float(scale),
        "pixel_width": width,
        "pixel_height": height,
        "detected_px": [float(upright["detected_px"][0]), float(upright["detected_px"][1])],
        "detected_world": _detected_world(upright).tolist(),
        "samples_sha256": _sha256(SAMPLES),
        "profile_sha256": _sha256(VERIFIED_PROFILE_PATH),
        "anchor_id": load_anchor_profile(VERIFIED_PROFILE_PATH)["anchor_id"],
    }


def _unproject(pixel_x, pixel_y, camera_x, camera_y, scale):
    return (
        camera_x + (pixel_x - 640.0) / scale,
        camera_y - (pixel_y - 360.0) / scale,
    )


def run() -> dict:
    live = _live_camera()
    df = pd.read_csv(DEFAULT_OUT / "trajectory.csv").set_index("frame")
    seed = df.loc[df.index.min()]
    if not bool(seed.camera_candidate):
        raise ValueError("TAS frame 203 has no camera; cannot lock to live spawn")
    scale0 = float(seed.scale_px_per_m)
    cx0 = float(seed.camera_x)
    cy0 = float(seed.camera_y)
    scale_ratio = live["scale_px_per_m"] / scale0
    d_cx = live["camera_x"] - cx0
    d_cy = live["camera_y"] - cy0

    locked = df.copy()
    locked["terrain_camera_x"] = df["camera_x"]
    locked["terrain_camera_y"] = df["camera_y"]
    locked["terrain_scale_px_per_m"] = df["scale_px_per_m"]
    locked["camera_x"] = df["camera_x"] + d_cx
    locked["camera_y"] = df["camera_y"] + d_cy
    locked["scale_px_per_m"] = df["scale_px_per_m"] * scale_ratio

    observed = np.full((len(locked), 2), np.nan, dtype=np.float64)
    ok = locked["player_candidate"].astype(bool) & locked["camera_candidate"].astype(bool)
    wx, wy = _unproject(
        locked.loc[ok, "pixel_x"].to_numpy(float),
        locked.loc[ok, "pixel_y"].to_numpy(float),
        locked.loc[ok, "camera_x"].to_numpy(float),
        locked.loc[ok, "camera_y"].to_numpy(float),
        locked.loc[ok, "scale_px_per_m"].to_numpy(float),
    )
    observed[ok.to_numpy()] = np.column_stack([wx, wy])
    locked["observed_world_x"] = observed[:, 0]
    locked["observed_world_y"] = observed[:, 1]

    _, polygons, _ = load_map()
    completed, sources, gaps = complete_short_gaps(
        locked["time_s"].to_numpy(float), observed, TerrainPenetration(polygons),
    )
    holdout = interpolation_holdout(
        locked["time_s"].to_numpy(float), observed, TerrainPenetration(polygons),
    )
    locked["world_x"] = completed[:, 0]
    locked["world_y"] = completed[:, 1]
    locked["point_source"] = sources
    locked["interpolated"] = sources == "interpolated"
    locked["has_position"] = sources != "missing"
    locked["accepted_candidate"] = ok
    locked["reason"] = np.select(
        [ok, ~locked["camera_candidate"] & ~locked["player_candidate"], ~locked["camera_candidate"]],
        ["live_camera_reprojection", "camera_and_player_unresolved", "camera_unresolved"],
        default="player_unresolved",
    )
    locked.loc[locked["interpolated"], "reason"] = "short_gap_interpolation"

    segment = -1
    previous = -2
    segments = []
    locked["segment_id"] = -1
    for fid, row in locked[locked["has_position"]].iterrows():
        if fid != previous + 1:
            segment += 1
            segments.append([])
        locked.loc[fid, "segment_id"] = segment
        segments[-1].append({
            "frame": int(fid),
            "time_s": float(row.time_s),
            "x": float(row.world_x),
            "y": float(row.world_y),
            "source": str(row.point_source),
        })
        previous = fid

    start = np.array([segments[0][0]["x"], segments[0][0]["y"]], dtype=np.float64)
    live_pot = np.asarray(live["detected_world"], dtype=np.float64)
    old = json.loads(LEGACY_TERRAIN_FIT_REFERENCE.read_text(encoding="utf-8"))
    old_start = np.array([old["segments"][0][0]["x"], old["segments"][0][0]["y"]], dtype=np.float64)

    info = {
        "status": "live_camera_reprojection_v1",
        "training_ready": False,
        "source_video": old["source_video"],
        "source_sha256": old["source_sha256"],
        "map": old["map"],
        "map_sha256": old["map_sha256"],
        "source_terrain_fit_json": str(LEGACY_TERRAIN_FIT_REFERENCE.relative_to(ROOT)),
        "source_terrain_fit_sha256": _sha256(LEGACY_TERRAIN_FIT_REFERENCE),
        "coordinate_system": (
            "world x right, world y up; TAS pot pixels unprojected with live Unity "
            "spawn camera lock, then TAS relative camera motion"
        ),
        "coordinate_warning": (
            "Spawn camera is the live Unity camera at spawn_upright. Later cameras "
            "keep TAS relative motion and zoom; they are not independently live-measured. "
            "Pot pixels remain TAS gold/rigid fusion, not live_dark_contour_v1."
        ),
        "live_camera_lock": {
            **live,
            "tas_seed_frame": int(seed.name),
            "tas_seed_camera_x": cx0,
            "tas_seed_camera_y": cy0,
            "tas_seed_scale_px_per_m": scale0,
            "delta_camera_x": d_cx,
            "delta_camera_y": d_cy,
            "scale_ratio": scale_ratio,
        },
        "actions_available": False,
        "frame_range": [int(locked.index.min()), int(locked.index.max())],
        "frames_evaluated": int(len(locked)),
        "player_candidate_frames": int(locked["player_candidate"].sum()),
        "camera_candidate_frames": int(locked["camera_candidate"].sum()),
        "accepted_candidate_frames": int(ok.sum()),
        "accepted_candidate_fraction": float(ok.mean()),
        "interpolated_frames": int(locked["interpolated"].sum()),
        "completed_frames": int(locked["has_position"].sum()),
        "completed_fraction": float(locked["has_position"].mean()),
        "interpolation_max_gap_s": 0.3,
        "interpolation_method": old["interpolation_method"],
        "interpolation_holdout": holdout,
        "interpolation_gaps": gaps,
        "spawn_check": {
            "tas_start_live_camera": start.tolist(),
            "tas_start_terrain_fit": old_start.tolist(),
            "live_detected_pot": live_pot.tolist(),
            "dist_new_to_live_pot_m": float(np.linalg.norm(start - live_pot)),
            "dist_old_to_live_pot_m": float(np.linalg.norm(old_start - live_pot)),
        },
        "segments": segments,
    }
    dest = LIVE_CAMERA_REFERENCE
    dest.write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    locked.to_csv(DEFAULT_OUT / "trajectory_live_camera.csv", index_label="frame")
    summary = {key: value for key, value in info.items() if key not in ("segments", "interpolation_gaps")}
    (DEFAULT_OUT / "quality_report_live_camera.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _plot(locked, old, start, old_start, live_pot, dest.with_suffix(".png"))
    summary["game_world"] = lock_to_game_world(info)
    return summary


def lock_to_game_world(live_info: dict | None = None) -> dict:
    """Translate the live-camera polyline so spawn equals Unity TransformPoint(b).

    After this step the training reference is a game-world pot path. Camera
    parameters are provenance only; they are not part of the coordinates.
    """
    payload = live_info or json.loads(LIVE_CAMERA_REFERENCE.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))["samples"]
    upright = next(s for s in samples if s["pose_id"] == LIVE_POSE_ID)
    profile = load_anchor_profile(VERIFIED_PROFILE_PATH)
    game_pot = body_transform_point(
        profile["local_xy"], upright["position_xy"], upright["rotation_z_deg"], upright["scale_xy"],
    )
    tas_start = np.array([payload["segments"][0][0]["x"], payload["segments"][0][0]["y"]], dtype=np.float64)
    delta = game_pot - tas_start
    segments = []
    for rows in payload["segments"]:
        segments.append([
            {**point, "x": float(point["x"] + delta[0]), "y": float(point["y"] + delta[1])}
            for point in rows
        ])
    locked_start = np.array([segments[0][0]["x"], segments[0][0]["y"]], dtype=np.float64)
    live_pot = np.asarray(payload["spawn_check"]["live_detected_pot"], dtype=np.float64)
    info = {
        "status": "game_world_spawn_lock_v1",
        "training_ready": False,
        "source_video": payload["source_video"],
        "source_sha256": payload["source_sha256"],
        "map": payload["map"],
        "map_sha256": payload["map_sha256"],
        "source_live_camera_json": str(LIVE_CAMERA_REFERENCE.relative_to(ROOT)),
        "source_live_camera_sha256": _sha256(LIVE_CAMERA_REFERENCE) if LIVE_CAMERA_REFERENCE.exists() else None,
        "source_terrain_fit_json": payload["source_terrain_fit_json"],
        "source_terrain_fit_sha256": payload["source_terrain_fit_sha256"],
        "coordinate_system": "game_world_m; TAS pot path locked to Unity TransformPoint(b) at spawn",
        "coordinate_warning": (
            "Spawn equals the live Player.TransformPoint(b). The remaining path is that "
            "same world translation of the TAS shape. Later points are not independently "
            "sampled from the live game."
        ),
        "game_world_lock": {
            "anchor_id": profile["anchor_id"],
            "body_frame": profile["body_frame"],
            "local_xy": profile["local_xy"],
            "pose_id": LIVE_POSE_ID,
            "player_origin": [float(upright["position_xy"][0]), float(upright["position_xy"][1])],
            "rotation_z_deg": float(upright["rotation_z_deg"]),
            "scale_xy": [float(upright["scale_xy"][0]), float(upright["scale_xy"][1])],
            "spawn_pot_world": game_pot.tolist(),
            "translation_m": delta.tolist(),
            "profile_sha256": _sha256(VERIFIED_PROFILE_PATH),
        },
        "actions_available": False,
        "frame_range": payload["frame_range"],
        "accepted_candidate_frames": payload["accepted_candidate_frames"],
        "completed_frames": payload["completed_frames"],
        "interpolation_max_gap_s": payload["interpolation_max_gap_s"],
        "interpolation_method": payload["interpolation_method"],
        "spawn_check": {
            "tas_start_game_world": locked_start.tolist(),
            "tas_start_live_camera": tas_start.tolist(),
            "unity_transform_point_b": game_pot.tolist(),
            "live_detected_pot": live_pot.tolist(),
            "dist_to_unity_b_m": float(np.linalg.norm(locked_start - game_pot)),
            "dist_to_live_detected_m": float(np.linalg.norm(locked_start - live_pot)),
        },
        "segments": segments,
    }
    DEFAULT_REFERENCE_TRAJECTORY.write_text(
        json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = {key: value for key, value in info.items() if key != "segments"}
    (DEFAULT_OUT / "quality_report_game_world.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def _plot(locked, old, start, old_start, live_pot, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    _, polys, _ = load_map()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4), facecolor="#faf9f6")
    overview, zoom = axes
    lines = [np.vstack([p, p[0]]) for p in polys]
    old_pts = np.array([[p["x"], p["y"]] for seg in old["segments"] for p in seg], dtype=np.float64)
    new_obs = locked[locked["accepted_candidate"]]
    for ax, xlim, ylim in (
        (overview, (-58, 40), (-12, 80)),
        (zoom, (-45.8, -42.6), (-3.7, -1.5)),
    ):
        ax.set_facecolor("#faf9f6")
        ax.add_collection(LineCollection(lines, colors="#c5bdb3", linewidths=0.6))
        ax.plot(old_pts[:, 0], old_pts[:, 1], color="#2f7a4c", lw=1.2, alpha=0.85, label="TAS terrain-fit")
        ax.plot(new_obs.world_x, new_obs.world_y, color="#c0392b", lw=1.2, alpha=0.9, label="TAS live-camera")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.grid(alpha=0.15)
    zoom.plot(old_start[0], old_start[1], "s", color="#1b4332", ms=8, label="old TAS start")
    zoom.plot(start[0], start[1], "D", color="#c0392b", ms=8, label="new TAS start")
    zoom.plot(live_pot[0], live_pot[1], "*", color="#d35400", ms=14, label="live pot")
    overview.set_title("Reprojection vs terrain-fit")
    zoom.set_title("Spawn")
    overview.legend(loc="upper left", fontsize=8, frameon=False)
    zoom.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("TAS pot pixels × live Unity spawn camera")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))

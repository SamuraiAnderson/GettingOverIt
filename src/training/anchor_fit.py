"""Fit a fixed body-local pot anchor. Never marks a profile verified."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .anchor_projection import (
    body_transform_point,
    metres_per_pixel,
    project_orthographic,
    project_unity_local,
)
from .reference_anchor import ANCHOR_POT_CALIBRATED, PROFILE_STATUS_UNVERIFIED

CANDIDATE_P95_M = 0.15


def predict_pixel(local_xy: np.ndarray, sample: dict) -> np.ndarray:
    if sample.get("unity_body_img") is not None:
        return project_unity_local(
            local_xy, sample["unity_body_img"],
            sample["unity_axis_x_img"], sample["unity_axis_y_img"],
        )
    world = body_transform_point(
        local_xy, sample["position_xy"], sample["rotation_z_deg"], sample["scale_xy"],
    )
    return project_orthographic(
        world, sample["camera_xy"], sample["orthographic_size"],
        sample["pixel_width"], sample["pixel_height"],
    )


def evaluate_fit(local_xy, samples: list[dict]) -> dict:
    pixel_err = []
    world_err = []
    for sample in samples:
        pred = predict_pixel(local_xy, sample)
        det = np.asarray(sample["detected_px"], dtype=np.float64)
        e_px = float(np.linalg.norm(pred - det))
        pixel_err.append(e_px)
        if sample.get("unity_body_img") is not None:
            origin = np.asarray(sample["unity_body_img"], dtype=np.float64)
            axis_x = np.asarray(sample["unity_axis_x_img"], dtype=np.float64) - origin
            axis_y = np.asarray(sample["unity_axis_y_img"], dtype=np.float64) - origin
            A = np.column_stack([axis_x, axis_y])
            try:
                local_det = np.linalg.solve(A, det - origin)
            except np.linalg.LinAlgError:
                local_det = np.linalg.pinv(A) @ (det - origin)
            world_pred = body_transform_point(
                local_xy, sample["position_xy"], sample["rotation_z_deg"], sample["scale_xy"],
            )
            world_det = body_transform_point(
                local_det, sample["position_xy"], sample["rotation_z_deg"], sample["scale_xy"],
            )
            world_err.append(float(np.linalg.norm(world_pred - world_det)))
        else:
            mpp = metres_per_pixel(sample["orthographic_size"], sample["pixel_height"])
            world_err.append(e_px * mpp)
    pixel_err = np.asarray(pixel_err)
    world_err = np.asarray(world_err)
    return {
        "local_xy": [float(local_xy[0]), float(local_xy[1])],
        "n": int(len(samples)),
        "pixel_error_median": float(np.median(pixel_err)),
        "pixel_error_p95": float(np.quantile(pixel_err, 0.95)),
        "world_error_median_m": float(np.median(world_err)),
        "world_error_p95_m": float(np.quantile(world_err, 0.95)),
        "meets_candidate_p95_m": bool(np.quantile(world_err, 0.95) <= CANDIDATE_P95_M),
    }


def _pixel_affine(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    origin = predict_pixel(np.zeros(2), sample)
    axis_x = predict_pixel(np.array([1.0, 0.0]), sample) - origin
    axis_y = predict_pixel(np.array([0.0, 1.0]), sample) - origin
    return np.column_stack([axis_x, axis_y]), origin


def fit_local_anchor(samples: list[dict], iterations: int = 8, huber_delta: float = 1.5) -> dict:
    if len(samples) < 4:
        raise ValueError("Need at least 4 independent poses to fit a local anchor")
    affines = [_pixel_affine(sample) for sample in samples]
    detected = [np.asarray(sample["detected_px"], dtype=np.float64) for sample in samples]
    weights = np.ones(len(samples) * 2, dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    for _ in range(iterations):
        rows, rhs = [], []
        for (A, origin), det in zip(affines, detected):
            rows.append(A)
            rhs.append(det - origin)
        design = np.vstack(rows)
        target = np.concatenate(rhs)
        b, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
        residual = design @ b - target
        scale = np.maximum(np.abs(residual) / huber_delta, 1.0)
        weights = 1.0 / scale
    return evaluate_fit(b, samples)


def split_fit_holdout(samples: list[dict], holdout_fraction: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    cut = max(1, int(round(len(samples) * (1.0 - holdout_fraction))))
    cut = min(max(cut, 3), len(samples) - 1)
    fit = [samples[i] for i in order[:cut]]
    hold = [samples[i] for i in order[cut:]]
    return fit, hold


def build_unverified_profile(fit_result: dict, holdout: dict, source_path: str | Path) -> dict:
    raw = Path(source_path).read_bytes() if Path(source_path).is_file() else b""
    return {
        "anchor_id": ANCHOR_POT_CALIBRATED,
        "local_xy": fit_result["local_xy"],
        "body_frame": "Player",
        "units": "game_world_m",
        "status": PROFILE_STATUS_UNVERIFIED,
        "detector_version": "tas_color_track_v1",
        "game_asset": "GettingOverIt/Player",
        "source_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        "independent_validation": holdout,
        "acceptance_target_p95_m": CANDIDATE_P95_M,
        "note": "Not verified. Do not enable official pot-anchor training until holdout passes.",
    }

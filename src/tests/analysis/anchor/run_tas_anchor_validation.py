"""Collect same-frame game captures, detect TAS pot centres, fit and hold out b.

Does not mark a profile verified. A passing candidate P95 is reported as a
measurement, not an authorization for official 35D training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from env.goi_env import GoiEnv
from start import GameLauncher, GameModeController
from training.anchor_fit import (
    CANDIDATE_P95_M,
    build_unverified_profile,
    evaluate_fit,
    fit_local_anchor,
)
from training.config import TrainConfig
from training.anchor_projection import (
    metres_per_pixel,
    project_orthographic,
    unity_screen_axes_to_image,
)
from training.rollout import _warmup_and_snapshot, _write_num_duplicates
from training.tas_pot_detector import LIVE_DETECTOR_VERSION, detect_live_pot_centre

def _agent0(packet: dict) -> dict:
    agents = packet.get("agents") or []
    if not agents:
        raise ValueError("diagnostic packet has no agents")
    return agents[0]


def _sample_from_packet(packet: dict, pose_id: str, split: str, dest_dir: Path) -> dict | None:
    if not packet.get("aligned", False):
        raise ValueError(f"{pose_id}: diagnostic is not aligned")
    agent = _agent0(packet)
    if not agent.get("bodyKnown") or not agent.get("unityScreenKnown") or not agent.get("camKnown"):
        return None
    src = packet.get("screenshotPath") or ""
    if not src or not Path(src).is_file():
        return None
    image_path = dest_dir / "images" / f"{pose_id}_seq{packet['sequence']:06d}.ppm"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    pixel_h = float(packet.get("screenshotHeight") or agent["camPixelH"] or image.shape[0])
    axes = {key: np.asarray(value, dtype=np.float64).tolist()
            for key, value in unity_screen_axes_to_image(agent, pixel_h).items()}
    search_radius = max(40.0, 1.5 / max(metres_per_pixel(agent["camOrthoSize"], pixel_h), 1e-6))
    detection = detect_live_pot_centre(
        image, search_center_px=axes["unity_body_img"], search_radius_px=search_radius,
    )
    ortho_body = project_orthographic(
        [agent["bodyX"], agent["bodyY"]],
        [agent["camPosX"], agent["camPosY"]],
        agent["camOrthoSize"],
        float(packet.get("screenshotWidth") or image.shape[1]),
        pixel_h,
    )
    return {
        "pose_id": pose_id,
        "split": split,
        "sequence": packet.get("sequence"),
        "agent_id": 0,
        "position_xy": [agent["bodyX"], agent["bodyY"]],
        "rotation_z_deg": agent["bodyRz"],
        "scale_xy": [agent["bodySx"], agent["bodySy"]],
        "camera_xy": [agent["camPosX"], agent["camPosY"]],
        "orthographic_size": agent["camOrthoSize"],
        "pixel_width": float(packet.get("screenshotWidth") or image.shape[1]),
        "pixel_height": pixel_h,
        **axes,
        "detected_px": detection.get("detected_px"),
        "detection": detection,
        "image_path": str(image_path),
        "camera_registration_px": float(np.linalg.norm(ortho_body - axes["unity_body_img"])),
        "metres_per_pixel": metres_per_pixel(agent["camOrthoSize"], pixel_h),
    }


def _drive(env: GoiEnv, actions, steps: int) -> None:
    action = np.asarray(actions, dtype=np.float32).reshape(1, 2)
    for _ in range(steps):
        env.step(action)


def _circle(env: GoiEnv, radius: float, cycles: int, quad_steps: int) -> None:
    sequence = ((radius, 0.0), (0.0, radius), (-radius, 0.0), (0.0, -radius))
    for _ in range(cycles):
        for action in sequence:
            _drive(env, action, quad_steps)


def collect(env: GoiEnv, dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    env.toggle_collider_visual(False)
    env.set_camera_free(False)
    samples = []

    def take(pose_id: str, split: str) -> None:
        packet = env.capture_calibration_frame()
        sample = _sample_from_packet(packet, pose_id, split, dest)
        if sample is not None:
            samples.append(sample)

    env.reset()
    take("spawn_upright", "fit")
    _drive(env, (-80.0, 25.0), 28)
    take("spawn_tilt_left", "fit")
    env.reset()
    _drive(env, (80.0, 25.0), 28)
    take("spawn_tilt_right", "fit")
    env.reset()
    _drive(env, (55.0, -50.0), 40)
    take("spawn_plant", "fit")
    env.reset()
    _drive(env, (-90.0, -55.0), 36)
    _drive(env, (80.0, 70.0), 36)
    take("spawn_yank_left", "fit")
    env.reset()
    _drive(env, (90.0, -55.0), 36)
    _drive(env, (-80.0, 70.0), 36)
    take("spawn_yank_right", "fit")
    env.reset()
    _circle(env, 85.0, 3, 10)
    take("spawn_circle", "fit")

    # Stay near spawn. Far TAS teleports put the camera on rock and the pot
    # off-screen. Live pot is dark brown, not the TAS gold appearance.
    nearby = [
        (-44.29, -2.42),
        (-42.5, -1.6),
        (-46.0, -1.9),
        (-43.2, 0.8),
        (-45.4, -1.2),
        (-41.8, -0.2),
    ]
    for i, (x, y) in enumerate(nearby):
        env.reset()
        env.teleport(float(x), float(y), agent_index=0)
        _drive(env, (0.0, 0.0), 8)
        take(f"near_{i:02d}", "holdout")
        if i % 2 == 0:
            _drive(env, (-90.0, -50.0), 30)
            _drive(env, (75.0, 65.0), 30)
        else:
            _circle(env, 80.0, 2, 8)
        take(f"near_{i:02d}_move", "holdout")
    env.reset()
    env.teleport(-44.3, 1.2, agent_index=0)
    _circle(env, 90.0, 3, 12)
    take("air_spin", "holdout")
    env.reset()
    _drive(env, (30.0, -70.0), 35)
    take("spawn_downswing", "holdout")
    env.reset()
    _drive(env, (-90.0, -10.0), 32)
    take("spawn_hard_left", "holdout")
    return samples


def _usable(samples: list[dict]) -> list[dict]:
    return [s for s in samples if s.get("detected_px") is not None and s.get("unity_body_img") is not None]


def residual_table(local_xy, samples: list[dict]) -> list[dict]:
    from training.anchor_fit import predict_pixel
    rows = []
    for sample in samples:
        pred = predict_pixel(local_xy, sample)
        det = np.asarray(sample["detected_px"], dtype=np.float64)
        delta = det - pred
        rows.append({
            "pose_id": sample["pose_id"],
            "split": sample["split"],
            "rotation_z_deg": sample["rotation_z_deg"],
            "pixel_error": float(np.linalg.norm(delta)),
            "pixel_dx": float(delta[0]),
            "pixel_dy": float(delta[1]),
            "camera_registration_px": sample["camera_registration_px"],
        })
    return rows


def write_overlays(local_xy, samples: list[dict], dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    from training.anchor_fit import predict_pixel
    for sample in samples:
        image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        pred = predict_pixel(local_xy, sample)
        det = np.asarray(sample["detected_px"])
        body = np.asarray(sample["unity_body_img"])
        cv2.circle(image, (int(body[0]), int(body[1])), 7, (255, 80, 80), 2)
        cv2.circle(image, (int(det[0]), int(det[1])), 7, (0, 255, 255), 2)
        cv2.circle(image, (int(pred[0]), int(pred[1])), 5, (255, 0, 255), -1)
        cv2.putText(image, sample["pose_id"], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(dest / f"{sample['pose_id']}.png"), image)


def _wrap_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def rotation_span_deg(samples: list[dict]) -> dict:
    angles = [_wrap_deg(s["rotation_z_deg"]) for s in samples]
    if not angles:
        return {"min_deg": None, "max_deg": None, "span_deg": 0.0, "sufficient_for_local_b": False}
    lo, hi = min(angles), max(angles)
    span = hi - lo
    return {
        "min_deg": lo,
        "max_deg": hi,
        "span_deg": span,
        "sufficient_for_local_b": bool(span >= 25.0),
    }


def redetect_samples(samples: list[dict]) -> list[dict]:
    updated = []
    for sample in samples:
        image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            updated.append(sample)
            continue
        search_radius = max(40.0, 1.5 / max(float(sample["metres_per_pixel"]), 1e-6))
        detection = detect_live_pot_centre(
            image, search_center_px=sample["unity_body_img"], search_radius_px=search_radius,
        )
        item = dict(sample)
        item["detected_px"] = detection.get("detected_px")
        item["detection"] = detection
        updated.append(item)
    return updated


def write_profile(fit_result, holdout, samples_path: Path) -> dict:
    return build_unverified_profile(
        fit_result,
        holdout,
        samples_path,
        detector_version=LIVE_DETECTOR_VERSION,
        note="Measured holdout is reported separately. status stays unverified until a reviewer accepts the overlays.",
    )


def write_report(samples: list[dict], dest: Path) -> dict:
    usable = _usable(samples)
    fit_set = [s for s in usable if s["split"] == "fit"]
    hold_set = [s for s in usable if s["split"] == "holdout"]
    span = rotation_span_deg(usable)
    report = {
        "n_captured": len(samples),
        "n_detected": len(usable),
        "n_fit": len(fit_set),
        "n_holdout": len(hold_set),
        "missing_or_uncertain": [s["pose_id"] for s in samples if s.get("detected_px") is None],
        "candidate_p95_m": CANDIDATE_P95_M,
        "detector_version": LIVE_DETECTOR_VERSION,
        "rotation": span,
        "status": "insufficient_samples",
    }
    samples_path = dest / "samples.json"
    if len(fit_set) >= 4 and len(hold_set) >= 3:
        fitted = fit_local_anchor(fit_set)
        holdout = evaluate_fit(fitted["local_xy"], hold_set)
        residuals = residual_table(fitted["local_xy"], usable)
        write_overlays(fitted["local_xy"], usable, dest / "overlays")
        profile = write_profile(fitted, holdout, samples_path)
        (dest / "anchor_profile_unverified.json").write_text(
            json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        by_angle = sorted(residuals, key=lambda row: _wrap_deg(row["rotation_z_deg"]))
        report.update({
            "status": "measured_unverified",
            "fit": fitted,
            "holdout": holdout,
            "meets_candidate_p95_m": holdout["meets_candidate_p95_m"],
            "camera_registration_px_median": float(np.median([s["camera_registration_px"] for s in usable])),
            "camera_registration_px_p95": float(np.quantile([s["camera_registration_px"] for s in usable], 0.95)),
            "residual_vs_angle": by_angle,
            "pose_dependent": bool(np.corrcoef(
                [_wrap_deg(row["rotation_z_deg"]) for row in residuals],
                [row["pixel_error"] for row in residuals],
            )[0, 1] > 0.45) if len(residuals) >= 4 else None,
        })
    (dest / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# TAS 锚点标定测量",
        "",
        f"采集 {report['n_captured']} 帧，检测成功 {report['n_detected']}。",
        f"拟合 {report['n_fit']}，独立 holdout {report['n_holdout']}。",
        f"检测器 `{LIVE_DETECTOR_VERSION}`（实机深色锅轮廓，不是 TAS 金色检测）。",
        f"候选门槛 P95 ≤ {CANDIDATE_P95_M} m，这是门槛不是已有精度。",
        f"身体转角范围 {span['min_deg']:.1f} ~ {span['max_deg']:.1f} deg，跨度 {span['span_deg']:.1f} deg。",
        "",
    ]
    if report.get("holdout"):
        hold = report["holdout"]
        lines += [
            f"拟合局部点 b = {report['fit']['local_xy']}",
            f"Holdout 世界误差 median={hold['world_error_median_m']:.4f} m，P95={hold['world_error_p95_m']:.4f} m。",
            f"是否达到候选门槛: {hold['meets_candidate_p95_m']}",
            f"相机正交近似配准误差 median={report['camera_registration_px_median']:.2f} px。",
            "",
        ]
        if not span["sufficient_for_local_b"]:
            lines.append("转角跨度 < 25°，局部点 b 与世界常值平移仍不可分。本 P95 只描述近直立姿态。")
            lines.append("")
        else:
            lines.append("转角跨度达到 25° 门槛，但未覆盖大角度翻滚；大倾角与 TAS 金色外观未测。")
            lines.append("")
        lines.append("TAS 金色检测在锅贴砂岩时与岩石并块，不能当实机检测器。")
        lines.append("profile 状态保持 unverified。通过门槛也不自动授权正式 35D 训练。")
    else:
        lines.append("有效样本不足，未拟合。")
    if report["missing_or_uncertain"]:
        lines += ["", "未检出/不确定: " + ", ".join(report["missing_or_uncertain"])]
    (dest / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tas_anchor_calib_20260926")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--reprocess", action="store_true", help="Re-detect existing samples.json images")
    args = parser.parse_args()
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)

    if args.reprocess:
        payload = json.loads((dest / "samples.json").read_text(encoding="utf-8"))
        samples = redetect_samples(payload["samples"])
    else:
        game_root = Path(TrainConfig().game_root)
        process = None
        env = None
        for sub in ("images", "overlays"):
            folder = dest / sub
            if folder.exists():
                shutil.rmtree(folder)
        try:
            if not args.no_launch:
                _write_num_duplicates(game_root, 1)
                GameModeController(str(game_root)).set_game_runtime_mode()
                process = GameLauncher().launch(wait=False)
            port = TrainConfig().port if args.port is None else args.port
            env = GoiEnv(port=port, num_agents=1, protocol="legacy33", startup_settle_s=8.0)
            env.connect()
            env.reset()
            _warmup_and_snapshot(env, 1, args.warmup)
            samples = collect(env, dest)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if process is not None:
                try:
                    process.terminate()
                except Exception:
                    pass

    samples_path = dest / "samples.json"
    samples_path.write_text(json.dumps({
        "collected_at": time.time(),
        "pixel_origin": "image_top_left",
        "unity_screen_origin": "bottom_left",
        "conversion": "training.anchor_projection.unity_screen_to_image_xy",
        "detector_version": LIVE_DETECTOR_VERSION,
        "samples": samples,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_report(samples, dest)
    print(json.dumps({k: report[k] for k in report if k != "residual_vs_angle"}, indent=2))


if __name__ == "__main__":
    main()

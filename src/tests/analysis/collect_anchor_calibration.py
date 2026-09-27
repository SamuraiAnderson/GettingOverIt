"""Collect same-frame native pose/camera packets for TAS anchor fitting.

Does not advance physics beyond the caller-supplied teleports. Duplicate agents
in one snapshot are not treated as independent poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from env.goi_env import GoiEnv


def _agent0(packet: dict) -> dict:
    agents = packet.get("agents") or []
    if not agents:
        raise ValueError("diagnostic packet has no agents")
    return agents[0]


def record_sample(env: GoiEnv, pose_id: str) -> dict:
    packet = env.observation_diagnostics()
    if not packet.get("aligned", False):
        raise ValueError("diagnostic packet is not aligned; refuse wall-clock pairing")
    agent = _agent0(packet)
    return {
        "pose_id": pose_id,
        "sequence": packet.get("sequence"),
        "agent_id": 0,
        "collected_at": time.time(),
        "diagnostic": packet,
        "body": {
            "position_xy": [agent.get("bodyX"), agent.get("bodyY")],
            "rotation_z_deg": agent.get("bodyRz"),
            "scale_xy": [agent.get("bodySx"), agent.get("bodySy")],
        },
        "camera": {
            "xy": [agent.get("camPosX"), agent.get("camPosY")],
            "orthographic_size": agent.get("camOrthoSize"),
            "pixel_width": agent.get("camPixelW"),
            "pixel_height": agent.get("camPixelH"),
        },
        "detected_px": None,
        "image_path": None,
        "note": "detected_px/image_path must be filled by the TAS detector on this exact sequence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--poses", type=int, default=8)
    args = parser.parse_args()
    samples = []
    with GoiEnv(port=args.port, num_agents=1, protocol="legacy33", startup_settle_s=0) as env:
        env.reset()
        for i in range(args.poses):
            samples.append(record_sample(env, f"pose_{i:02d}"))
    payload = {
        "status": "unlabeled",
        "samples": samples,
        "pixel_origin": "image_top_left",
        "unity_screen_origin": "bottom_left",
        "conversion": "training.anchor_projection.unity_screen_to_image_xy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(samples)} unlabeled diagnostic samples to {args.output}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from training.anchor_fit import evaluate_fit, fit_local_anchor
from training.anchor_projection import body_transform_point, project_orthographic, project_unity_local
from training.tas_pot_detector import detect_live_pot_centre, detect_pot_centre


def _sample(local, position, rotation, camera=(0.0, 0.0), size=10.0):
    world = body_transform_point(local, position, rotation, (1.0, 1.0))
    detected = project_orthographic(world, camera, size, 320, 180)
    return {
        "position_xy": position,
        "rotation_z_deg": rotation,
        "scale_xy": (1.0, 1.0),
        "camera_xy": camera,
        "orthographic_size": size,
        "pixel_width": 320,
        "pixel_height": 180,
        "detected_px": detected,
    }


class AnchorCalibrationTests(unittest.TestCase):
    def test_transform_point_rotation_and_scale(self):
        np.testing.assert_allclose(
            body_transform_point((1.0, 0.0), (10.0, 20.0), 90.0, (2.0, 1.0)),
            [10.0, 22.0], atol=1e-6,
        )

    def test_fit_recovers_known_local_offset(self):
        true_b = np.array([0.12, -0.35])
        samples = [
            _sample(true_b, (0.0, 0.0), 0.0),
            _sample(true_b, (1.0, 0.5), 20.0),
            _sample(true_b, (-0.5, 1.2), -35.0),
            _sample(true_b, (2.0, -1.0), 80.0),
            _sample(true_b, (0.3, 0.4), 140.0, camera=(0.2, -0.1), size=8.0),
        ]
        result = fit_local_anchor(samples)
        np.testing.assert_allclose(result["local_xy"], true_b, atol=0.03)
        hold = evaluate_fit(result["local_xy"], samples)
        self.assertTrue(hold["meets_candidate_p95_m"])
        self.assertLessEqual(hold["world_error_p95_m"], 0.15)

    def test_unity_axes_recover_local_point(self):
        true_b = np.array([0.18, -0.40])
        samples = []
        for pos, rot in (((0, 0), 0), ((1, 0.4), 25), ((-0.8, 1.1), -40), ((2.2, -0.6), 95)):
            origin = project_orthographic(pos, (0, 0), 9, 320, 180)
            axis_x = project_orthographic(
                body_transform_point((1, 0), pos, rot, (1, 1)), (0, 0), 9, 320, 180)
            axis_y = project_orthographic(
                body_transform_point((0, 1), pos, rot, (1, 1)), (0, 0), 9, 320, 180)
            detected = project_unity_local(true_b, origin, axis_x, axis_y)
            samples.append({
                "position_xy": pos, "rotation_z_deg": rot, "scale_xy": (1, 1),
                "unity_body_img": origin, "unity_axis_x_img": axis_x, "unity_axis_y_img": axis_y,
                "detected_px": detected, "orthographic_size": 9, "pixel_height": 180,
            })
        result = fit_local_anchor(samples)
        np.testing.assert_allclose(result["local_xy"], true_b, atol=1e-3)

    def test_tas_color_detector_finds_gold_blob(self):
        hsv_px = np.uint8([[[28, 200, 180]]])
        bgr = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.ellipse(image, (250, 220), (22, 28), 0, 0, 360, (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
        found = detect_pot_centre(image)
        self.assertEqual(found["reason"], "ok")
        self.assertLess(np.linalg.norm(np.array(found["detected_px"]) - [250, 220]), 6)

    def test_live_dark_detector_finds_dark_blob_and_ignores_tan_rock(self):
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image[:] = (90, 30, 10)
        tan = cv2.cvtColor(np.uint8([[[28, 80, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
        dark = cv2.cvtColor(np.uint8([[[35, 90, 38]]]), cv2.COLOR_HSV2BGR)[0, 0]
        image[280:, :] = (int(tan[0]), int(tan[1]), int(tan[2]))
        cv2.ellipse(image, (250, 240), (22, 26), 0, 0, 360, (int(dark[0]), int(dark[1]), int(dark[2])), -1)
        found = detect_live_pot_centre(image, search_center_px=(250, 220), search_radius_px=80)
        self.assertEqual(found["reason"], "ok")
        self.assertLess(np.linalg.norm(np.array(found["detected_px"]) - [250, 240]), 8)
        far = detect_live_pot_centre(image, search_center_px=(80, 80), search_radius_px=40)
        self.assertIsNone(far["detected_px"])

    def test_unverified_until_reviewer_marks_it(self):
        from training.reference_anchor import PROFILE_STATUS_UNVERIFIED, load_anchor_profile
        import json
        import tempfile
        profile = {
            "anchor_id": "pot_visible_center_v1",
            "local_xy": [0.1, -0.2],
            "body_frame": "Player",
            "units": "game_world_m",
            "status": PROFILE_STATUS_UNVERIFIED,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            loaded = load_anchor_profile(path)
            self.assertEqual(loaded["status"], PROFILE_STATUS_UNVERIFIED)

    def test_verified_profile_is_accepted(self):
        from training.reference_anchor import (
            PROFILE_STATUS_VERIFIED, VERIFIED_CALIBRATION_SHA256,
            VERIFIED_PROFILE_PATH, load_anchor_profile, profile_is_verified,
        )
        loaded = load_anchor_profile(VERIFIED_PROFILE_PATH)
        self.assertEqual(loaded["status"], PROFILE_STATUS_VERIFIED)
        self.assertTrue(profile_is_verified(loaded))
        self.assertEqual(loaded["sha256"], VERIFIED_CALIBRATION_SHA256)


if __name__ == "__main__":
    unittest.main()

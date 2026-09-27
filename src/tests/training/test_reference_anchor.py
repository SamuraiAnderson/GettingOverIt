from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.reference_anchor import (
    ANCHOR_PLAYER_ORIGIN,
    ANCHOR_POT_CALIBRATED,
    VERIFIED_CALIBRATION_SHA256,
    VERIFIED_PROFILE_PATH,
    ReferenceMetadata,
    load_anchor_profile,
    profile_is_verified,
    reference_position,
    scoring_context,
)
from training.state_schema import schema_anchor35, schema_legacy33


class ReferenceAnchorTests(unittest.TestCase):
    def test_player_origin_reads_first_two(self):
        raw = np.array([[1.0, 2.0] + [0.0] * 31], dtype=np.float64)
        point = reference_position(raw, schema_legacy33(), ReferenceMetadata())
        np.testing.assert_allclose(point, [[1.0, 2.0]])

    def test_pot_mode_refuses_unverified_and_33d(self):
        raw33 = np.zeros(33)
        raw35 = np.zeros(35)
        raw35[33:35] = [4.0, 5.0]
        meta = ReferenceMetadata(anchor_id=ANCHOR_POT_CALIBRATED, calibration_verified=False)
        with self.assertRaises(ValueError):
            reference_position(raw33, schema_legacy33(), meta)
        with self.assertRaises(ValueError):
            reference_position(raw35, schema_anchor35(
                calibration_available=True, calibration_sha256="abc",
                anchor_id=ANCHOR_POT_CALIBRATED,
            ), meta)

    def test_pot_mode_reads_33_35_when_verified(self):
        raw = np.zeros(35)
        raw[0:2] = 1.0
        raw[33:35] = [4.5, 6.5]
        schema = schema_anchor35(
            calibration_available=True, calibration_sha256="abc",
            anchor_id=ANCHOR_POT_CALIBRATED,
        )
        meta = ReferenceMetadata(
            anchor_id=ANCHOR_POT_CALIBRATED,
            calibration_verified=True,
            calibration_sha256="abc",
        )
        point = reference_position(raw, schema, meta)
        np.testing.assert_allclose(point, [4.5, 6.5])

    def test_hash_mismatch_and_nonfinite_fail(self):
        schema = schema_anchor35(
            calibration_available=True, calibration_sha256="abc",
            anchor_id=ANCHOR_POT_CALIBRATED,
        )
        meta = ReferenceMetadata(
            anchor_id=ANCHOR_POT_CALIBRATED,
            calibration_verified=True,
            calibration_sha256="xyz",
        )
        with self.assertRaises(ValueError):
            reference_position(np.zeros(35), schema, meta)
        meta = ReferenceMetadata(
            anchor_id=ANCHOR_POT_CALIBRATED,
            calibration_verified=True,
            calibration_sha256="abc",
        )
        raw = np.zeros(35)
        raw[33] = np.nan
        with self.assertRaises(ValueError):
            reference_position(raw, schema, meta)

    def test_player_origin_on_35d_ignores_anchor_columns(self):
        raw = np.zeros(35)
        raw[0:2] = [1.0, 2.0]
        raw[33:35] = [9.0, 8.0]
        point = reference_position(
            raw, schema_anchor35(calibration_available=False, calibration_sha256=None, anchor_id=None),
            ReferenceMetadata(anchor_id=ANCHOR_PLAYER_ORIGIN),
        )
        np.testing.assert_allclose(point, [1.0, 2.0])

    def test_unknown_anchor_id(self):
        with self.assertRaises(ValueError):
            reference_position(np.zeros(33), schema_legacy33(), ReferenceMetadata(anchor_id="nope"))

    def test_offline_scoring_context_uses_verified_config(self):
        raw = np.zeros(35)
        raw[33:35] = [4.5, 6.5]
        schema, meta = scoring_context(raw, TrainConfig())
        self.assertTrue(schema.calibration_available)
        self.assertEqual(schema.calibration_sha256, VERIFIED_CALIBRATION_SHA256)
        np.testing.assert_allclose(reference_position(raw, schema, meta), [4.5, 6.5])

    def test_default_config_is_verified_pot_anchor35(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.tcp_protocol, "anchor35")
        self.assertEqual(cfg.reference_anchor_id, ANCHOR_POT_CALIBRATED)
        self.assertTrue(cfg.reference_anchor_verified)
        self.assertEqual(cfg.reference_anchor_calibration_sha256, VERIFIED_CALIBRATION_SHA256)
        profile = load_anchor_profile(VERIFIED_PROFILE_PATH)
        self.assertTrue(profile_is_verified(profile))
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["sha256"], VERIFIED_CALIBRATION_SHA256)


if __name__ == "__main__":
    unittest.main()

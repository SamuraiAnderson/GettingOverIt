from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.dataset import build_dynamics
from training.state_schema import recognize_legacy_file, schema_from_raw_width, schema_legacy33


class StateSchemaTests(unittest.TestCase):
    def test_width_does_not_invent_calibration(self):
        schema = schema_from_raw_width(35)
        self.assertEqual(schema.schema_id, "anchor35")
        self.assertFalse(schema.calibration_available)
        self.assertIsNone(schema.calibration_sha256)

    def test_unknown_width_rejected(self):
        with self.assertRaises(ValueError):
            schema_from_raw_width(34)

    def test_legacy_file_needs_known_format(self):
        self.assertTrue(recognize_legacy_file({"state_dim": 33}))
        self.assertFalse(recognize_legacy_file({"state_schema": "anchor35"}))
        self.assertFalse(recognize_legacy_file({"reference_fingerprint": {"x": 1}}))

    def test_build_dynamics_33_and_35_match_on_shared_fields(self):
        raw33 = np.linspace(-1, 1, 33, dtype=np.float32)
        raw35 = np.concatenate([raw33, np.array([9.0, -4.0], dtype=np.float32)])
        a = build_dynamics(raw33)
        b = build_dynamics(raw35)
        np.testing.assert_allclose(a, b)
        self.assertEqual(a.shape[-1], 39)

    def test_schema_assert_raw(self):
        schema_legacy33().assert_raw(np.zeros(33))
        with self.assertRaises(ValueError):
            schema_legacy33().assert_raw(np.zeros(35))


if __name__ == "__main__":
    unittest.main()

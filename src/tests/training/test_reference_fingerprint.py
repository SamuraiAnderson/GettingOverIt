from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.reference_fingerprint import (
    assert_fingerprint_compatible,
    load_fingerprint_sidecar,
    recognize_legacy_archive,
    reference_fingerprint,
    write_fingerprint_sidecar,
)
from training.run_logging import load_elites, save_elites


class _Traj:
    def __init__(self):
        self.raw_states = np.zeros((3, 33), dtype=np.float32)
        self.actions = np.zeros((2, 2), dtype=np.float32)
        self.contact = None
        self.score = 1.0
        self.iteration = 0


class ReferenceFingerprintTests(unittest.TestCase):
    def test_route_hash_alone_is_not_enough(self):
        config = TrainConfig(
            reference_anchor_id="player_origin",
            reference_anchor_verified=False,
            reference_anchor_calibration_sha256=None,
            tcp_protocol="legacy33",
        )
        a = reference_fingerprint(config, route_meta={"sha256": "route"}, schema_id="legacy33", raw_dim=33)
        config.reference_anchor_id = "pot_visible_center_v1"
        config.reference_anchor_verified = True
        b = reference_fingerprint(config, route_meta={"sha256": "route"}, schema_id="anchor35", raw_dim=35)
        self.assertEqual(a["route_sha256"], b["route_sha256"])
        self.assertNotEqual(a["fingerprint"], b["fingerprint"])

    def test_legacy_missing_metadata(self):
        self.assertTrue(recognize_legacy_archive(None, 33))
        self.assertFalse(recognize_legacy_archive(None, 35))
        self.assertFalse(recognize_legacy_archive({"state_schema": "anchor35"}, 35))

    def test_sidecar_roundtrip_and_conflict(self):
        config = TrainConfig(
            reference_anchor_id="player_origin",
            reference_anchor_verified=False,
            reference_anchor_calibration_sha256=None,
            tcp_protocol="legacy33",
        )
        fp = reference_fingerprint(config, route_meta={"sha256": "abc"}, schema_id="legacy33", raw_dim=33)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elites_iter_0001.npz"
            save_elites(path, [_Traj()], fingerprint=fp)
            loaded = load_fingerprint_sidecar(path)
            self.assertEqual(loaded["fingerprint"], fp["fingerprint"])
            rows = load_elites(path)
            self.assertEqual(len(rows), 1)
            other = dict(fp)
            other["fingerprint"] = "nope"
            with self.assertRaises(ValueError):
                assert_fingerprint_compatible(fp, other, context="pool")
            assert_fingerprint_compatible(fp, None, context="legacy-seed")
            pot = reference_fingerprint(TrainConfig(), route_meta={"sha256": "abc"}, schema_id="anchor35", raw_dim=35)
            with self.assertRaises(ValueError):
                assert_fingerprint_compatible(pot, None, context="pot-seed")


if __name__ == "__main__":
    unittest.main()

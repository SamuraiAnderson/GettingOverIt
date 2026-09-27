from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from env.protocol import (
    ANCHOR_DIM,
    HELLO_MARKER,
    INTERNAL_STRIDE,
    LEGACY_DIM,
    ProtocolError,
    decode_hello_reply_bytes,
    encode_hello_request,
    pack_legacy_state,
    pack_v2_error,
    pack_v2_state,
    unpack_legacy_state,
    unpack_v2_error,
    unpack_v2_frame,
    unpack_v2_state,
    validate_hello_reply,
)


class ProtocolHelloTests(unittest.TestCase):
    def test_hello_request_is_length_prefixed(self):
        packet = encode_hello_request(anchor_id="pot_visible_center_v1", require_calibration=True)
        self.assertEqual(packet[:1], b"H")
        length = int.from_bytes(packet[1:5], "little")
        self.assertEqual(len(packet), 5 + length)

    def test_hello_rejects_old_server_reply(self):
        with self.assertRaises(ProtocolError):
            decode_hello_reply_bytes(bytes([2]) + b"\x00" * 10)

    def test_hello_requires_calibration_for_new_task(self):
        payload = {
            "accepted": True,
            "protocol": "v2",
            "state_schema": "anchor35",
            "raw_dim": 35,
            "num_agents": 2,
            "calibration_available": False,
        }
        with self.assertRaises(ProtocolError):
            validate_hello_reply(
                payload, expected_schema="anchor35", expected_anchor_id=None,
                expected_calibration_sha256=None, require_calibration=True, num_agents=2,
            )

    def test_legacy_pack_uses_internal_stride(self):
        states = np.zeros((2, 35), dtype=np.float32)
        states[0, 0] = 1.5
        states[1, 0] = 7.5
        states[1, 33] = 99
        dones = np.array([False, True])
        packet = pack_legacy_state(states, dones, internal_stride=INTERNAL_STRIDE)
        out, flags = unpack_legacy_state(packet, LEGACY_DIM, 2)
        self.assertEqual(out.shape, (2, 33))
        self.assertAlmostEqual(float(out[0, 0]), 1.5)
        self.assertAlmostEqual(float(out[1, 0]), 7.5)
        self.assertTrue(flags[1])
        self.assertEqual(out.shape[-1], 33)

    def test_v2_state_and_error_roundtrip(self):
        states = np.arange(70, dtype=np.float32).reshape(2, 35)
        dones = np.array([False, False])
        packet = pack_v2_state(states, dones, sequence=9)
        frame_type, sequence, payload = unpack_v2_frame(packet)
        self.assertEqual(frame_type, 1)
        self.assertEqual(sequence, 9)
        got, flags = unpack_v2_state(payload, ANCHOR_DIM, 2)
        np.testing.assert_array_equal(got, states)
        err = pack_v2_error(3, 1, "reference_anchor_unavailable")
        _t, seq, body = unpack_v2_frame(err)
        code, message = unpack_v2_error(body)
        self.assertEqual(seq, 3)
        self.assertEqual(code, 1)
        self.assertEqual(message, "reference_anchor_unavailable")


if __name__ == "__main__":
    unittest.main()

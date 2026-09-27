"""TCP frame codecs for legacy33 and negotiated anchor35.

Module-level STATE_DIM remains the legacy wire width. Live transport width is
whatever HELLO accepted; this file only packs and unpacks bytes.
"""
from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

HELLO_CMD = b"H"
HELLO_MARKER = 0xFE
V2_STATE = 0x01
V2_ERROR = 0x02
V2_DIAGNOSTIC = 0x03
LEGACY_DIAGNOSTIC = 0xFF

PROTOCOL_V2 = "v2"
SCHEMA_LEGACY33 = "legacy33"
SCHEMA_ANCHOR35 = "anchor35"
FIELD_HASH_LEGACY33 = "goi-state-v1-33"
FIELD_HASH_ANCHOR35 = "goi-state-v1-35"
LEGACY_DIM = 33
ANCHOR_DIM = 35
INTERNAL_STRIDE = 35

HELLO_TIMEOUT_S = 5.0


class ProtocolError(RuntimeError):
    pass


def _u32(n: int) -> bytes:
    return struct.pack("<I", int(n))


def _read_u32(data: bytes, offset: int = 0) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def encode_hello_request(
    *,
    protocol: str = PROTOCOL_V2,
    state_schema: str = SCHEMA_ANCHOR35,
    anchor_id: str | None = None,
    calibration_sha256: str | None = None,
    require_calibration: bool = True,
) -> bytes:
    payload = {
        "protocol": protocol,
        "state_schema": state_schema,
        "anchor_id": anchor_id or "",
        "calibration_sha256": calibration_sha256 or "",
        "require_calibration": bool(require_calibration),
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return HELLO_CMD + _u32(len(blob)) + blob


def decode_hello_reply_bytes(packet: bytes) -> dict[str, Any]:
    if not packet or packet[0] != HELLO_MARKER:
        raise ProtocolError("HELLO reply missing 0xFE marker; old server or unknown version")
    if len(packet) < 5:
        raise ProtocolError("HELLO reply truncated")
    length = _read_u32(packet, 1)
    body = packet[5:5 + length]
    if len(body) != length:
        raise ProtocolError("HELLO reply length mismatch")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"HELLO reply is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("HELLO reply is not an object")
    return payload


def validate_hello_reply(
    payload: dict[str, Any],
    *,
    expected_schema: str,
    expected_anchor_id: str | None,
    expected_calibration_sha256: str | None,
    require_calibration: bool,
    num_agents: int,
) -> dict[str, Any]:
    if payload.get("protocol") != PROTOCOL_V2:
        raise ProtocolError(f"unsupported protocol {payload.get('protocol')!r}")
    if payload.get("state_schema") != expected_schema:
        raise ProtocolError(f"schema {payload.get('state_schema')!r} != {expected_schema!r}")
    raw_dim = int(payload.get("raw_dim", -1))
    if expected_schema == SCHEMA_ANCHOR35 and raw_dim != ANCHOR_DIM:
        raise ProtocolError(f"anchor35 raw_dim {raw_dim} != {ANCHOR_DIM}")
    if expected_schema == SCHEMA_LEGACY33 and raw_dim != LEGACY_DIM:
        raise ProtocolError(f"legacy33 raw_dim {raw_dim} != {LEGACY_DIM}")
    if int(payload.get("num_agents", -1)) != num_agents:
        raise ProtocolError("HELLO num_agents does not match the client")
    if not payload.get("accepted", False):
        raise ProtocolError(f"server rejected HELLO: {payload.get('reject_reason', 'unspecified')}")
    if require_calibration and not payload.get("calibration_available", False):
        raise ProtocolError("new reference task refused to downgrade: calibration unavailable")
    live_anchor = payload.get("anchor_id") or None
    if expected_anchor_id and live_anchor not in (None, "", expected_anchor_id):
        raise ProtocolError(f"live anchor_id {live_anchor!r} != {expected_anchor_id!r}")
    live_hash = payload.get("calibration_sha256") or None
    if expected_calibration_sha256 and live_hash != expected_calibration_sha256:
        raise ProtocolError("HELLO calibration hash does not match the expected profile")
    return payload


def pack_legacy_state(states: np.ndarray, dones: np.ndarray, *, internal_stride: int | None = None) -> bytes:
    """Pack the historical `[n][state][done]...` packet from an internal array."""
    states = np.asarray(states, dtype="<f4")
    dones = np.asarray(dones, dtype=bool)
    n = int(dones.shape[0])
    if n == 0:
        return bytes([0])
    stride = int(internal_stride if internal_stride is not None else states.shape[-1])
    if states.shape[0] != n:
        raise ValueError("states/dones agent count mismatch")
    if states.shape[-1] < LEGACY_DIM and stride == LEGACY_DIM:
        raise ValueError("legacy pack requires at least 33 columns")
    out = bytearray([n])
    for i in range(n):
        row = np.asarray(states[i], dtype="<f4")
        if row.shape[0] != stride and stride == INTERNAL_STRIDE:
            raise ValueError("internal stride 35 required to slice agent rows")
        wire = row[:LEGACY_DIM].astype("<f4", copy=False)
        out += wire.tobytes()
        out += bytes([1 if dones[i] else 0])
    return bytes(out)


def unpack_legacy_state(payload: bytes, state_dim: int, num_agents: int) -> tuple[np.ndarray, np.ndarray]:
    if not payload:
        raise ProtocolError("empty legacy state packet")
    n = payload[0]
    offset = 1
    rows, flags = [], []
    for _ in range(n):
        need = state_dim * 4 + 1
        chunk = payload[offset:offset + need]
        if len(chunk) != need:
            raise ProtocolError("truncated legacy state packet")
        rows.append(np.frombuffer(chunk[:state_dim * 4], dtype="<f4").copy())
        flags.append(bool(chunk[-1]))
        offset += need
    if n != num_agents and n != 0:
        raise ProtocolError(f"legacy agent count {n} != {num_agents}")
    if not rows:
        return np.zeros((0, state_dim), dtype="<f4"), np.zeros((0,), dtype=bool)
    return np.stack(rows), np.asarray(flags, dtype=bool)


def pack_v2_frame(frame_type: int, sequence: int, payload: bytes) -> bytes:
    body = _u32(sequence) + payload
    return bytes([frame_type]) + _u32(len(body)) + body


def pack_v2_state(states: np.ndarray, dones: np.ndarray, sequence: int) -> bytes:
    states = np.asarray(states, dtype="<f4")
    dones = np.asarray(dones, dtype=bool)
    n = int(dones.shape[0])
    payload = bytearray([n])
    for i in range(n):
        payload += np.asarray(states[i], dtype="<f4").tobytes()
        payload += bytes([1 if dones[i] else 0])
    return pack_v2_frame(V2_STATE, sequence, bytes(payload))


def pack_v2_error(sequence: int, code: int, message: str) -> bytes:
    msg = message.encode("utf-8")
    payload = struct.pack("<iH", int(code), len(msg)) + msg
    return pack_v2_frame(V2_ERROR, sequence, payload)


def pack_v2_diagnostic(sequence: int, json_text: str) -> bytes:
    return pack_v2_frame(V2_DIAGNOSTIC, sequence, json_text.encode("utf-8"))


def unpack_v2_frame(packet: bytes) -> tuple[int, int, bytes]:
    if len(packet) < 9:
        raise ProtocolError("v2 frame truncated")
    frame_type = packet[0]
    length = _read_u32(packet, 1)
    body = packet[5:5 + length]
    if len(body) != length:
        raise ProtocolError("v2 payload length mismatch")
    sequence = _read_u32(body, 0)
    return frame_type, sequence, body[4:]


def unpack_v2_state(payload: bytes, state_dim: int, num_agents: int) -> tuple[np.ndarray, np.ndarray]:
    if not payload:
        raise ProtocolError("empty v2 state payload")
    n = payload[0]
    if n != num_agents and n != 0:
        raise ProtocolError(f"v2 agent count {n} != {num_agents}")
    offset = 1
    rows, flags = [], []
    for _ in range(n):
        need = state_dim * 4 + 1
        chunk = payload[offset:offset + need]
        if len(chunk) != need:
            raise ProtocolError("truncated v2 state payload")
        row = np.frombuffer(chunk[:state_dim * 4], dtype="<f4").copy()
        if not np.isfinite(row).all():
            raise ProtocolError("v2 state contains non-finite values")
        rows.append(row)
        flags.append(bool(chunk[-1]))
        offset += need
    if not rows:
        return np.zeros((0, state_dim), dtype="<f4"), np.zeros((0,), dtype=bool)
    return np.stack(rows), np.asarray(flags, dtype=bool)


def unpack_v2_error(payload: bytes) -> tuple[int, str]:
    if len(payload) < 6:
        raise ProtocolError("v2 error payload truncated")
    code, msg_len = struct.unpack_from("<iH", payload, 0)
    message = payload[6:6 + msg_len].decode("utf-8", errors="replace")
    return int(code), message

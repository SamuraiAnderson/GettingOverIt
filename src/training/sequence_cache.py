"""Bounded CPU cache of weight-independent, reset-to-end SIL observations.

Owned by the live sequence encoder, never by a checkpoint or a sampler. Hash
actual input/geometry contents rather than pool indices or ndarray identities.
No neural features, hidden states, values, returns or advantages are retained.
"""
from collections import OrderedDict
import hashlib
import json
import time
from types import MappingProxyType

import numpy as np


OBSERVATION_VERSION = 1
# Dependencies of TrajectoryDataset.encode_sequence/build_patch/build_sparse_geometry.
# Model-side normalization and training/sampling options do not affect these arrays.
OBSERVATION_FIELDS = (
    "patch_mode", "patch_size", "patch_channels", "patch_resolution", "grid_resolution",
    "wide_patch_resolution", "contact_patch_resolution", "body_angle_offset", "body_scale",
    "tip_angle_source", "tip_angle_offset", "tip_scale", "sparse_geometry",
    "sparse_window_meters", "sparse_max_contour_tokens", "sparse_relation_k",
    "feature_position_scale", "feature_velocity_scale", "feature_clip",
    "reference_guidance_sigma", "relation_features", "event_features", "geometry_features",
    "trend_features", "action_segment_steps",
)
TRAJECTORY_INPUTS = ("raw_states", "actions", "contact", "guidance_index", "terminated")


def _hash_array(digest, name, value):
    digest.update(name.encode())
    if value is None:
        digest.update(b"none")
        return
    array = np.ascontiguousarray(value)
    digest.update(str((array.dtype.str, array.shape)).encode())
    if array.size:
        digest.update(memoryview(array).cast("B"))


def trajectory_input_key(trajectory):
    digest = hashlib.sha256()
    for name in TRAJECTORY_INPUTS:
        _hash_array(digest, name, getattr(trajectory, name))
    return digest.hexdigest()


def observation_context_key(encoder, *, uses_efficiency):
    """Include map contents: the archive's reward fingerprint is insufficient."""
    digest = hashlib.sha256(json.dumps({
        "version": OBSERVATION_VERSION,
        "config": {name: getattr(encoder.config, name) for name in OBSERVATION_FIELDS},
        "origin": [encoder._min_gx, encoder._min_gy],
        "uses_efficiency": uses_efficiency,
    }, sort_keys=True, allow_nan=False).encode())
    _hash_array(digest, "terrain", encoder.terrain_mask)
    _hash_array(digest, "efficiency", encoder.efficiency_arr if uses_efficiency else None)
    _hash_array(digest, "tip", encoder.tip_local)
    for name in ("reference_maps", "solid_polygons", "body_local"):
        arrays = getattr(encoder, name)
        digest.update(str((name, None if arrays is None else len(arrays))).encode())
        for index, array in enumerate(arrays or []):
            _hash_array(digest, f"{name}:{index}", array)
    return digest.hexdigest()


class SILObservationCache:
    """LRU byte limit applies to retained NumPy payloads, not build temporaries.

    With limit=0 preserve the former round-local cache; begin_round clears it.
    Geometry/config changes are checked once per synchronous SIL round. The
    encoder's context must not be mutated in the middle of an update.
    """

    def __init__(self):
        self._entries = OrderedDict()
        self._context = None
        self._round_local = {}
        self.bytes = 0
        self.limit_bytes = 0
        self.stats = {}

    def _remove(self, key):
        _, size = self._entries.pop(key)
        self.bytes -= size

    def begin_round(self, encoder, trajectories, limit_bytes):
        if not isinstance(limit_bytes, int) or isinstance(limit_bytes, bool) or limit_bytes < 0:
            raise ValueError("SIL observation cache limit must be a nonnegative integer")
        started = time.perf_counter()
        self.stats = dict(hits=0, misses=0, evictions=0, obsolete=0, invalidations=0,
                          oversized=0, build_seconds=0., lookup_seconds=0.)
        self._round_local.clear()
        self.limit_bytes = limit_bytes
        context = observation_context_key(encoder, uses_efficiency=any(
            t.guidance_index is None for t in trajectories))
        if context != self._context:
            self.stats["invalidations"] = int(self._context is not None)
            self._entries.clear()
            self.bytes = 0
            # The encoder also keeps derived terrain edges/part shapes. Rebuild
            # them when geometry changes, otherwise even a cache miss is stale.
            encoder._sparse_cache = None
            self._context = context
        active = {trajectory_input_key(t) for t in trajectories}
        for key in list(self._entries):
            if key not in active:
                self._remove(key)
                self.stats["obsolete"] += 1
        while self._entries and self.bytes > limit_bytes:
            self._remove(next(iter(self._entries)))
            self.stats["evictions"] += 1
        self.stats["context_seconds"] = time.perf_counter() - started

    def get(self, encoder, trajectory):
        started = time.perf_counter()
        key = trajectory_input_key(trajectory)
        if key in self._entries:
            self._entries.move_to_end(key)
            frames = self._entries[key][0]
            self.stats["hits"] += 1
        elif key in self._round_local:
            frames = self._round_local[key]
            self.stats["hits"] += 1
        else:
            self.stats["lookup_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            arrays = encoder.encode_sequence(trajectory)
            self.stats["build_seconds"] += time.perf_counter() - started
            self.stats["misses"] += 1
            size = sum(array.nbytes for array in arrays.values())
            for array in arrays.values():
                array.setflags(write=False)
            frames = MappingProxyType(arrays)
            if not self.limit_bytes:
                self._round_local[key] = frames
            elif size <= self.limit_bytes:
                while self._entries and self.bytes + size > self.limit_bytes:
                    self._remove(next(iter(self._entries)))
                    self.stats["evictions"] += 1
                self._entries[key] = frames, size
                self.bytes += size
            else:
                self.stats["oversized"] += 1
            return frames
        self.stats["lookup_seconds"] += time.perf_counter() - started
        return frames

    def end_round(self):
        """Release the disabled-cache fallback even when an update raises."""
        self._round_local.clear()

    def metrics(self):
        return {f"gru_sil_cache_{key}": value for key, value in dict(
            self.stats, bytes=self.bytes, entries=len(self._entries),
            limit_bytes=self.limit_bytes).items()}

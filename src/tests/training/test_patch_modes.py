"""dualscale 与 legacy 的 patch 通道，以及两种网络在这两种模式下的前向。"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import DYNAMICS_DIM, build_patch
from training.model import ActionPredictor


def _square(cx, cy, w, h):
    return np.array(
        [[cx - w, cy - h], [cx + w, cy - h], [cx + w, cy + h], [cx - w, cy + h]],
        dtype=np.float64,
    )


def _make_state():
    state = np.zeros(33, dtype=np.float32)
    state[0], state[1] = 10.0, 10.0
    state[5], state[6] = 10.1, 10.9
    state[10], state[11] = 10.0, 10.4
    state[15], state[16] = 11.0, 10.5
    state[19], state[20] = 11.6, 10.6
    state[23], state[24] = 12.2, 10.7
    state[27] = 15.0
    return state


def _geometry():
    terrain = np.zeros((100, 100), dtype=np.float32)
    terrain[:5, :] = 1.0
    efficiency = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    polygons = [_square(11.0, 9.0, 3.0, 1.0), _square(12.2, 10.0, 0.4, 0.4)]
    tip_local = _square(0.0, 0.0, 0.07, 0.22)
    body_local = [_square(0.0, 0.5, 0.3, 0.5), _square(0.0, -0.8, 0.35, 0.14)]
    return terrain, efficiency, polygons, tip_local, body_local


def test_dualscale_channels_and_windows() -> None:
    base = TrainConfig()
    state = _make_state()
    terrain, efficiency, polygons, tip_local, body_local = _geometry()
    config = replace(base, patch_mode="dualscale")
    patch = build_patch(
        state, terrain, efficiency, config, 0, 0,
        solid_polygons=polygons, tip_local=tip_local, body_local=body_local,
    )
    assert patch.shape == (config.patch_channels, config.patch_size, config.patch_size)
    channel_sums = [float(patch[i].sum()) for i in range(patch.shape[0])]
    assert channel_sums[0] > 0
    assert channel_sums[1] > 0
    assert channel_sums[2] > 0
    assert channel_sums[3] > 0
    assert np.isfinite(patch).all()
    assert abs(config.patch_size * config.contact_patch_resolution - 8.64) < 1e-6
    assert abs(config.patch_size * config.wide_patch_resolution - 64.0) < 1e-6


def test_legacy_patch_is_finite() -> None:
    base = TrainConfig()
    terrain, efficiency, _, _, _ = _geometry()
    config = replace(base, patch_mode="legacy")
    patch = build_patch(_make_state(), terrain, efficiency, config, 0, 0)
    assert patch.shape == (config.patch_channels, config.patch_size, config.patch_size)
    assert np.isfinite(patch).all()


def test_both_networks_forward_in_each_mode() -> None:
    torch.manual_seed(0)
    base = TrainConfig()
    batch, steps = 2, base.context_len
    for mode in ("dualscale", "legacy"):
        config = replace(base, patch_mode=mode)
        dynamics = torch.randn(batch, steps, DYNAMICS_DIM)
        patches = torch.randn(
            batch, steps, config.patch_channels, config.patch_size, config.patch_size,
        )
        actions = torch.randn(batch, steps, config.action_dim)
        valid = torch.ones(batch, steps)
        predictor = ActionPredictor(config).eval()
        with torch.no_grad():
            output = predictor(dynamics, patches, actions, valid)
        assert output.shape == (batch, config.action_dim)
        actor = ActorCritic(config).eval()
        with torch.no_grad():
            _distribution, value = actor.forward(dynamics, patches, actions, valid)
        assert value.shape[0] == batch


def main() -> None:
    test_dualscale_channels_and_windows()
    test_legacy_patch_is_finite()
    test_both_networks_forward_in_each_mode()


if __name__ == "__main__":
    main()

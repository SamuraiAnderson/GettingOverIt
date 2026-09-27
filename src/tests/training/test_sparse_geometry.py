"""Sparse contour / contact-relation tokens stay continuous and never rasterize air."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import DYNAMICS_DIM, Trajectory, TrajectoryDataset
from training.sparse_geometry import (
    CLASS_PLAYER,
    CLASS_TERRAIN,
    CLASS_TIP,
    CONTOUR_DIM,
    N_RELATION_PART,
    PART_TIP,
    RELATION_DIM,
    SparseGeometryCache,
    build_sparse_geometry,
    empty_sparse_frame,
    parse_window_batch,
)

_SQUARE = np.array([[-0.08, -0.08], [0.08, -0.08], [0.08, 0.08], [-0.08, 0.08]], dtype=np.float64)
_POT = np.array([[-0.10, -0.20], [0.10, -0.20], [0.10, -0.05], [-0.10, -0.05]], dtype=np.float64)
_NEAR = np.array([[0.25, -0.08], [0.45, -0.08], [0.45, 0.08], [0.25, 0.08]], dtype=np.float64)
_FAR = np.array([[1.40, -0.08], [1.60, -0.08], [1.60, 0.08], [1.40, 0.08]], dtype=np.float64)
_OUTSIDE = np.array([[40.0, 40.0], [41.0, 40.0], [41.0, 41.0], [40.0, 41.0]], dtype=np.float64)


def _cfg(**kwargs) -> TrainConfig:
    return TrainConfig(
        d_model=16,
        nhead=2,
        num_layers=1,
        context_len=4,
        body_angle_offset=0.0,
        tip_angle_offset=0.0,
        sparse_max_contour_tokens=32,
        sparse_relation_k=4,
        **kwargs,
    )


def _state(px=0.0, py=0.0, tip_x=0.0, tip_y=0.0, vx=0.0, vy=0.0, tvx=1.0, tvy=0.0) -> np.ndarray:
    s = np.zeros(33, dtype=np.float32)
    s[0], s[1] = px, py
    s[2], s[3] = vx, vy
    s[5], s[6] = px, py + 0.35
    s[19], s[20] = px + (tip_x - px) * 0.5, py + (tip_y - py) * 0.5
    s[23], s[24] = tip_x, tip_y
    s[25], s[26] = tvx, tvy
    return s


def _cache(polygons, cfg=None):
    cfg = cfg or _cfg()
    return SparseGeometryCache.build(polygons, _SQUARE, _SQUARE, _POT), cfg


def test_default_coarse_map_and_sparse_window_stay_aligned():
    cfg = TrainConfig()
    assert cfg.patch_size == 32
    assert abs(cfg.patch_size * cfg.wide_patch_resolution - 64.0) < 1e-9
    assert abs(cfg.patch_size * cfg.contact_patch_resolution - 8.64) < 1e-9
    assert cfg.sparse_geometry
    assert cfg.sparse_window_cells == 256
    assert abs(cfg.sparse_window_meters - 8.64) < 1e-9


def test_empty_window_emits_no_air_tokens():
    cache, cfg = _cache([])
    contour, cmask, relation, rmask = build_sparse_geometry(_state(), cache, cfg)
    assert contour.shape == (cfg.sparse_max_contour_tokens, CONTOUR_DIM)
    assert relation.shape == (N_RELATION_PART * cfg.sparse_relation_k, RELATION_DIM)
    # 无地形时仍有 player/tip 轮廓；空气位必须全 0
    assert int(cmask.sum()) > 0
    assert int(cmask.sum()) == int((cmask > 0).sum())
    assert np.all(contour[cmask < 0.5] == 0)
    assert np.all(relation[rmask < 0.5] == 0)
    blank = empty_sparse_frame(cfg)
    empty_c, empty_m, _, _ = build_sparse_geometry(_state(), None, cfg)
    np.testing.assert_array_equal(empty_c, blank[0])
    np.testing.assert_array_equal(empty_m, blank[1])


def test_player_and_tip_edges_keep_class_and_world_axes():
    cache, cfg = _cache([_NEAR])
    contour, cmask, _, _ = build_sparse_geometry(
        _state(px=0.0, py=0.0, tip_x=0.2, tip_y=0.0), cache, cfg,
    )
    used = contour[cmask > 0.5]
    classes = used[:, :3].argmax(axis=1)
    assert CLASS_PLAYER in classes
    assert CLASS_TIP in classes
    assert CLASS_TERRAIN in classes
    terrain = used[classes == CLASS_TERRAIN]
    mid = terrain[0, 3:5] * cfg.feature_position_scale
    # 世界轴：近处箱子在 player 右侧，中点相对位移 x>0，不随朝向旋转
    assert mid[0] > 0
    assert used[:, -1].min() == 1.0
    assert int(cmask.sum()) < cfg.sparse_window_cells ** 2


def test_outside_terrain_does_not_enter_window():
    cache, cfg = _cache([_OUTSIDE])
    contour, cmask, _, _ = build_sparse_geometry(_state(), cache, cfg)
    used = contour[cmask > 0.5]
    assert used.size == 0 or CLASS_TERRAIN not in used[:, :3].argmax(axis=1)


def test_relations_keep_k_nearest_sorted_from_collider():
    cache, cfg = _cache([_NEAR, _FAR])
    _, _, relation, rmask = build_sparse_geometry(_state(tip_x=0.0, tip_y=0.0, tvx=2.0), cache, cfg)
    tip = relation[: cfg.sparse_relation_k]
    tip_mask = rmask[: cfg.sparse_relation_k]
    valid = tip[tip_mask > 0.5]
    assert len(valid) >= 2
    dist = valid[:, 5] * cfg.feature_position_scale
    assert np.all(np.diff(dist) >= -1e-6)
    assert dist[0] < dist[-1]
    # 相对速度投影有限，且 known=1
    assert np.isfinite(valid[:, 8:10]).all()
    assert np.allclose(valid[:, -1], 1.0)
    assert np.allclose(valid[:, PART_TIP], 1.0)


def test_cache_reuses_static_edges():
    cache, cfg = _cache([_NEAR])
    assert cache._edges == {}
    a = build_sparse_geometry(_state(), cache, cfg)
    b = build_sparse_geometry(_state(), cache, cfg)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    assert 0 in cache._edges
    # 编码器输出禁止放进 cache：cache 只有静态几何字段
    assert not hasattr(cache, "encoder")
    assert not hasattr(cache, "tokens")


def test_unreached_region_is_not_precomputed():
    cache, cfg = _cache([_NEAR, _OUTSIDE])
    assert cache._edges == {}
    build_sparse_geometry(_state(), cache, cfg)
    assert 0 in cache._edges
    assert 1 not in cache._edges
    far = _state(px=40.5, py=40.5, tip_x=40.7, tip_y=40.5)
    build_sparse_geometry(far, cache, cfg)
    assert 1 in cache._edges


def test_parse_window_batch_accepts_five_and_nine_tuples():
    t = torch.zeros(2, 4, 3)
    five = (t, t, t, t, t)
    dyn, *_rest, ct, cm, rt, rm, w, vt = parse_window_batch(five)
    assert ct is None and w is None and vt is None
    nine = (t, t, t, t, t, torch.zeros(2, 4, 8, 11), torch.zeros(2, 4, 8),
            torch.zeros(2, 4, 12, 11), torch.zeros(2, 4, 12))
    *_, ct, cm, rt, rm, w, vt = parse_window_batch(nine)
    assert ct.ndim == 4 and w is None
    weighted = (*nine, torch.ones(2), torch.zeros(2))
    *_, w, vt = parse_window_batch(weighted)
    assert w.shape == (2,) and vt.shape == (2,)


def test_model_none_matches_zero_residual_at_init():
    cfg = _cfg()
    model = ActorCritic(cfg).eval()
    B, T = 2, cfg.context_len
    dyn = torch.zeros(B, T, DYNAMICS_DIM)
    pat = torch.zeros(B, T, cfg.patch_channels, cfg.patch_size, cfg.patch_size)
    act = torch.zeros(B, T, 2)
    vm = torch.ones(B, T)
    cache, _ = _cache([_NEAR], cfg)
    contour, cmask, relation, rmask = build_sparse_geometry(_state(), cache, cfg)
    ct = torch.from_numpy(np.stack([contour] * T)[None].repeat(B, 0))
    cm = torch.from_numpy(np.stack([cmask] * T)[None].repeat(B, 0))
    rt = torch.from_numpy(np.stack([relation] * T)[None].repeat(B, 0))
    rm = torch.from_numpy(np.stack([rmask] * T)[None].repeat(B, 0))
    with torch.no_grad():
        empty, v0 = model(dyn, pat, act, vm)
        filled, v1 = model(dyn, pat, act, vm, ct, cm, rt, rm)
        # 输出投影零初始化：有 token 时残差仍为 0，与缺省输入同值
        torch.testing.assert_close(empty.mean, filled.mean, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(v0, v1, atol=1e-6, rtol=1e-5)
        model.contour_encoder.proj.weight.fill_(0.05)
        model.contour_encoder.proj.bias.fill_(0.01)
        shifted, _ = model(dyn, pat, act, vm, ct, cm, rt, rm)
        assert not torch.allclose(empty.mean, shifted.mean, atol=1e-5)


def test_dataset_returns_nine_tuple_from_shared_builder():
    cfg = _cfg()
    states = np.zeros((6, 33), dtype=np.float32)
    states[:, 1] = np.linspace(4.0, 6.0, 6)
    states[:, 5] = states[:, 1] + 0.35
    states[:, 23] = 0.2
    states[:, 24] = states[:, 1]
    states[:, 19] = 0.1
    states[:, 20] = states[:, 1]
    traj = Trajectory(raw_states=states, actions=np.zeros((5, 2), dtype=np.float32))
    ds = TrajectoryDataset(
        cfg, np.ones((8, 8), dtype=np.float32), None,
        solid_polygons=[_NEAR], tip_local=_SQUARE, body_local=[_SQUARE, _POT],
    )
    ds.set_trajectories([traj])
    sample = ds[0]
    assert len(sample) == 9
    dyn, pat, act, vm, target, ct, cm, rt, rm = sample
    assert dyn.shape == (cfg.context_len, DYNAMICS_DIM)
    assert pat.shape == (cfg.context_len, cfg.patch_channels, cfg.patch_size, cfg.patch_size)
    assert ct.shape == (cfg.context_len, cfg.sparse_max_contour_tokens, CONTOUR_DIM)
    assert rm.shape == (cfg.context_len, N_RELATION_PART * cfg.sparse_relation_k)
    assert target.shape == (2,)
    # 与原语对拍最后一帧
    raw = build_sparse_geometry(states[0], ds._sparse_geometry_cache(), cfg)
    np.testing.assert_allclose(ct[-1].numpy(), raw[0], atol=1e-6)
    np.testing.assert_allclose(cm[-1].numpy(), raw[1], atol=1e-6)


def test_experimental_collect_stores_sparse_and_trains_encoder():
    from training.experimental_policy import create_actor_critic
    from training.ppo_trainer import PPOTrainer
    from training.rollout import RolloutWorker

    class Env:
        def __init__(self):
            self.index = 0

        def reset(self):
            self.index = 0
            return np.stack([_state(), _state()])

        def step(self, actions):
            self.index += 1
            return np.stack([_state(), _state()]), np.array([False, False])

    cfg = _cfg(
        normalize_action_history=True, num_agents=2,
        ppo_epochs=1, ppo_batch_size=4, ou_enabled=False,
        contact_heads_enabled=False, random_deploy=False, deploy_exclusion_check=False,
    )
    model = create_actor_critic(cfg)
    worker = RolloutWorker(cfg)
    worker.env = Env()
    rng = np.random.default_rng(3)
    worker._ppo_patches = lambda *args: rng.normal(
        0, .1, (2, cfg.patch_channels, cfg.patch_size, cfg.patch_size)
    ).astype(np.float32)
    worker._compute_contact_batch = lambda obs, prev: (np.zeros((2, 5), np.float32), prev)
    blank = empty_sparse_frame(cfg)
    contour, cmask, relation, rmask = [item.copy() for item in blank]
    contour[0, :3] = (1, 0, 0)
    contour[0, 3:5] = (0.2, -0.1)
    contour[0, -1] = 1
    cmask[0] = 1
    relation[0, :3] = (1, 0, 0)
    relation[0, -1] = 1
    rmask[0] = 1
    worker._ppo_sparse = lambda obs: (
        np.stack([contour] * 2), np.stack([cmask] * 2),
        np.stack([relation] * 2), np.stack([rmask] * 2),
    )
    buffer, _ = worker.collect_ppo(model, 8)
    assert "contour_tokens" in buffer.episodes[0]
    assert float(buffer.episodes[0]["contour_mask"].sum()) > 0
    before = model.contour_encoder.proj.weight.detach().clone()
    metrics = PPOTrainer(cfg).update(model, buffer)
    assert metrics["n_updates"] > 0
    assert not torch.equal(before, model.contour_encoder.proj.weight.detach().cpu())


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all sparse geometry tests passed")

"""离线单测：精英池入池前剪枝（prune_trajectory）的口径一致性与边界行为。无需游戏。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset, prune_trajectory
from training.reference_corridor import (
    ReferenceCorridor,
    corridor_cap_index,
    replay_corridor_fields,
)
from training.reward import corridor_score, progress_metric, score_trajectory


def _make_rise_fall_traj(T: int = 200, rise_to: int = 100, peak_y: float = 5.0,
                         with_contact: bool = True) -> Trajectory:
    """合成轨迹：0..rise_to 线性爬到 peak_y，随后急跌并大幅震荡（峰后负面段）。"""
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    y = np.empty(T + 1, dtype=np.float64)
    y[: rise_to + 1] = np.linspace(0.0, peak_y, rise_to + 1)
    tail = np.arange(1, T - rise_to + 1)
    # 急跌 + 震荡；钳制保证峰后不越峰、不触水（water_y_threshold=-10），
    # 使主测试聚焦"口径一致"；越峰/触水的行为变化见专项测试。
    y[rise_to + 1:] = peak_y - 0.8 * tail + 0.3 * np.sin(tail * 0.9)
    y = np.clip(y, -5.0, peak_y)
    raw[:, 1] = y
    raw[:, 0] = np.linspace(0.0, 3.0, T + 1)  # 缓慢右移
    raw[:, 23] = raw[:, 0] + 1.0
    raw[:, 24] = raw[:, 1] + 1.0
    acts = np.random.default_rng(0).normal(size=(T, 2)).astype(np.float32)
    contact = None
    if with_contact:
        contact = np.zeros((T + 1, 5), dtype=np.float32)
        contact[:, 3] = 1.0
    return Trajectory(raw_states=raw, actions=acts, contact=contact)


def _bc_taus(ds_traj: Trajectory, cfg: TrainConfig) -> list[int]:
    """_rebuild_index 产生的 BC 监督 tau 集合（用假 mask，只取索引逻辑）。"""
    ds = TrajectoryDataset(cfg, np.ones((4, 4), dtype=np.float32), None)
    ds.set_trajectories([ds_traj])
    return sorted(tau for _, tau in ds._index)


def test_prune_cut_point_and_shape() -> None:
    """截断点 = peak_idx + 1 + dwell_steps；states/actions/contact 长度对齐。"""
    cfg = TrainConfig()
    traj = _make_rise_fall_traj()
    _, peak_idx, _, _ = progress_metric(traj.raw_states, cfg)
    assert 0 < peak_idx < len(traj.actions) - 1  # 峰在中段，确有峰后负面段

    pruned = prune_trajectory(traj, cfg)
    expected_states = min(peak_idx + 1 + cfg.dwell_steps, len(traj.raw_states))
    assert len(pruned.raw_states) == expected_states
    assert len(pruned.actions) == expected_states - 1
    assert pruned.contact is not None and len(pruned.contact) == expected_states
    # 峰后震荡/跌落段被物理移除
    assert len(pruned.actions) < len(traj.actions)
    # 保留段与原轨迹逐点一致
    np.testing.assert_array_equal(pruned.raw_states, traj.raw_states[:expected_states])
    np.testing.assert_array_equal(pruned.actions, traj.actions[:expected_states - 1])


def test_prune_preserves_metrics() -> None:
    """剪枝后 peak_idx / BC 窗口 / score / bd 与完整轨迹一致（口径不被剪枝改变）。"""
    cfg = TrainConfig()
    traj = _make_rise_fall_traj()
    pruned = prune_trajectory(traj, cfg)

    p_full = progress_metric(traj.raw_states, cfg)
    p_pruned = progress_metric(pruned.raw_states, cfg)
    assert p_pruned[1] == p_full[1]                      # peak_idx 不变
    assert p_pruned[2] == p_full[2]                      # secured_dy 不变

    assert _bc_taus(pruned, cfg) == _bc_taus(traj, cfg)  # BC 窗口集合一致

    s_full = score_trajectory(traj.raw_states, None, cfg)
    s_pruned = score_trajectory(pruned.raw_states, None, cfg)
    assert abs(s_full - s_pruned) < 1e-6                 # score 一致（abs_height 峰在保留窗内）

    ds = TrajectoryDataset(cfg, np.ones((4, 4), dtype=np.float32), None)
    assert ds._behavior_cell_id(pruned.raw_states, 5.0) == \
        ds._behavior_cell_id(traj.raw_states, 5.0)       # MAP-Elites bd 不变


def test_prune_monotonic_rise_is_noop() -> None:
    """持续爬升（peak 在末尾）→ 不剪。"""
    cfg = TrainConfig()
    traj = _make_rise_fall_traj(T=100, rise_to=100)  # 全程上升，无峰后段
    pruned = prune_trajectory(traj, cfg)
    assert len(pruned.raw_states) == len(traj.raw_states)
    assert len(pruned.actions) == len(traj.actions)


def test_prune_flat_traj_keeps_minimal() -> None:
    """无爬升（peak_idx=0）→ 截到 dwell 观察窗长度，不炸且结构合法。"""
    cfg = TrainConfig()
    T = 200
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    raw[:, 1] = 1.0
    raw[:, 23] = 1.0
    raw[:, 24] = 2.0
    traj = Trajectory(raw_states=raw, actions=np.zeros((T, 2), dtype=np.float32))
    pruned = prune_trajectory(traj, cfg)
    assert len(pruned.raw_states) == min(1 + cfg.dwell_steps, T + 1)
    assert len(pruned.actions) == len(pruned.raw_states) - 1
    assert len(pruned.actions) >= 1


def test_prune_short_traj_noop() -> None:
    """短轨迹（不足 peak+dwell 窗）→ 不剪。"""
    cfg = TrainConfig()
    T = 5  # 远小于 dwell_steps
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    raw[:, 1] = np.linspace(0, 1, T + 1)
    traj = Trajectory(raw_states=raw, actions=np.zeros((T, 2), dtype=np.float32))
    pruned = prune_trajectory(traj, cfg)
    assert len(pruned.raw_states) == T + 1
    assert len(pruned.actions) == T


def test_prune_without_contact() -> None:
    """contact=None 的旧轨迹也能剪。"""
    cfg = TrainConfig()
    traj = _make_rise_fall_traj(with_contact=False)
    assert traj.contact is None
    pruned = prune_trajectory(traj, cfg)
    assert pruned.contact is None
    assert len(pruned.actions) == len(pruned.raw_states) - 1


def test_prune_idempotent() -> None:
    """对已剪枝轨迹再剪枝：结果不变（幂等）。"""
    cfg = TrainConfig()
    traj = _make_rise_fall_traj()
    once = prune_trajectory(traj, cfg)
    twice = prune_trajectory(once, cfg)
    assert len(twice.raw_states) == len(once.raw_states)
    assert len(twice.actions) == len(once.actions)
    np.testing.assert_array_equal(twice.raw_states, once.raw_states)
    np.testing.assert_array_equal(twice.actions, once.actions)


def test_prune_drops_fake_post_peak_high() -> None:
    """峰后甩飞出更高瞬时尖峰（守不住）→ 剪枝后 abs_height_bonus 移除该虚假峰，score 下降。

    这是剪枝的期望效果：甩飞段不再经 abs_height_weight 抬高排序分；secured peak 本身
    （held 口径）与 BC 窗口不受影响。
    """
    cfg = TrainConfig()
    T, rise_to = 200, 100
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    y = np.empty(T + 1, dtype=np.float64)
    y[: rise_to + 1] = np.linspace(0.0, 5.0, rise_to + 1)
    tail = np.arange(1, T - rise_to + 1)
    y[rise_to + 1:] = 5.0 - 0.8 * tail + 0.3 * np.sin(tail * 0.9)
    y = np.clip(y, -5.0, np.inf)  # 不触水，聚焦验证 abs_height 虚假峰被剪
    y[rise_to + 40] = 6.5  # 峰后远期甩飞尖峰（远超 dwell 窗，守不住）
    raw[:, 1] = y
    raw[:, 23] = 1.0
    raw[:, 24] = raw[:, 1] + 1.0
    traj = Trajectory(raw_states=raw, actions=np.zeros((T, 2), dtype=np.float32))

    pruned = prune_trajectory(traj, cfg)
    assert pruned.raw_states[:, 1].max() < traj.raw_states[:, 1].max()  # 虚假峰被剪掉
    s_full = score_trajectory(traj.raw_states, None, cfg)
    s_pruned = score_trajectory(pruned.raw_states, None, cfg)
    assert s_pruned < s_full
    # 但 secured peak / BC 窗口仍一致
    assert progress_metric(pruned.raw_states, cfg)[1] == progress_metric(traj.raw_states, cfg)[1]
    assert _bc_taus(pruned, cfg) == _bc_taus(traj, cfg)


def test_prune_revives_water_tail_traj() -> None:
    """峰后段落水 → 完整轨迹 is_water 硬拒（score=-inf）；剪枝掉落水段后恢复合法。

    这是剪枝的期望效果：'甩到顶又落水'的轨迹，其有效推进段仍可入池，
    不再被尾部落水整条废弃。
    """
    cfg = TrainConfig()
    T, rise_to = 200, 100
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    y = np.empty(T + 1, dtype=np.float64)
    y[: rise_to + 1] = np.linspace(0.0, 5.0, rise_to + 1)
    tail = np.arange(1, T - rise_to + 1)
    y[rise_to + 1:] = 5.0 - 0.5 * tail  # 峰后持续下跌，最终 y=-45 触水
    raw[:, 1] = y
    raw[:, 23] = 1.0
    raw[:, 24] = raw[:, 1] + 1.0
    traj = Trajectory(raw_states=raw, actions=np.zeros((T, 2), dtype=np.float32))

    from training.reward import is_water_trajectory
    assert is_water_trajectory(traj.raw_states, cfg)
    assert score_trajectory(traj.raw_states, None, cfg) == float("-inf")

    pruned = prune_trajectory(traj, cfg)
    assert not is_water_trajectory(pruned.raw_states, cfg)
    assert score_trajectory(pruned.raw_states, None, cfg) > float("-inf")


def test_prune_score_and_iteration_preserved() -> None:
    """score / iteration 元数据沿用原轨迹。"""
    cfg = TrainConfig()
    traj = _make_rise_fall_traj()
    traj.score = 7.5
    traj.iteration = 42
    pruned = prune_trajectory(traj, cfg)
    assert pruned.score == 7.5
    assert pruned.iteration == 42


# ───────────────────────── 走廊口径（corridor_progress）─────────────────────────

def _make_corridor_traj(T: int = 200, push_to: int = 100) -> Trajectory:
    """合成走廊轨迹：corridor_progress 前 push_to 步有推进、之后全零（震荡/跌落段）。"""
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    raw[:, 0] = np.linspace(0.0, 8.0, T + 1)
    raw[:, 1] = np.linspace(0.0, 2.0, T + 1)
    raw[:, 23] = raw[:, 0] + 1.0
    raw[:, 24] = raw[:, 1] + 1.0
    acts = np.zeros((T, 2), dtype=np.float32)
    progress = np.zeros(T, dtype=np.float32)
    progress[:push_to] = 8.0 / push_to          # 前 push_to 步均匀推进，总 credited = 8.0
    guidance = np.zeros(T + 1, dtype=np.int8)   # 全程第 0 段
    return Trajectory(raw_states=raw, actions=acts,
                      corridor_progress=progress, guidance_index=guidance)


def test_corridor_cap_index() -> None:
    """cap = credited 累计首次封顶的动作索引；全零轨迹为 0。"""
    progress = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    assert corridor_cap_index(progress) == 1
    assert corridor_cap_index(np.zeros(5, dtype=np.float32)) == 0
    # 封顶后再次推进（震荡中偶发）→ cap 取最后一次非零
    progress2 = np.array([0.5, 0.0, 0.3, 0.0, 0.2], dtype=np.float32)
    assert corridor_cap_index(progress2) == 4


def test_prune_corridor_cut_point() -> None:
    """走廊口径：cut = cap + 2 + dwell；corridor 字段同步切片。"""
    cfg = TrainConfig()
    traj = _make_corridor_traj(T=200, push_to=100)
    cap = corridor_cap_index(traj.corridor_progress)
    assert cap == 99  # 第 99 个动作后累计达到 8.0
    pruned = prune_trajectory(traj, cfg)
    expected_states = min(cap + 2 + cfg.dwell_steps, 201)
    assert len(pruned.raw_states) == expected_states
    assert len(pruned.actions) == expected_states - 1
    assert len(pruned.corridor_progress) == expected_states - 1
    assert len(pruned.guidance_index) == expected_states
    # 震荡段被移除
    assert len(pruned.actions) < 200
    np.testing.assert_array_equal(pruned.corridor_progress, traj.corridor_progress[:expected_states - 1])


def test_corridor_bc_window_covers_push() -> None:
    """走廊口径 BC 截断：τ ≤ cap，监督覆盖全部推进动作（不被高度峰二次截短）。"""
    cfg = TrainConfig()
    traj = _make_corridor_traj(T=200, push_to=100)
    pruned = prune_trajectory(traj, cfg)
    ds = TrajectoryDataset(cfg, np.ones((4, 4), dtype=np.float32), None)
    ds.set_trajectories([pruned])
    taus = sorted(tau for _, tau in ds._index)
    assert max(taus) == min(99, len(pruned.actions) - 1)  # cap=99 全覆盖
    assert len(taus) == max(taus) + 1


def test_corridor_score() -> None:
    """走廊排序分 = credited 总量 + 速度项；零推进得 0；快轨迹优于慢轨迹。"""
    cfg = TrainConfig()
    fast = np.concatenate([np.full(50, 0.16, dtype=np.float32), np.zeros(150, dtype=np.float32)])
    slow = np.concatenate([np.full(100, 0.08, dtype=np.float32), np.zeros(100, dtype=np.float32)])
    assert corridor_score(fast, cfg) > corridor_score(slow, cfg) > 0.0
    assert corridor_score(np.zeros(10, dtype=np.float32), cfg) == 0.0


def test_corridor_behavior_cell() -> None:
    """走廊 bd = credited 封顶点位置，而非高度峰位置。"""
    cfg = TrainConfig(
        reference_anchor_id="player_origin",
        reference_anchor_verified=False,
        reference_anchor_calibration_sha256=None,
        tcp_protocol="legacy33",
    )
    traj = _make_corridor_traj(T=200, push_to=100)
    ds = TrajectoryDataset(cfg, np.ones((4, 4), dtype=np.float32), None)
    cid = ds._behavior_cell_id(traj.raw_states, 5.0, traj.corridor_progress)
    # cap=99 → 状态索引 100 → x = 8.0 * 100/200 = 4.0 → cell (0, ?)
    expected_x = 8.0 * 100 / 200
    assert cid == (int(np.floor(expected_x / 5.0)), int(np.floor(2.0 * 100 / 200 / 5.0)))


def test_corridor_patch_uses_guidance_map() -> None:
    """patch 重建：guidance_index 选 reference_maps 对应段；-1 → wide 全零。"""
    cfg = TrainConfig()
    T = 40
    traj = _make_corridor_traj(T=T, push_to=20)
    traj.guidance_index[10:] = 1    # 中段切到第 1 段
    traj.guidance_index[30:] = -1   # 末段之后无引导
    # 两张可区分的引导场（常数 0.5 / 2.0）
    maps = [np.full((4, 4), 0.5, dtype=np.float32), np.full((4, 4), 2.0, dtype=np.float32)]
    ds = TrajectoryDataset(cfg, np.ones((4, 4), dtype=np.float32), None,
                           reference_maps=maps)
    p0 = ds._build_patch(traj.raw_states[5], int(traj.guidance_index[5]))
    p1 = ds._build_patch(traj.raw_states[15], int(traj.guidance_index[15]))
    p_gap = ds._build_patch(traj.raw_states[35], int(traj.guidance_index[35]))
    # wide 通道 crop 自常数图 → 窗口内均匀为该常数（边界外 0，中心区域非零）
    assert p0[0].max() == 0.5 and p1[0].max() == 2.0
    assert p_gap[0].max() == 0.0
    # __getitem__ 端到端不炸
    ds.set_trajectories([traj])
    sample = ds[0]
    assert sample[1].shape == (cfg.context_len, cfg.patch_channels, cfg.patch_size, cfg.patch_size)


def test_corridor_patch_requires_reference_maps() -> None:
    """轨迹带 guidance_index 但数据集无 reference_maps → 明确报错（不静默用错图）。"""
    cfg = TrainConfig()
    traj = _make_corridor_traj(T=40, push_to=20)
    ds = TrajectoryDataset(cfg, np.ones((4, 4), dtype=np.float32), None)
    ds.set_trajectories([traj])
    try:
        ds[0]
    except ValueError as e:
        assert "reference_maps" in str(e)
    else:
        raise AssertionError("应抛出 ValueError")


def test_replay_corridor_fields_matches_eval_trace() -> None:
    """重放一致性：对真实评估 episode，replay 的 cumsum 应复现 trace 的 credited_progress_m。"""
    ep = _REPO_ROOT / ("runs/corridor_formal_20260922_T_s31/checkpoints/"
                       "eval_midpoint_096k/exploration_001_agent0.json")
    manifest = _REPO_ROOT / ("runs/corridor_formal_20260922_T_s31/checkpoints/"
                             "eval_midpoint_096k/manifest.json")
    ref_path = _REPO_ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json"
    if not (ep.exists() and manifest.exists() and ref_path.exists()):
        return  # 证据文件不在当前检出（CI/新机器）时跳过
    cfg = TrainConfig(**json.loads(manifest.read_text(encoding="utf-8"))["config"])
    route = ReferenceCorridor.from_file(str(ref_path), cfg)
    d = json.loads(ep.read_text(encoding="utf-8"))
    states = np.asarray(d["states"], dtype=np.float32)
    progress, guidance = replay_corridor_fields(route, states)
    trace_credited = np.array([0.0] + [t["credited_progress_m"] for t in d["trace"]])
    replayed = np.concatenate([[0.0], np.cumsum(progress)])
    # tracker 确定性重放：与评估时在线记录逐点一致（浮点累加误差容差 1e-4）
    np.testing.assert_allclose(replayed, trace_credited, atol=1e-4)
    assert progress.shape == (len(states) - 1,)
    assert guidance.shape == (len(states) - 1 + 1,)
    assert guidance.min() >= -1
    # 该 episode credited 封顶于 step 590（1-based）→ cap 动作索引 589
    assert corridor_cap_index(progress) == 589


if __name__ == "__main__":
    test_prune_cut_point_and_shape()
    test_prune_preserves_metrics()
    test_prune_monotonic_rise_is_noop()
    test_prune_flat_traj_keeps_minimal()
    test_prune_short_traj_noop()
    test_prune_without_contact()
    test_prune_idempotent()
    test_prune_drops_fake_post_peak_high()
    test_prune_revives_water_tail_traj()
    test_prune_score_and_iteration_preserved()
    # 走廊口径
    test_corridor_cap_index()
    test_prune_corridor_cut_point()
    test_corridor_bc_window_covers_push()
    test_corridor_score()
    test_corridor_behavior_cell()
    test_corridor_patch_uses_guidance_map()
    test_corridor_patch_requires_reference_maps()
    test_replay_corridor_fields_matches_eval_trace()
    print("all prune_trajectory tests passed")

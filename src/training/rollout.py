"""
数据采集 — 参考 L8 模式与游戏交互。

RolloutWorker:
- 使用 GoiEnv 进行帧级交互
- 支持随机动作采集和模型推理采集
- 存储原始 29D 状态，推理时即时构建 17D dynamics + 4ch patch
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from env import GoiEnv
from start import GameLauncher, GameModeController

if TYPE_CHECKING:
    from .config import TrainConfig
    from .model import ActionPredictor
    from .reward import ClimbingEfficiencyMap

from .dataset import (
    BODY_POS_INDICES,
    DYNAMICS_INDICES,
    HAMMER_POS_INDICES,
    Trajectory,
    crop_centered,
    render_gaussian,
)

logger = logging.getLogger(__name__)


def _load_drop_points(game_root: Path) -> list[list[float]]:
    """从 L7 缓存的 drop_points.json 中加载投放点坐标（固定点回退用）。"""
    import json
    dp_path = game_root / "GoiData" / "Colliders" / "drop_points.json"
    if not dp_path.exists():
        return []
    with open(dp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("drop_points", [])


def _precompute_segment_arcs(
    segments: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """预计算各线段弧长和累积弧长，用于加权随机采样。"""
    arc_lengths = []
    cum_lengths = []
    for seg in segments:
        d = np.diff(seg, axis=0)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
        arc_lengths.append(cum[-1])
        cum_lengths.append(cum)
    return np.array(arc_lengths), cum_lengths


def sample_from_segments(
    segments: list[np.ndarray],
    arc_lengths: np.ndarray,
    cum_lengths: list[np.ndarray],
    n: int,
    drop_height: float,
    rng: np.random.Generator,
) -> list[list[float]]:
    """
    从表面线段上随机采样 n 个投放点。

    按弧长加权选择线段，在线段上均匀采样位置，y 加上 drop_height。
    每次调用产生完全随机的位置，覆盖所有可着陆表面。
    """
    total = arc_lengths.sum()
    if total < 1e-6 or len(segments) == 0:
        return []

    weights = arc_lengths / total
    seg_indices = rng.choice(len(segments), size=n, p=weights)

    points = []
    for si in seg_indices:
        seg = segments[si]
        cum = cum_lengths[si]
        t = rng.uniform(0.0, cum[-1])
        x = float(np.interp(t, cum, seg[:, 0]))
        y = float(np.interp(t, cum, seg[:, 1]))
        points.append([x, y + drop_height])
    return points


def _write_num_duplicates(game_root: Path, n: int) -> None:
    """在 runtime_config.json 中写入 numDuplicates。"""
    import json
    config_path = game_root / "GoiData" / "runtime_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    cfg["numDuplicates"] = n
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _warmup_and_snapshot(env: GoiEnv, num_agents: int, warmup_steps: int) -> None:
    """零动作 warmup 后拍快照。"""
    zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
    for _ in range(warmup_steps):
        env.step(zero_actions)
    env.new_snapshot()
    logger.info("Warmup 完成，快照已拍摄")


def _deploy_agents(
    env: GoiEnv,
    drop_points: list[list[float]],
    settle_steps: int,
    num_agents: int,
) -> np.ndarray:
    """将复制体 agent 传送到对应投放点并稳定。"""
    for i, (x, y) in enumerate(drop_points):
        agent_idx = i + 1
        env.teleport(x, y, agent_index=agent_idx)

    zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
    obs = None
    for _ in range(max(settle_steps, 1)):
        obs, _ = env.step(zero_actions)
    return obs


def _build_patches_batch(
    obs: np.ndarray,
    eff_map: ClimbingEfficiencyMap | None,
    config: TrainConfig,
    min_gx: int,
    min_gy: int,
    terrain_mask: np.ndarray | None,
) -> np.ndarray:
    """批量构建 patches，用于推理时即时生成。"""
    n = obs.shape[0]
    ps = config.patch_size
    ch = config.patch_channels
    patches = np.zeros((n, ch, ps, ps), dtype=np.float32)

    eff_arr = eff_map.get_arr() if eff_map is not None else None

    for i in range(n):
        px, py = float(obs[i, 0]), float(obs[i, 1])

        if terrain_mask is not None:
            patches[i, 0] = crop_centered(
                terrain_mask.astype(np.float32),
                px, py, min_gx, min_gy,
                size=ps, patch_res=config.patch_resolution,
                arr_res=config.grid_resolution,
            )

        if eff_arr is not None:
            patches[i, 1] = crop_centered(
                eff_arr.astype(np.float32),
                px, py, min_gx, min_gy,
                size=ps, patch_res=config.patch_resolution,
                arr_res=config.grid_resolution,
            )

        for bx_idx, by_idx in BODY_POS_INDICES:
            render_gaussian(
                patches[i, 2],
                float(obs[i, bx_idx]), float(obs[i, by_idx]),
                px, py, patch_res=config.patch_resolution,
            )

        for hx_idx, hy_idx in HAMMER_POS_INDICES:
            render_gaussian(
                patches[i, 3],
                float(obs[i, hx_idx]), float(obs[i, hy_idx]),
                px, py, patch_res=config.patch_resolution,
            )

    return patches


def _filter_water_trajectories(
    trajectories: list[Trajectory],
    water_y: float,
) -> list[Trajectory]:
    """过滤落水轨迹：y.min() < water_y 的轨迹被丢弃。"""
    kept = []
    dropped = 0
    for traj in trajectories:
        y_min = float(traj.raw_states[:, 1].min())
        if y_min < water_y:
            dropped += 1
        else:
            kept.append(traj)
    if dropped > 0:
        logger.info("丢弃 %d 条落水轨迹 (y < %.1f), 保留 %d 条",
                     dropped, water_y, len(kept))
    return kept


class RolloutWorker:
    """
    数据采集工作器。

    复用 GoiEnv 和 L8 部署模式进行帧级交互。
    每轮采集从表面线段上随机采样投放位置，最大化空间探索多样性。
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.env: GoiEnv | None = None
        self._terrain_mask: np.ndarray | None = None
        self._min_gx: int = 0
        self._min_gy: int = 0

        # 表面随机采样
        self._segments: list[np.ndarray] = []
        self._arc_lengths: np.ndarray = np.array([])
        self._cum_lengths: list[np.ndarray] = []
        self._drop_height: float = config.drop_height
        self._rng = np.random.default_rng()

        # 固定投放点（回退用）
        self._fixed_drop_points: list[list[float]] = []

    def set_terrain_info(
        self,
        terrain_mask: np.ndarray,
        min_gx: int,
        min_gy: int,
    ) -> None:
        """设置地形信息（从效率图获取）。"""
        self._terrain_mask = terrain_mask
        self._min_gx = min_gx
        self._min_gy = min_gy

    def set_surface_segments(self, segments: list[np.ndarray]) -> None:
        """
        设置可着陆表面线段，启用随机投放位置采样。

        segments: L7 compute_landable_surfaces() 的输出，
                  每个元素 shape (N, 2) 为一条连续表面线段。
        """
        self._segments = segments
        self._arc_lengths, self._cum_lengths = _precompute_segment_arcs(segments)
        total_len = float(self._arc_lengths.sum())
        logger.info(
            "已加载 %d 条表面线段 (总弧长=%.1f), 启用随机投放",
            len(segments), total_len,
        )

    def _sample_positions(self, n: int) -> list[list[float]]:
        """采样 n 个随机投放位置。如果没有表面线段则回退到固定点。"""
        if len(self._segments) > 0:
            return sample_from_segments(
                self._segments, self._arc_lengths, self._cum_lengths,
                n, self._drop_height, self._rng,
            )
        if self._fixed_drop_points:
            indices = self._rng.choice(
                len(self._fixed_drop_points), size=min(n, len(self._fixed_drop_points)),
                replace=False,
            )
            return [self._fixed_drop_points[i] for i in indices]
        return []

    def setup(self) -> None:
        """启动游戏、连接、warmup、拍快照。"""
        game_root = Path(self.config.game_root)

        self._fixed_drop_points = _load_drop_points(game_root)

        _write_num_duplicates(game_root, self.config.num_agents)
        GameModeController(str(game_root)).set_game_runtime_mode()
        GameLauncher().launch(wait=False)

        self.env = GoiEnv(
            port=self.config.port,
            num_agents=self.config.num_agents,
        )
        self.env.connect()
        self.env.reset()

        _warmup_and_snapshot(
            self.env, self.config.num_agents, self.config.warmup_steps
        )

        logger.info("RolloutWorker setup 完成，%d agents 就绪", self.config.num_agents)

    def _reset_and_deploy_random(self) -> np.ndarray:
        """重置环境并将 agent 随机部署到表面上。"""
        num_agents = self.config.num_agents
        obs = self.env.reset()

        if num_agents > 1:
            positions = self._sample_positions(num_agents - 1)
            if positions:
                obs = _deploy_agents(
                    self.env, positions,
                    self.config.settle_steps, num_agents,
                )
        return obs

    def collect_random(self, n_steps: int) -> list[Trajectory]:
        """第 0 轮: 随机动作 + 随机投放位置采集轨迹。"""
        assert self.env is not None, "请先调用 setup()"
        num_agents = self.config.num_agents
        scale = self.config.action_scale

        obs = self._reset_and_deploy_random()

        all_states = [[] for _ in range(num_agents)]
        all_actions = [[] for _ in range(num_agents)]

        for i in range(num_agents):
            all_states[i].append(obs[i].copy())

        for _ in range(n_steps):
            actions = np.random.uniform(-scale, scale, (num_agents, 2)).astype(np.float32)
            obs, _ = self.env.step(actions)
            for i in range(num_agents):
                all_states[i].append(obs[i].copy())
                all_actions[i].append(actions[i].copy())

        trajectories = []
        for i in range(num_agents):
            trajectories.append(Trajectory(
                raw_states=np.array(all_states[i]),
                actions=np.array(all_actions[i]),
            ))
        return _filter_water_trajectories(trajectories, self.config.water_y_threshold)

    def collect_with_model(
        self,
        model: ActionPredictor,
        n_steps: int,
        noise_std: float,
        eff_map: ClimbingEfficiencyMap | None = None,
    ) -> list[Trajectory]:
        """用模型预测 + 高斯噪声 + 随机投放位置采集轨迹。"""
        assert self.env is not None, "请先调用 setup()"
        num_agents = self.config.num_agents
        scale = self.config.action_scale
        ctx = self.config.context_len

        obs = self._reset_and_deploy_random()

        all_states = [[] for _ in range(num_agents)]
        all_actions = [[] for _ in range(num_agents)]
        # 用于模型推理的历史缓冲
        dyn_history = [[] for _ in range(num_agents)]
        act_history = [[] for _ in range(num_agents)]
        patch_history = [[] for _ in range(num_agents)]

        for i in range(num_agents):
            all_states[i].append(obs[i].copy())

        for step in range(n_steps):
            # 即时构建当前 patches
            patches = _build_patches_batch(
                obs, eff_map, self.config,
                self._min_gx, self._min_gy, self._terrain_mask,
            )
            dynamics = obs[:, DYNAMICS_INDICES]  # (num_agents, 17)

            actions = np.zeros((num_agents, 2), dtype=np.float32)
            for i in range(num_agents):
                dyn_history[i].append(dynamics[i].copy())
                patch_history[i].append(patches[i].copy())

                # 构建历史窗口
                hist_len = min(len(dyn_history[i]), ctx)
                dyn_window = np.array(dyn_history[i][-hist_len:])       # (hist_len, 17)
                pat_window = np.array(patch_history[i][-hist_len:])     # (hist_len, 4, 32, 32)

                if len(act_history[i]) == 0:
                    act_window = np.zeros((hist_len, 2), dtype=np.float32)
                else:
                    act_len = min(len(act_history[i]), ctx)
                    act_window = np.array(act_history[i][-act_len:])
                    if len(act_window) < hist_len:
                        pad = np.zeros((hist_len - len(act_window), 2), dtype=np.float32)
                        act_window = np.concatenate([pad, act_window], axis=0)

                # 如果历史不足 ctx，左填充零
                if hist_len < ctx:
                    pad_len = ctx - hist_len
                    dyn_window = np.concatenate(
                        [np.zeros((pad_len, self.config.state_dim), dtype=np.float32), dyn_window]
                    )
                    pat_window = np.concatenate(
                        [np.zeros((pad_len, *pat_window.shape[1:]), dtype=np.float32), pat_window]
                    )
                    act_window = np.concatenate(
                        [np.zeros((pad_len, 2), dtype=np.float32), act_window]
                    )

                pred = model.predict(dyn_window, pat_window, act_window)
                noise = np.random.normal(0, noise_std * scale, size=2).astype(np.float32)
                actions[i] = np.clip(pred + noise, -scale, scale)

            obs, _ = self.env.step(actions)
            for i in range(num_agents):
                all_states[i].append(obs[i].copy())
                all_actions[i].append(actions[i].copy())
                act_history[i].append(actions[i].copy())

        trajectories = []
        for i in range(num_agents):
            trajectories.append(Trajectory(
                raw_states=np.array(all_states[i]),
                actions=np.array(all_actions[i]),
            ))
        return _filter_water_trajectories(trajectories, self.config.water_y_threshold)

    def teardown(self) -> None:
        """关闭连接。"""
        if self.env is not None:
            self.env.close()
            self.env = None
            logger.info("RolloutWorker 已关闭")

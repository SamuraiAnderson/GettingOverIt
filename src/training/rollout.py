"""
数据采集 — 参考 L8 模式与游戏交互。

RolloutWorker:
- 使用 GoiEnv 进行帧级交互
- 支持随机动作采集、BC 模型推理采集和 PPO 采集
- 存储原始 33D 状态，推理时即时构建动力学特征 + 4ch patch + valid_mask
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from env import GoiEnv, make_goi_env
from start import GameLauncher, GameModeController

if TYPE_CHECKING:
    from .actor_critic import ActorCritic
    from .config import TrainConfig
    from .model import ActionPredictor
    from .ppo_buffer import PPORolloutBuffer
    from .reward import ClimbingEfficiencyMap

from .contact_features import (
    CONTACT_DIM,
    ContactConfig,
    ContactPrevState,
    TerrainContactIndex,
    compute_contact_row,
    split_body_pot_local,
)
from .dataset import (
    Trajectory,
    build_dynamics,
    build_patch,
    left_pad_sequence,
)
from .reference_corridor import CorridorTracker, ReferenceCorridor
from .deploy_sampling import (
    default_candidate_points_path,
    default_exclusion_zones_path,
    extract_body_local,
    extract_tip_local,
    filter_points_by_exclusion,
    load_candidate_points,
    load_exclusion_zones,
    make_tip_recon_config,
    point_in_exclusion,
    precompute_segment_arcs,
    sample_from_fixed_pool,
    sample_from_segments,
    tip_in_obstacle,
)

logger = logging.getLogger(__name__)


def _load_drop_points(game_root: Path) -> list[list[float]]:
    """加载候选点集（L7 → train 的唯一文件：项目 checkpoints/drop_points_all_stable.json）。

    game_root 保留为兼容旧签名；候选文件现固定在项目 checkpoints/，与游戏目录解耦。
    """
    _ = game_root
    path = default_candidate_points_path()
    pts = load_candidate_points(path)
    if pts:
        logger.info("候选点集: 从 %s 加载 %d 个点", path, len(pts))
    else:
        logger.warning(
            "候选点集为空或缺失: %s（请先运行 test_l7_surface_airdrop.py --physics-filter-all 生成）",
            path,
        )
    return pts


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
    solid_polygons: list[np.ndarray] | None = None,
    tip_local: np.ndarray | None = None,
    body_local: list[np.ndarray] | None = None,
    guidance_arrays: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """批量构建 patches，逐样本委托 dataset.build_patch，保证与离线数据集完全一致。"""
    n = obs.shape[0]
    ps = config.patch_size
    ch = config.patch_channels
    patches = np.zeros((n, ch, ps, ps), dtype=np.float32)

    eff_arr = eff_map.get_arr() if eff_map is not None else None

    for i in range(n):
        patches[i] = build_patch(
            obs[i], terrain_mask,
            guidance_arrays.get(i) if guidance_arrays is not None else eff_arr,
            config, min_gx, min_gy,
            solid_polygons=solid_polygons, tip_local=tip_local, body_local=body_local,
        )

    return patches


def _filter_water_trajectories(
    trajectories: list[Trajectory],
    config: TrainConfig,
) -> list[Trajectory]:
    """过滤落水轨迹：复用 reward.is_water_trajectory 的统一阈值判定，丢弃触水轨迹。"""
    from .reward import is_water_trajectory

    kept = [t for t in trajectories if not is_water_trajectory(t.raw_states, config)]
    dropped = len(trajectories) - len(kept)
    if dropped > 0:
        logger.info("丢弃 %d 条落水轨迹 (y < %.1f), 保留 %d 条",
                     dropped, config.water_y_threshold, len(kept))
    return kept


def _build_history_window(
    dyn_history: list[np.ndarray],
    patch_history: list[np.ndarray],
    act_history: list[np.ndarray],
    ctx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从累积历史中构建固定长度的窗口（左填充零）。

    返回 (dyn_window, pat_window, act_window, valid_mask)：
    - dyn_window: (ctx, state_dim)
    - pat_window: (ctx, C, H, H)
    - act_window: (ctx, 2)
    - valid_mask: (ctx,) —— 1=有效观测，0=左填充位（供 key_padding_mask 使用）

    调用时约定 dyn_history/patch_history 含当前步、act_history 只到上一步，
    因此三者各自右对齐后经 left_pad_sequence 左填充；valid_mask 由观测（dyn）驱动。
    填充/掩码逻辑与离线 __getitem__ 共用 left_pad_sequence，保证逐窗口一致。
    """
    dyn_window, valid_mask = left_pad_sequence(np.asarray(dyn_history[-ctx:]), ctx)
    pat_window, _ = left_pad_sequence(np.asarray(patch_history[-ctx:]), ctx)

    if len(act_history) == 0:
        act_real = np.zeros((0, 2), dtype=np.float32)
    else:
        act_real = np.asarray(act_history[-ctx:], dtype=np.float32)
    act_window, _ = left_pad_sequence(act_real, ctx)

    return dyn_window, pat_window, act_window, valid_mask


def _pad_sparse_history(contour_h, mask_h, rel_h, rmask_h, ctx):
    if not contour_h:
        raise ValueError("sparse history is empty; append the current frame before padding")
    contour, _ = left_pad_sequence(np.asarray(contour_h[-ctx:]), ctx)
    mask, _ = left_pad_sequence(np.asarray(mask_h[-ctx:]), ctx)
    relation, _ = left_pad_sequence(np.asarray(rel_h[-ctx:]), ctx)
    rmask, _ = left_pad_sequence(np.asarray(rmask_h[-ctx:]), ctx)
    return contour, mask, relation, rmask


class RolloutWorker:
    """
    数据采集工作器。

    复用 GoiEnv 和 L8 部署模式进行帧级交互。
    每轮采集从表面线段上随机采样投放位置，最大化空间探索多样性。
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.env: GoiEnv | None = None
        self._game_process: subprocess.Popen | None = None
        self._terrain_mask: np.ndarray | None = None
        self._min_gx: int = 0
        self._min_gy: int = 0
        self.reference_corridor = (
            ReferenceCorridor.from_file(config.reference_trajectory, config)
            if config.reference_trajectory else None
        )
        self._reference_maps: list[np.ndarray] | None = None
        self.last_corridor_metrics: dict[str, float] = {}

        # 表面随机采样
        self._segments: list[np.ndarray] = []
        self._arc_lengths: np.ndarray = np.array([])
        self._cum_lengths: list[np.ndarray] = []
        self._drop_height: float = config.drop_height
        self._rng = np.random.default_rng()

        # 固定投放点（回退用）
        self._fixed_drop_points: list[list[float]] = []

        # 人工涂抹排除区（圆形），落点落入即剔除；None=不过滤
        # 在 __init__ 即加载：main_ppo 直接走 launch_game() 不经 setup()，须保证任何入口都生效
        self._exclusion_zones: np.ndarray | None = None
        self._load_exclusion_zones()

        # 运行时锤头卡障碍过滤（与 L7 离线筛选同一判据）
        self._solid_polygons: list[np.ndarray] | None = None
        self._contact_polygons: TerrainContactIndex | None = None
        self._tip_cfg = None
        # dualscale patch 接触分支几何 & 接触信号
        # （tip_contact/tip_grip/body_contact/pot_contact/fall_distance）
        # 复刻游戏原生 HammerCollisions + PlayerSounds 判断，详见 contact_features.py。
        # `_body_local` 保留 [body, pot] 合并 list 供 patch dualscale 接触通道使用；
        # `_body_only_local` / `_pot_only_local` 是按名称拆分后的独立多边形，
        # 用于 contact_features 分别输出 body_contact / pot_contact（对应游戏 IL 里
        # `rb.GetPoint().y > 0.4` vs `≤ 0.4` 分层）。
        self._tip_local: np.ndarray | None = None
        self._body_local: list[np.ndarray] | None = None
        self._body_only_local: np.ndarray | None = None
        self._pot_only_local: np.ndarray | None = None
        self._sparse_cache = None
        self._contact_cfg: ContactConfig = ContactConfig(
            tip_angle_source=self.config.tip_angle_source,
            tip_angle_offset_deg=self.config.tip_angle_offset,
            tip_scale=self.config.tip_scale,
            body_angle_offset_deg=self.config.body_angle_offset,
            body_scale=self.config.body_scale,
        )
        self._feature_surface_index = None

    def set_terrain_info(
        self,
        terrain_mask: np.ndarray,
        min_gx: int,
        min_gy: int,
    ) -> None:
        """设置地形信息（从效率图获取）。"""
        self._terrain_mask = terrain_mask
        self._reference_maps = None
        self._min_gx = min_gx
        self._min_gy = min_gy

    def set_surface_segments(self, segments: list[np.ndarray]) -> None:
        """
        设置可着陆表面线段，启用随机投放位置采样。

        segments: L7 compute_landable_surfaces() 的输出，
                  每个元素 shape (N, 2) 为一条连续表面线段。
        """
        self._segments = segments
        self._arc_lengths, self._cum_lengths = precompute_segment_arcs(segments)
        total_len = float(self._arc_lengths.sum())
        logger.info(
            "已加载 %d 条表面线段 (总弧长=%.1f), 启用随机投放",
            len(segments), total_len,
        )

    def set_deploy_obstacle_geometry(
        self,
        polygons: list[np.ndarray],
        player_data: dict | None,
    ) -> None:
        """
        登记 env 几何：既供运行时投放的锤头卡障碍过滤，也供 dualscale patch 的接触分支。

        polygons: env 实心多边形（extract_polygons 输出，与 L7 一致）。
        player_data: player_contour.json 内容；提供 tip / body∪pot 局部轮廓。

        patch 接触分支（_solid_polygons / _tip_local / _body_local）无条件登记；
        运行时锤头卡障碍过滤（_tip_cfg）受 config.deploy_tip_obstacle_check 开关控制。
        """
        # patch dualscale 接触分支几何（与 deploy 过滤开关无关）
        self._solid_polygons = polygons
        self._feature_surface_index = None  # New geometry invalidates candidate-surface features.
        self._contact_polygons = TerrainContactIndex(polygons)
        self._tip_local = extract_tip_local(player_data)
        self._body_local = extract_body_local(player_data)
        # 按名称拆分：body_contact / pot_contact 需分别对独立多边形做距离查询
        self._body_only_local, self._pot_only_local = split_body_pot_local(player_data)
        from .sparse_geometry import SparseGeometryCache
        self._sparse_cache = (
            SparseGeometryCache.build(
                polygons, self._tip_local, self._body_only_local, self._pot_only_local,
            )
            if self.config.sparse_geometry else None
        )

        if not self.config.deploy_tip_obstacle_check:
            self._tip_cfg = None
            logger.info("运行时锤头卡障碍过滤: 已关闭 (deploy_tip_obstacle_check=False)")
            return

        if self._tip_local is not None:
            self._tip_cfg = make_tip_recon_config(
                self._tip_local,
                self.config.tip_angle_source,
                self.config.tip_angle_offset,
                self.config.tip_scale,
            )
            logger.info(
                "运行时锤头判据: 多边形全顶点 source=%s offset=%.0f scale=%.2f (%d 顶点)",
                self.config.tip_angle_source, self.config.tip_angle_offset,
                self.config.tip_scale, len(self._tip_local),
            )
        else:
            self._tip_cfg = None
            logger.info("运行时锤头判据: tip 中心单点 (player_contour 缺 tip 轮廓)")

    def set_candidate_points(self, points: list[list[float]]) -> None:
        """设置候选投放点集（L7 唯一候选文件 + train 侧高度过滤后的结果）。

        候选集作为投放的唯一来源（pool-only）：清空表面线段，并按人工涂抹排除区过滤。
        """
        self._segments = []
        self._arc_lengths = np.array([])
        self._cum_lengths = []
        pts = [[float(x), float(y)] for x, y in points] if points else []
        if self._exclusion_zones is not None and pts:
            before = len(pts)
            pts = filter_points_by_exclusion(pts, self._exclusion_zones)
            if before != len(pts):
                logger.info("候选点集经排除区过滤: %d → %d", before, len(pts))
        self._fixed_drop_points = pts
        logger.info("候选投放点集已就绪: %d 个（pool-only 投放）", len(pts))

    def _sample_positions(self, n: int) -> list[list[float]]:
        """采样 n 个投放位置：优先候选点集（pool-only），无候选集时回退表面线段。

        启用人工涂抹排除区时，落入排除圆的采样点被剔除并重采补足（有限次尝试）。
        """
        if self._fixed_drop_points:
            return sample_from_fixed_pool(self._fixed_drop_points, n, self._rng)
        if len(self._segments) > 0:
            if self._exclusion_zones is None:
                return sample_from_segments(
                    self._segments, self._arc_lengths, self._cum_lengths,
                    n, self._drop_height, self._rng,
                )
            collected: list[list[float]] = []
            for _ in range(8):
                if len(collected) >= n:
                    break
                need = n - len(collected)
                batch = sample_from_segments(
                    self._segments, self._arc_lengths, self._cum_lengths,
                    max(need * 2, need), self._drop_height, self._rng,
                )
                collected.extend(filter_points_by_exclusion(batch, self._exclusion_zones))
            return collected[:n]
        return []

    def setup(self) -> None:
        """首次初始化：加载投放点 + 启动游戏。"""
        game_root = Path(self.config.game_root)
        self._fixed_drop_points = _load_drop_points(game_root)
        if self._exclusion_zones is not None and self._fixed_drop_points:
            before = len(self._fixed_drop_points)
            self._fixed_drop_points = filter_points_by_exclusion(
                self._fixed_drop_points, self._exclusion_zones
            )
            if before != len(self._fixed_drop_points):
                logger.info("固定回退池经排除区过滤: %d → %d",
                            before, len(self._fixed_drop_points))
        self.launch_game()

    def _load_exclusion_zones(self) -> None:
        """加载人工涂抹排除区（项目 checkpoints/，受 config.deploy_exclusion_check 开关控制）。"""
        if not self.config.deploy_exclusion_check:
            self._exclusion_zones = None
            return
        path = default_exclusion_zones_path()
        self._exclusion_zones = load_exclusion_zones(path)
        if self._exclusion_zones is not None:
            logger.info("运行时人工排除区: 从 %s 载入 %d 个圆",
                        path, len(self._exclusion_zones))

    def launch_game(self) -> None:
        """启动游戏进程、TCP 连接、warmup、拍快照、开启自由相机。可多次调用。"""
        game_root = Path(self.config.game_root)
        _write_num_duplicates(game_root, self.config.num_agents)
        GameModeController(str(game_root)).set_game_runtime_mode()
        self._game_process = GameLauncher().launch(wait=False)

        self.env = make_goi_env(self.config)
        self.env.connect()
        self.env.reset()

        _warmup_and_snapshot(
            self.env, self.config.num_agents, self.config.warmup_steps
        )
        self.env.set_camera_free(True)

        logger.info("游戏已启动，%d agents 就绪", self.config.num_agents)

    def close_game(self) -> None:
        """关闭 TCP 连接并终止游戏进程，释放资源。"""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

        if self._game_process is not None:
            try:
                self._game_process.terminate()
                self._game_process.wait(timeout=10)
            except Exception:
                self._game_process.kill()
            self._game_process = None

        logger.info("游戏已关闭")

    def _reset_and_deploy(self) -> np.ndarray:
        """重置环境并根据 config.random_deploy 决定部署方式。"""
        num_agents = self.config.num_agents
        obs = self.env.reset()

        self._last_deploy_positions: dict[int, float] = {}

        if not self.config.random_deploy:
            logger.debug("所有 agent 留在初始位置")
            return obs

        if num_agents > 1:
            positions = self._sample_positions(num_agents - 1)
            if positions:
                for i, (_, y) in enumerate(positions):
                    self._last_deploy_positions[i + 1] = y
                obs = _deploy_agents(
                    self.env, positions,
                    self.config.settle_steps, num_agents,
                )
        return obs

    def _get_stable_agents(self, obs: np.ndarray) -> set[int]:
        """返回 settle 后仍然稳定的 agent 索引集合。"""
        if not self.config.random_deploy:
            return set(range(self.config.num_agents))

        threshold = self.config.settle_drop_threshold
        stable = {0}
        for agent_idx, teleport_y in self._last_deploy_positions.items():
            settled_y = float(obs[agent_idx, 1])
            drop = teleport_y - settled_y
            if drop > threshold:
                logger.info(
                    "Agent %d 不稳定: teleport_y=%.1f settled_y=%.1f drop=%.1f > %.1f, 跳过",
                    agent_idx, teleport_y, settled_y, drop, threshold,
                )
                continue
            if self._solid_polygons is not None and tip_in_obstacle(
                obs[agent_idx], self._solid_polygons, self._tip_cfg,
            ):
                logger.info(
                    "Agent %d 锤头卡进障碍物 (tip=%.1f,%.1f), 跳过",
                    agent_idx, float(obs[agent_idx, 23]), float(obs[agent_idx, 24]),
                )
                continue
            # settle 后落入人工排除区（按实际身体位置判定），兜底剔除
            if self._exclusion_zones is not None and point_in_exclusion(
                float(obs[agent_idx, 0]), float(obs[agent_idx, 1]) + self._drop_height,
                self._exclusion_zones,
            ):
                logger.info(
                    "Agent %d 落在人工排除区 (pos=%.1f,%.1f), 跳过",
                    agent_idx, float(obs[agent_idx, 0]), float(obs[agent_idx, 1]),
                )
                continue
            stable.add(agent_idx)
        return stable

    def _compute_contact_batch(
        self,
        obs: np.ndarray,
        prev_states: dict[int, ContactPrevState],
    ) -> tuple[np.ndarray, dict[int, ContactPrevState]]:
        """批量算 (num_agents, CONTACT_DIM) 接触信号，并原地更新 prev_states。

        - 缺 tip/body/env polygons → 返回零 → 训练侧退化为"无接触信息"（等价旧 34D）
        - prev_states 里没有的 agent 用空 ContactPrevState() 作首帧
        """
        num_agents = obs.shape[0]
        contact = np.zeros((num_agents, CONTACT_DIM), dtype=np.float32)
        if (
            self._tip_local is None
            or not self._solid_polygons
        ):
            return contact, prev_states

        for i in range(num_agents):
            prev = prev_states.get(i, ContactPrevState())
            row, new_prev = compute_contact_row(
                obs[i], prev,
                tip_local=self._tip_local,
                body_local=self._body_only_local,
                pot_local=self._pot_only_local,
                env_polygons=self._contact_polygons if self._contact_polygons is not None else self._solid_polygons,
                cfg=self._contact_cfg,
            )
            contact[i] = row
            prev_states[i] = new_prev
        return contact, prev_states

    def collect_random(self, n_steps: int) -> list[Trajectory]:
        """第 0 轮: 随机动作 + 随机投放位置采集轨迹。"""
        assert self.env is not None, "请先调用 setup()"
        num_agents = self.config.num_agents
        scale = self.config.action_scale

        obs = self._reset_and_deploy()

        all_states = [[] for _ in range(num_agents)]
        all_actions = [[] for _ in range(num_agents)]
        all_contacts = [[] for _ in range(num_agents)]
        contact_prev: dict[int, ContactPrevState] = {}

        # 首帧接触信号（首帧 grip 恒 0，符合 compute_contact_row 首帧语义）
        c0, contact_prev = self._compute_contact_batch(obs, contact_prev)
        for i in range(num_agents):
            all_states[i].append(obs[i].copy())
            all_contacts[i].append(c0[i].copy())

        for _ in range(n_steps):
            actions = np.random.uniform(-scale, scale, (num_agents, 2)).astype(np.float32)
            obs, _ = self.env.step(actions)
            c, contact_prev = self._compute_contact_batch(obs, contact_prev)
            for i in range(num_agents):
                all_states[i].append(obs[i].copy())
                all_actions[i].append(actions[i].copy())
                all_contacts[i].append(c[i].copy())

        trajectories = []
        for i in range(num_agents):
            trajectories.append(Trajectory(
                raw_states=np.array(all_states[i]),
                actions=np.array(all_actions[i]),
                contact=np.array(all_contacts[i], dtype=np.float32),
            ))
        return _filter_water_trajectories(trajectories, self.config)

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

        obs = self._reset_and_deploy()

        all_states = [[] for _ in range(num_agents)]
        all_actions = [[] for _ in range(num_agents)]
        all_contacts = [[] for _ in range(num_agents)]
        dyn_history = [[] for _ in range(num_agents)]
        act_history = [[] for _ in range(num_agents)]
        patch_history = [[] for _ in range(num_agents)]
        contour_history = [[] for _ in range(num_agents)]
        cmask_history = [[] for _ in range(num_agents)]
        relation_history = [[] for _ in range(num_agents)]
        rmask_history = [[] for _ in range(num_agents)]
        contact_prev: dict[int, ContactPrevState] = {}

        c0, contact_prev = self._compute_contact_batch(obs, contact_prev)
        contact_now = c0
        for i in range(num_agents):
            all_states[i].append(obs[i].copy())
            all_contacts[i].append(c0[i].copy())

        for step in range(n_steps):
            patches = _build_patches_batch(
                obs, eff_map, self.config,
                self._min_gx, self._min_gy, self._terrain_mask,
                solid_polygons=self._solid_polygons,
                tip_local=self._tip_local, body_local=self._body_local,
            )
            dynamics = build_dynamics(obs, contact_now)
            sparse = self._ppo_sparse(obs)

            actions = np.zeros((num_agents, 2), dtype=np.float32)
            for i in range(num_agents):
                dyn_history[i].append(dynamics[i].copy())
                patch_history[i].append(patches[i].copy())
                contour_history[i].append(sparse[0][i].copy())
                cmask_history[i].append(sparse[1][i].copy())
                relation_history[i].append(sparse[2][i].copy())
                rmask_history[i].append(sparse[3][i].copy())

                dyn_window, pat_window, act_window, valid_mask = _build_history_window(
                    dyn_history[i], patch_history[i], act_history[i],
                    ctx,
                )
                ct, cm, rt, rm = _pad_sparse_history(
                    contour_history[i], cmask_history[i],
                    relation_history[i], rmask_history[i], ctx,
                )

                pred = model.predict(
                    dyn_window, pat_window, act_window, valid_mask=valid_mask,
                    contour_tokens=ct, contour_mask=cm,
                    relation_tokens=rt, relation_mask=rm,
                )
                noise = np.random.normal(0, noise_std * scale, size=2).astype(np.float32)
                actions[i] = np.clip(pred + noise, -scale, scale)

            obs, _ = self.env.step(actions)
            c_next, contact_prev = self._compute_contact_batch(obs, contact_prev)
            contact_now = c_next
            for i in range(num_agents):
                all_states[i].append(obs[i].copy())
                all_actions[i].append(actions[i].copy())
                all_contacts[i].append(c_next[i].copy())
                act_history[i].append(actions[i].copy())

        trajectories = []
        for i in range(num_agents):
            trajectories.append(Trajectory(
                raw_states=np.array(all_states[i]),
                actions=np.array(all_actions[i]),
                contact=np.array(all_contacts[i], dtype=np.float32),
            ))
        return _filter_water_trajectories(trajectories, self.config)

    def _ppo_patches(self, obs, eff_map, trackers: dict[int, CorridorTracker]):
        guidance = None
        if self.reference_corridor is not None:
            if self._reference_maps is None:
                if self._terrain_mask is None:
                    raise ValueError("Reference corridor observations require set_terrain_info()")
                self._reference_maps = self.reference_corridor.guidance_maps(
                    self._terrain_mask.shape, self._min_gx, self._min_gy,
                    self.config.grid_resolution,
                )
            guidance = {
                i: self._reference_maps[t.guidance_index]
                for i, t in trackers.items() if t.guidance_index is not None
            }
        return _build_patches_batch(
            obs, eff_map, self.config, self._min_gx, self._min_gy, self._terrain_mask,
            solid_polygons=self._solid_polygons, tip_local=self._tip_local,
            body_local=self._body_local, guidance_arrays=guidance,
        )

    def _ppo_sparse(self, obs):
        from .sparse_geometry import build_sparse_geometry, empty_sparse_frame
        blank = empty_sparse_frame(self.config)
        n = len(obs)
        contour = np.zeros((n, *blank[0].shape), dtype=np.float32)
        cmask = np.zeros((n, *blank[1].shape), dtype=np.float32)
        relation = np.zeros((n, *blank[2].shape), dtype=np.float32)
        rmask = np.zeros((n, *blank[3].shape), dtype=np.float32)
        cache = self._sparse_cache
        for i in range(n):
            contour[i], cmask[i], relation[i], rmask[i] = build_sparse_geometry(
                obs[i], cache, self.config,
            )
        return contour, cmask, relation, rmask

    def collect_ppo(
        self,
        model: ActorCritic,
        n_steps: int,
        eff_map: ClimbingEfficiencyMap | None = None,
        normalizer: RewardNormalizer | None = None,
        *,
        deterministic: bool = False,
        initial_offsets: np.ndarray | None = None,
        native_diagnostics: bool = False,
    ) -> tuple[PPORolloutBuffer, list[Trajectory]]:
        """
        PPO 采集：每步记录 (action, log_prob, value, reward)。

        每个 agent 使用独立 buffer，采集结束后 per-agent 计算 GAE
        （各自的 bootstrap value），再合并为单一 buffer 用于训练。
        同时收集 per-agent 原始状态/动作，返回 Trajectory 列表供效率图更新。
        """
        from .ppo_buffer import PPORolloutBuffer as _Buffer
        from .reward import NewHighConfirmer, RewardNormalizer, is_water, step_reward
        from .policy_audit import policy_context
        from .policy_features import experimental_enabled
        if experimental_enabled(self.config):
            from .experimental_rollout import collect_decisions
            return collect_decisions(self, model, n_steps, eff_map, normalizer,
                deterministic=deterministic, initial_offsets=initial_offsets, native_diagnostics=native_diagnostics)

        started = time.perf_counter()
        timings = {name: 0.0 for name in ("patch", "contact", "inference", "env", "bootstrap", "diagnostics")}

        assert self.env is not None, "请先调用 setup() 或 launch_game()"
        if n_steps < 1 or self.config.ppo_inference_batch_size < 0:
            raise ValueError("Positive rollout length and nonnegative inference batch size required")
        num_agents = self.config.num_agents
        ctx = self.config.context_len
        device = next(model.parameters()).device

        # PPO 采集必须关 dropout：否则 old_log_prob 用随机 dropout mask 计算，与更新时重算的
        # new_log_prob 对不上 → 重要性采样比 ratio 系统性偏离 1 → approx_kl 虚高、策略被污染。
        model.eval()

        if native_diagnostics:
            # Enable before RESET so the first retained packet is sampled beside the state.
            self.env.observation_diagnostics()
        obs = self._reset_and_deploy()
        if initial_offsets is not None:
            if self.config.random_deploy or np.shape(initial_offsets) != (num_agents, 2):
                raise ValueError("Fixed evaluation offsets require fixed deployment and shape (N, 2)")
            positions = obs[:, :2].copy() + initial_offsets
            for i, position in enumerate(positions):
                if np.any(initial_offsets[i]):
                    obs = self.env.teleport(float(position[0]), float(position[1]), agent_index=i)

        stable_agents = self._get_stable_agents(obs)
        n_skipped = num_agents - len(stable_agents)
        if n_skipped > 0:
            logger.info(
                "稳定性过滤: %d/%d agents 稳定, %d 跳过",
                len(stable_agents), num_agents, n_skipped,
            )

        agent_buffers: dict[int, _Buffer] = {i: _Buffer() for i in stable_agents}

        raw_states: dict[int, list[np.ndarray]] = {
            i: [obs[i].copy()] for i in stable_agents
        }
        raw_actions: dict[int, list[np.ndarray]] = {
            i: [] for i in stable_agents
        }

        # 接触信号跨帧状态与历史缓冲（与 raw_states 一样：首帧存进 all_contacts）
        contact_prev: dict[int, ContactPrevState] = {}
        c0, contact_prev = self._compute_contact_batch(obs, contact_prev)
        contact_now = c0
        self.last_native_diagnostics = []
        if native_diagnostics:
            self.last_native_diagnostics.append(self.env.observation_diagnostics())
            if not self.last_native_diagnostics[-1].get("aligned", False):
                raise ValueError("Native diagnostics must be captured at the physics-step boundary (plugin v2)")
        all_contacts: dict[int, list[np.ndarray]] = {
            i: [c0[i].copy()] for i in stable_agents
        }

        dyn_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        act_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        patch_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        contour_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        cmask_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        relation_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        rmask_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        contour_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        cmask_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        relation_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}
        rmask_history: dict[int, list[np.ndarray]] = {i: [] for i in stable_agents}

        # OU（时序连贯探索噪声）per-agent AR(1) 状态。
        # z_prev 初值 ~ N(0,1)：保证第 0 步条件采样的 z_0 仍 marginal N(0,1)
        # （若起手 z_{-1}=0，则 σ_cond=std·√(1-φ²) 会系统性压低首步探索方差）。
        # φ=0 或 ou_enabled=False → 条件退化 iid，z_t 即当步 η_t。
        # 每步存 z_prev 到 buffer，PPO 更新用条件式 log_prob → ratio 数学自洽。
        ou_phi = self.config.ou_phi if self.config.ou_enabled and not deterministic else 0.0
        for b in agent_buffers.values():
            b.policy_context = policy_context(model, ou_phi, deterministic)
        ou_state: dict[int, np.ndarray] = {
            i: np.random.randn(self.config.action_dim).astype(np.float32)
            for i in stable_agents
        }
        prev_obs = obs.copy()
        running_max_y = {i: float(obs[i, 1]) for i in stable_agents}
        # 每 agent 一个确认式创新高器：新高守住 dwell 步才结算，滤除甩飞尖峰
        confirmers = {
            i: NewHighConfirmer(
                float(obs[i, 1]), self.config.dwell_steps, self.config.dwell_drop_tol
            )
            for i in stable_agents
        }
        score_schema, score_meta = (None, None)
        if self.reference_corridor is not None and stable_agents:
            from .reference_anchor import scoring_context
            first = next(iter(stable_agents))
            score_schema, score_meta = scoring_context(
                obs[first], self.config, self.env, self.reference_corridor,
            )
        trackers = {
            i: self.reference_corridor.tracker_from_state(obs[i], score_schema, score_meta)
            for i in stable_agents
        } if self.reference_corridor is not None else {}
        # 走廊模式逐帧字段（供 SIL 剪枝/排序/patch 重建；非走廊模式保持空 dict）：
        # corridor_credit[i] 与 raw_actions[i] 对齐（动作 t 获得的 credited 进度）；
        # corridor_guidance[i] 与 raw_states[i] 对齐（处理完 state[t] 后的引导段索引，
        # 即该帧 patch wide 通道所用地图；-1 = 无引导）。首帧为 tracker 初始状态。
        corridor_credit: dict[int, list[float]] = {i: [] for i in stable_agents} if trackers else {}
        corridor_guidance: dict[int, list[int]] = (
            {i: [trackers[i].guidance_index if trackers[i].guidance_index is not None else -1]
             for i in stable_agents} if trackers else {}
        )
        finished: set[int] = set()
        corridor_steps = corridor_inside = corridor_gaps = 0
        corridor_progress = corridor_penalties = 0.0
        self.last_corridor_metrics = {}

        for step in range(n_steps):
            phase_start = time.perf_counter()
            patches = self._ppo_patches(obs, eff_map, trackers)
            sparse = self._ppo_sparse(obs)
            timings["patch"] += time.perf_counter() - phase_start
            dynamics = build_dynamics(obs, contact_now)

            actions_batch = np.zeros((num_agents, 2), dtype=np.float32)
            step_data: dict[int, tuple] = {}
            phase_start = time.perf_counter()
            active = sorted(stable_agents - finished)
            windows = {}
            for i in active:
                dyn_history[i].append(dynamics[i].copy())
                patch_history[i].append(patches[i].copy())
                contour_history[i].append(sparse[0][i].copy())
                cmask_history[i].append(sparse[1][i].copy())
                relation_history[i].append(sparse[2][i].copy())
                rmask_history[i].append(sparse[3][i].copy())
                windows[i] = (
                    *_build_history_window(
                        dyn_history[i], patch_history[i], act_history[i],
                        ctx,
                    ),
                    *_pad_sparse_history(
                        contour_history[i], cmask_history[i],
                        relation_history[i], rmask_history[i], ctx,
                    ),
                )
            inference_batch = self.config.ppo_inference_batch_size or max(1, len(active))
            for start in range(0, len(active), inference_batch):
                ids = active[start:start + inference_batch]
                dyn_t, pat_t, act_t, vm_t, ct_t, cm_t, rt_t, rm_t = (
                    torch.from_numpy(np.stack([windows[i][k] for i in ids])).to(device)
                    for k in range(8)
                )
                z_prev_t = torch.from_numpy(np.stack([ou_state[i] for i in ids])).to(device)
                with torch.no_grad():
                    av = model.get_action_and_value(
                        dyn_t, pat_t, act_t, valid_mask=vm_t,
                        noise_prev=z_prev_t, noise_phi=ou_phi,
                        deterministic=deterministic,
                        contour_tokens=ct_t, contour_mask=cm_t,
                        relation_tokens=rt_t, relation_mask=rm_t,
                    )
                output = [tensor.detach().cpu().numpy() for tensor in
                          (av.action, av.raw_action, av.log_prob, av.value, av.noise)]
                if not all(np.isfinite(array).all() for array in output):
                    raise FloatingPointError(f"Invalid policy output at step={step}, agents={ids}")
                for j, i in enumerate(ids):
                    action_np, latent_np, lp, val, z_t = (array[j] for array in output)
                    z_prev = ou_state[i]
                    ou_state[i] = z_t.copy()
                    actions_batch[i] = action_np
                    step_data[i] = (*windows[i], float(lp), float(val), z_prev.copy(), latent_np.copy())
            timings["inference"] += time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            new_obs, dones = self.env.step(actions_batch)
            timings["env"] += time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            contact_next, contact_prev = self._compute_contact_batch(new_obs, contact_prev)
            timings["contact"] += time.perf_counter() - phase_start
            if native_diagnostics:
                phase_start = time.perf_counter()
                self.last_native_diagnostics.append(self.env.observation_diagnostics())
                if (not self.last_native_diagnostics[-1].get("aligned", False)
                        or self.last_native_diagnostics[-1]["sequence"] <= self.last_native_diagnostics[-2]["sequence"]):
                    raise ValueError("Native diagnostic snapshot did not advance with the environment")
                timings["diagnostics"] += time.perf_counter() - phase_start

            for i in stable_agents:
                if i not in step_data:
                    continue
                dw, pw, aw, vm, ct, cm, rt, rm, lp, val, np_prev, latent = step_data[i]
                reward, running_max_y[i] = step_reward(
                    prev_obs[i], new_obs[i], eff_map, self.config,
                    running_max_y=running_max_y[i],
                    normalizer=normalizer,
                    confirmer=confirmers[i],
                    corridor=trackers.get(i),
                    state_schema=score_schema,
                    reference_metadata=score_meta,
                )
                done = bool(dones[i]) or is_water(new_obs[i, 1], self.config)
                if done:
                    finished.add(i)
                if trackers:
                    result = trackers[i].last_step
                    corridor_steps += 1
                    corridor_inside += int(result.inside)
                    corridor_gaps += int(result.gap)
                    corridor_progress += result.progress
                    corridor_penalties += result.deviation_penalty
                    if done:
                        finished.add(i)
                agent_buffers[i].add(
                    dynamics_window=dw,
                    patch_window=pw,
                    act_history_window=aw,
                    valid_mask_window=vm,
                    action=actions_batch[i].copy(),
                    log_prob=lp,
                    value=val,
                    reward=reward,
                    done=done,
                    noise_prev=np_prev,
                    raw_action=latent,
                    patch_frames=tuple(patch_history[i][-ctx:]),
                    contour_window=ct,
                    contour_mask_window=cm,
                    relation_window=rt,
                    relation_mask_window=rm,
                )
                act_history[i].append(actions_batch[i].copy())
                raw_states[i].append(new_obs[i].copy())
                raw_actions[i].append(actions_batch[i].copy())
                all_contacts[i].append(contact_next[i].copy())
                if trackers:
                    # 此时 tracker 已在 step_reward 内完成本步 step：last_step.progress
                    # 是本动作获得的 credited；guidance_index 是 new_obs 帧的引导段。
                    corridor_credit[i].append(float(trackers[i].last_step.progress))
                    gid = trackers[i].guidance_index
                    corridor_guidance[i].append(gid if gid is not None else -1)

            prev_obs = new_obs.copy()
            obs = new_obs
            contact_now = contact_next
            if len(finished) == len(stable_agents):
                break

        # per-agent bootstrap value + GAE
        phase_start = time.perf_counter()
        patches = self._ppo_patches(obs, eff_map, trackers)
        sparse = self._ppo_sparse(obs)
        # bootstrap 帧的 contact：接续 rollout 末尾 prev 状态，保证 grip 判定连续
        dynamics = build_dynamics(obs, contact_now)
        for i in stable_agents:
            if i in finished:
                agent_buffers[i].compute_gae(0.0, self.config.gamma, self.config.gae_lambda)
                continue
            dyn_history[i].append(dynamics[i].copy())
            patch_history[i].append(patches[i].copy())
            contour_history[i].append(sparse[0][i].copy())
            cmask_history[i].append(sparse[1][i].copy())
            relation_history[i].append(sparse[2][i].copy())
            rmask_history[i].append(sparse[3][i].copy())
            dyn_window, pat_window, act_window, valid_mask = _build_history_window(
                dyn_history[i], patch_history[i], act_history[i],
                ctx,
            )
            ct, cm, rt, rm = _pad_sparse_history(
                contour_history[i], cmask_history[i],
                relation_history[i], rmask_history[i], ctx,
            )
            dyn_t = torch.from_numpy(dyn_window[np.newaxis].astype(np.float32)).to(device)
            pat_t = torch.from_numpy(pat_window[np.newaxis].astype(np.float32)).to(device)
            act_t = torch.from_numpy(act_window[np.newaxis].astype(np.float32)).to(device)
            vm_t = torch.from_numpy(valid_mask[np.newaxis].astype(np.float32)).to(device)
            with torch.no_grad():
                _, v = model.forward(
                    dyn_t, pat_t, act_t, valid_mask=vm_t,
                    contour_tokens=torch.from_numpy(ct[np.newaxis]).to(device),
                    contour_mask=torch.from_numpy(cm[np.newaxis]).to(device),
                    relation_tokens=torch.from_numpy(rt[np.newaxis]).to(device),
                    relation_mask=torch.from_numpy(rm[np.newaxis]).to(device),
                )
            agent_buffers[i].compute_gae(
                last_value=v.item(),
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

        buffer = _Buffer.merge(list(agent_buffers.values()))
        timings["bootstrap"] = time.perf_counter() - phase_start
        timings["total"] = time.perf_counter() - started
        self.last_profile = {f"rollout_{k}_seconds": v for k, v in timings.items()}
        self.last_profile["rollout_samples_per_second"] = buffer.size / timings["total"]
        self.last_profile["rollout_patch_storage_bytes"] = float(buffer.patch_storage_bytes())
        self.last_profile["rollout_patch_window_bytes"] = float(
            buffer.size * ctx * self.config.patch_channels * self.config.patch_size**2 * 4)

        if trackers:
            self.last_corridor_metrics = {
                "corridor_progress_m": corridor_progress,
                "corridor_deviation_penalty": corridor_penalties,
                "corridor_inside_fraction": corridor_inside / max(1, corridor_steps),
                "corridor_gap_fraction": corridor_gaps / max(1, corridor_steps),
            }

        trajectories = []
        for i in stable_agents:
            if len(raw_actions[i]) > 0:
                trajectories.append(Trajectory(
                    raw_states=np.array(raw_states[i]),
                    actions=np.array(raw_actions[i]),
                    contact=np.array(all_contacts[i], dtype=np.float32),
                    corridor_progress=(np.array(corridor_credit[i], dtype=np.float32)
                                       if trackers else None),
                    guidance_index=(np.array(corridor_guidance[i], dtype=np.int8)
                                    if trackers else None),
                ))

        return buffer, trajectories

    def teardown(self) -> None:
        """关闭连接并终止游戏进程。"""
        self.close_game()
        logger.info("RolloutWorker 已关闭")

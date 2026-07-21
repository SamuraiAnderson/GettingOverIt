"""
游戏内物理探测：空投点 settle 后竖直落差，与 training.rollout._get_stable_agents 一致。

drop = teleport_y - settled_y（settled_y = obs[agent, 1]），stable 当且仅当 drop <= settle_drop_threshold。

默认使用与 L8 / RolloutWorker._deploy_agents 相同的「多复制体同时传送后统一 settle」批量探测；
可设 use_l8_batch_deploy=False 退回仅 agent 1 的逐点探测（num_agents=2）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from training.deploy_sampling import (
    TipReconConfig,
    make_tip_recon_config,
    tip_in_obstacle as _tip_in_obstacle,
)

logger = logging.getLogger(__name__)

# 与 PlayerDuplicateManager.MAX_DUPLICATES 一致：最多 64 个复制体 + agent 0
UNITY_MAX_DUPLICATES = 64

# 锤头判定逻辑（TipReconConfig / make_tip_recon_config / _tip_in_obstacle）已下沉到
# 生产模块 training.deploy_sampling，供离线筛选（本文件）与运行时投放（rollout）共用；
# 此处再导出，保持 L7 入口 `from drop_point_physics import make_tip_recon_config` 不变。
__all__ = ["TipReconConfig", "make_tip_recon_config"]


def _tip_check_mode(solid_polygons, tip_cfg) -> str:
    """记录到 meta 的锤头检查模式说明。"""
    if solid_polygons is None or len(solid_polygons) == 0:
        return "off"
    # 判据已统一为「锤头中心点」；tip_cfg 保留仅为兼容，不再影响判定。
    return "tip_center_point"


def try_drop_stable(
    env,
    x: float,
    y: float,
    *,
    agent_index: int = 1,
    settle_steps: int,
    threshold: float,
    solid_polygons=None,
    tip_cfg=None,
) -> tuple[bool, float, bool]:
    """
    调用前须已 reset() 到基准快照。
    teleport → settle_steps 步零动作 → 计算 drop 与 threshold 比较。

    返回 (ok, drop, tip_in_obstacle)。ok 要求竖直落差 <= threshold 且锤头未卡进障碍物。
    """
    teleport_y = float(y)
    env.teleport(x, y, agent_index=agent_index)
    n = env.num_agents
    zero = np.zeros((n, 2), dtype=np.float32)
    obs = None
    for _ in range(max(settle_steps, 1)):
        obs, _ = env.step(zero)
    settled_y = float(obs[agent_index, 1])
    drop = teleport_y - settled_y
    tip_bad = _tip_in_obstacle(obs[agent_index], solid_polygons, tip_cfg)
    return (drop <= threshold and not tip_bad), drop, tip_bad


def _warmup_and_snapshot(env, num_agents: int, warmup_steps: int) -> None:
    env.reset()
    z = np.zeros((num_agents, 2), dtype=np.float32)
    for _ in range(warmup_steps):
        env.step(z)
    env.new_snapshot()


def _teardown_env_and_process(env: Any, proc: Any) -> None:
    try:
        env.close()
    except Exception:
        pass
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _deploy_chunk_settle(env, chunk: np.ndarray, settle_steps: int) -> np.ndarray:
    """L8 / rollout._deploy_agents：对 agents 1..B 传送后统一零动作 settle。"""
    env.reset()
    b = len(chunk)
    for i in range(b):
        env.teleport(float(chunk[i, 0]), float(chunk[i, 1]), agent_index=i + 1)
    n_ag = env.num_agents
    z = np.zeros((n_ag, 2), dtype=np.float32)
    obs = None
    for _ in range(max(settle_steps, 1)):
        obs, _ = env.step(z)
    assert obs is not None
    return obs


def _chunk_drops_and_stable(
    chunk: np.ndarray,
    obs: np.ndarray,
    threshold: float,
    solid_polygons=None,
    tip_cfg=None,
) -> tuple[list[bool], list[float], list[bool]]:
    out_ok: list[bool] = []
    out_drop: list[float] = []
    out_tip_bad: list[bool] = []
    for i in range(len(chunk)):
        ty = float(chunk[i, 1])
        sy = float(obs[i + 1, 1])
        dr = ty - sy
        tip_bad = _tip_in_obstacle(obs[i + 1], solid_polygons, tip_cfg)
        out_ok.append(dr <= threshold and not tip_bad)
        out_drop.append(dr)
        out_tip_bad.append(tip_bad)
    return out_ok, out_drop, out_tip_bad


def filter_drop_points_stable(
    candidates: np.ndarray,
    *,
    target_count: int,
    max_tries: int | None,
    settle_drop_threshold: float,
    settle_steps: int,
    warmup_steps: int,
    port: int,
    game_root: Path,
    timeout: float = 60.0,
    no_launch: bool = False,
    use_l8_batch_deploy: bool = True,
    l8_batch_size: int = UNITY_MAX_DUPLICATES,
    solid_polygons=None,
    tip_cfg=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    顺序考察 candidates，保留稳定点直到凑满 target_count 或达到 max_tries。

    use_l8_batch_deploy=True（默认）：每批最多 l8_batch_size 个点，一次 reset 后多 agent
    同时传送再统一 settle，与 L8 / _deploy_agents 一致。

    solid_polygons 非空时，settle 后锤头卡进障碍物的候选也判为不稳定并计入 rejected。
    """
    from env.goi_env import GoiEnv
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    from training.rollout import _write_num_duplicates

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.size == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "tried": 0, "kept": 0, "rejected": 0, "target_count": target_count,
            "settle_drop_threshold": settle_drop_threshold, "settle_steps": settle_steps,
            "deploy_mode": "none",
        }
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must be of shape (N, 2)")

    n_cand = len(candidates)
    limit = n_cand if max_tries is None else min(max_tries, n_cand)
    eff_bs = max(1, min(int(l8_batch_size), UNITY_MAX_DUPLICATES))

    meta: dict[str, Any] = {
        "target_count": target_count,
        "settle_drop_threshold": settle_drop_threshold,
        "settle_steps": settle_steps,
        "warmup_steps": warmup_steps,
        "use_l8_batch_deploy": use_l8_batch_deploy,
        "l8_batch_size": eff_bs,
        "tip_obstacle_check": solid_polygons is not None and len(solid_polygons) > 0,
        "tip_check_mode": _tip_check_mode(solid_polygons, tip_cfg),
    }

    if not use_l8_batch_deploy:
        _write_num_duplicates(game_root, 2)
        proc = None
        if not no_launch:
            GameModeController(str(game_root)).set_game_runtime_mode()
            proc = GameLauncher().launch(wait=False)
        env = GoiEnv(port=port, num_agents=2, timeout=timeout)
        env.connect()
        meta["deploy_mode"] = "sequential_2agent"
        meta["num_agents"] = 2
        try:
            _warmup_and_snapshot(env, 2, warmup_steps)
            logger.info(
                "[physics_filter] 快照完成（逐点 2-agent），最多尝试 %d 个候选，目标稳定数 %d",
                limit, target_count,
            )
            kept: list[list[float]] = []
            rejected = 0
            rejected_tip = 0
            tried = 0
            for i in range(limit):
                if len(kept) >= target_count:
                    break
                x, y = float(candidates[i, 0]), float(candidates[i, 1])
                env.reset()
                ok, drop, tip_bad = try_drop_stable(
                    env, x, y,
                    agent_index=1,
                    settle_steps=settle_steps,
                    threshold=settle_drop_threshold,
                    solid_polygons=solid_polygons,
                    tip_cfg=tip_cfg,
                )
                tried += 1
                if ok:
                    kept.append([x, y])
                    logger.info(
                        "[physics_filter] 候选 #%d 稳定 drop=%.3f", i, drop,
                    )
                elif tip_bad:
                    rejected += 1
                    rejected_tip += 1
                    logger.info(
                        "[physics_filter] 候选 #%d 锤头卡入障碍物 drop=%.3f", i, drop,
                    )
                else:
                    rejected += 1
                    logger.info(
                        "[physics_filter] 候选 #%d 不稳定 drop=%.3f > %.3f",
                        i, drop, settle_drop_threshold,
                    )
            meta.update({
                "tried": tried, "kept": len(kept), "rejected": rejected,
                "rejected_tip_in_obstacle": rejected_tip,
            })
            return np.array(kept, dtype=np.float64), meta
        finally:
            _teardown_env_and_process(env, proc)

    m_probe = limit
    chunk_cap = min(eff_bs, m_probe, UNITY_MAX_DUPLICATES)
    num_agents = chunk_cap + 1
    _write_num_duplicates(game_root, num_agents)
    proc = None
    if not no_launch:
        GameModeController(str(game_root)).set_game_runtime_mode()
        proc = GameLauncher().launch(wait=False)
    env = GoiEnv(port=port, num_agents=num_agents, timeout=timeout)
    env.connect()
    meta["deploy_mode"] = "l8_batch"
    meta["num_agents"] = num_agents
    try:
        _warmup_and_snapshot(env, num_agents, warmup_steps)
        logger.info(
            "[physics_filter] 快照完成（L8 批量，≤%d 点/批，%d agents），"
            "最多尝试 %d 个候选，目标稳定数 %d",
            eff_bs, num_agents, limit, target_count,
        )
        kept_b: list[list[float]] = []
        rejected = 0
        rejected_tip = 0
        tried = 0
        i = 0
        while len(kept_b) < target_count and i < limit:
            take = min(eff_bs, limit - i)
            chunk = candidates[i : i + take]
            i += take
            obs = _deploy_chunk_settle(env, chunk, settle_steps)
            oks, drops, tip_bads = _chunk_drops_and_stable(
                chunk, obs, settle_drop_threshold, solid_polygons, tip_cfg,
            )
            # 本批已统一 settle，须逐点统计；已满 target 时仍计入 tried，不再收录稳定点
            for j in range(len(chunk)):
                tried += 1
                x, y = float(chunk[j, 0]), float(chunk[j, 1])
                if oks[j]:
                    if len(kept_b) < target_count:
                        kept_b.append([x, y])
                        logger.info(
                            "[physics_filter] 候选稳定 drop=%.3f (批内 #%d)", drops[j], j,
                        )
                    else:
                        logger.info(
                            "[physics_filter] 候选稳定但已满额 drop=%.3f (批内 #%d)",
                            drops[j], j,
                        )
                elif tip_bads[j]:
                    rejected += 1
                    rejected_tip += 1
                    logger.info(
                        "[physics_filter] 候选锤头卡入障碍物 drop=%.3f (批内 #%d)",
                        drops[j], j,
                    )
                else:
                    rejected += 1
                    logger.info(
                        "[physics_filter] 候选不稳定 drop=%.3f > %.3f (批内 #%d)",
                        drops[j], settle_drop_threshold, j,
                    )
        meta.update({
            "tried": tried, "kept": len(kept_b), "rejected": rejected,
            "rejected_tip_in_obstacle": rejected_tip,
        })
        return np.array(kept_b, dtype=np.float64), meta
    finally:
        _teardown_env_and_process(env, proc)


def filter_all_drop_points_stable(
    candidates: np.ndarray,
    *,
    max_probes: int | None,
    settle_drop_threshold: float,
    settle_steps: int,
    warmup_steps: int,
    port: int,
    game_root: Path,
    timeout: float = 60.0,
    no_launch: bool = False,
    use_l8_batch_deploy: bool = True,
    l8_batch_size: int = UNITY_MAX_DUPLICATES,
    solid_polygons=None,
    tip_cfg=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    对候选探测，收集全部稳定点。max_probes: 最多探测前若干个；None 表示全部。

    solid_polygons 非空时，settle 后锤头卡进障碍物的候选也判为不稳定并计入 rejected。
    """
    from env.goi_env import GoiEnv
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    from training.rollout import _write_num_duplicates

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.size == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "tried": 0, "kept": 0, "rejected": 0,
            "settle_drop_threshold": settle_drop_threshold,
            "settle_steps": settle_steps,
            "deploy_mode": "none",
        }
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must be of shape (N, 2)")

    n_cand = len(candidates)
    m = n_cand if max_probes is None else min(max_probes, n_cand)
    eff_bs = max(1, min(int(l8_batch_size), UNITY_MAX_DUPLICATES))

    meta: dict[str, Any] = {
        "settle_drop_threshold": settle_drop_threshold,
        "settle_steps": settle_steps,
        "warmup_steps": warmup_steps,
        "use_l8_batch_deploy": use_l8_batch_deploy,
        "l8_batch_size": eff_bs,
        "tip_obstacle_check": solid_polygons is not None and len(solid_polygons) > 0,
        "tip_check_mode": _tip_check_mode(solid_polygons, tip_cfg),
    }

    if not use_l8_batch_deploy:
        _write_num_duplicates(game_root, 2)
        proc = None
        if not no_launch:
            GameModeController(str(game_root)).set_game_runtime_mode()
            proc = GameLauncher().launch(wait=False)
        env = GoiEnv(port=port, num_agents=2, timeout=timeout)
        env.connect()
        meta["deploy_mode"] = "sequential_2agent"
        meta["num_agents"] = 2
        try:
            _warmup_and_snapshot(env, 2, warmup_steps)
            logger.info(
                "[physics_filter_all] 快照完成（逐点 2-agent），将探测 %d 个候选（共 %d 个）",
                m, n_cand,
            )
            kept: list[list[float]] = []
            rejected = 0
            rejected_tip = 0
            for i in range(m):
                x, y = float(candidates[i, 0]), float(candidates[i, 1])
                env.reset()
                ok, drop, tip_bad = try_drop_stable(
                    env, x, y,
                    agent_index=1,
                    settle_steps=settle_steps,
                    threshold=settle_drop_threshold,
                    solid_polygons=solid_polygons,
                    tip_cfg=tip_cfg,
                )
                if ok:
                    kept.append([x, y])
                    logger.info(
                        "[physics_filter_all] #%d 稳定 drop=%.3f", i, drop,
                    )
                elif tip_bad:
                    rejected += 1
                    rejected_tip += 1
                    logger.info(
                        "[physics_filter_all] #%d 锤头卡入障碍物 drop=%.3f", i, drop,
                    )
                else:
                    rejected += 1
                    logger.info(
                        "[physics_filter_all] #%d 不稳定 drop=%.3f > %.3f",
                        i, drop, settle_drop_threshold,
                    )
            meta.update({
                "tried": m, "kept": len(kept), "rejected": rejected,
                "rejected_tip_in_obstacle": rejected_tip,
            })
            return np.array(kept, dtype=np.float64), meta
        finally:
            _teardown_env_and_process(env, proc)

    chunk_cap = min(eff_bs, m, UNITY_MAX_DUPLICATES)
    num_agents = chunk_cap + 1
    _write_num_duplicates(game_root, num_agents)
    proc = None
    if not no_launch:
        GameModeController(str(game_root)).set_game_runtime_mode()
        proc = GameLauncher().launch(wait=False)
    env = GoiEnv(port=port, num_agents=num_agents, timeout=timeout)
    env.connect()
    meta["deploy_mode"] = "l8_batch"
    meta["num_agents"] = num_agents
    try:
        _warmup_and_snapshot(env, num_agents, warmup_steps)
        logger.info(
            "[physics_filter_all] 快照完成（L8 批量，≤%d 点/批，%d agents），"
            "将探测 %d 个候选（共 %d 个）",
            eff_bs, num_agents, m, n_cand,
        )
        kept_b: list[list[float]] = []
        rejected = 0
        rejected_tip = 0
        tried = 0
        i = 0
        while i < m:
            take = min(eff_bs, m - i)
            chunk = candidates[i : i + take]
            i += take
            obs = _deploy_chunk_settle(env, chunk, settle_steps)
            oks, drops, tip_bads = _chunk_drops_and_stable(
                chunk, obs, settle_drop_threshold, solid_polygons, tip_cfg,
            )
            for j in range(len(chunk)):
                tried += 1
                x, y = float(chunk[j, 0]), float(chunk[j, 1])
                if oks[j]:
                    kept_b.append([x, y])
                    logger.info(
                        "[physics_filter_all] 稳定 drop=%.3f (批内 #%d)", drops[j], j,
                    )
                elif tip_bads[j]:
                    rejected += 1
                    rejected_tip += 1
                    logger.info(
                        "[physics_filter_all] 锤头卡入障碍物 drop=%.3f (批内 #%d)",
                        drops[j], j,
                    )
                else:
                    rejected += 1
                    logger.info(
                        "[physics_filter_all] 不稳定 drop=%.3f > %.3f (批内 #%d)",
                        drops[j], settle_drop_threshold, j,
                    )
        meta.update({
            "tried": tried, "kept": len(kept_b), "rejected": rejected,
            "rejected_tip_in_obstacle": rejected_tip,
        })
        return np.array(kept_b, dtype=np.float64), meta
    finally:
        _teardown_env_and_process(env, proc)


def probe_drop_points_stability(
    candidates: np.ndarray,
    *,
    settle_drop_threshold: float,
    settle_steps: int,
    warmup_steps: int,
    port: int,
    game_root: Path,
    timeout: float = 60.0,
    no_launch: bool = False,
    max_probes: int | None = None,
    use_l8_batch_deploy: bool = True,
    l8_batch_size: int = UNITY_MAX_DUPLICATES,
    solid_polygons=None,
    tip_cfg=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    对前 max_probes 个（或全部）候选探测，返回每条是否稳定及 drop 值（顺序与 candidates 一致）。

    solid_polygons 非空时，settle 后锤头卡进障碍物的候选也标记为不稳定。
    """
    from env.goi_env import GoiEnv
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    from training.rollout import _write_num_duplicates

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.size == 0:
        return (
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.float64),
            {"tried": 0, "settle_drop_threshold": settle_drop_threshold, "deploy_mode": "none"},
        )
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must be of shape (N, 2)")

    n_cand = len(candidates)
    m = n_cand if max_probes is None else min(max_probes, n_cand)
    eff_bs = max(1, min(int(l8_batch_size), UNITY_MAX_DUPLICATES))

    meta: dict[str, Any] = {
        "settle_drop_threshold": settle_drop_threshold,
        "settle_steps": settle_steps,
        "warmup_steps": warmup_steps,
        "use_l8_batch_deploy": use_l8_batch_deploy,
        "l8_batch_size": eff_bs,
        "tip_obstacle_check": solid_polygons is not None and len(solid_polygons) > 0,
        "tip_check_mode": _tip_check_mode(solid_polygons, tip_cfg),
    }

    if not use_l8_batch_deploy:
        _write_num_duplicates(game_root, 2)
        proc = None
        if not no_launch:
            GameModeController(str(game_root)).set_game_runtime_mode()
            proc = GameLauncher().launch(wait=False)
        env = GoiEnv(port=port, num_agents=2, timeout=timeout)
        env.connect()
        meta["deploy_mode"] = "sequential_2agent"
        meta["num_agents"] = 2
        stable_mask: list[bool] = []
        drops_list: list[float] = []
        try:
            _warmup_and_snapshot(env, 2, warmup_steps)
            logger.info("[physics_probe] 快照完成（逐点 2-agent），探测 %d 个候选", m)
            for i in range(m):
                x, y = float(candidates[i, 0]), float(candidates[i, 1])
                env.reset()
                ok, drop, _tip_bad = try_drop_stable(
                    env, x, y,
                    agent_index=1,
                    settle_steps=settle_steps,
                    threshold=settle_drop_threshold,
                    solid_polygons=solid_polygons,
                    tip_cfg=tip_cfg,
                )
                stable_mask.append(ok)
                drops_list.append(drop)
                logger.info(
                    "[physics_probe] #%d drop=%.3f %s",
                    i, drop, "stable" if ok else "unstable",
                )
            meta["tried"] = m
            return (
                np.array(stable_mask, dtype=bool),
                np.array(drops_list, dtype=np.float64),
                meta,
            )
        finally:
            _teardown_env_and_process(env, proc)

    chunk_cap = min(eff_bs, m, UNITY_MAX_DUPLICATES)
    num_agents = chunk_cap + 1
    _write_num_duplicates(game_root, num_agents)
    proc = None
    if not no_launch:
        GameModeController(str(game_root)).set_game_runtime_mode()
        proc = GameLauncher().launch(wait=False)
    env = GoiEnv(port=port, num_agents=num_agents, timeout=timeout)
    env.connect()
    meta["deploy_mode"] = "l8_batch"
    meta["num_agents"] = num_agents
    stable_out = np.zeros(m, dtype=bool)
    drops_out = np.zeros(m, dtype=np.float64)
    try:
        _warmup_and_snapshot(env, num_agents, warmup_steps)
        logger.info(
            "[physics_probe] 快照完成（L8 批量，≤%d 点/批，%d agents），探测 %d 个候选",
            eff_bs, num_agents, m,
        )
        offset = 0
        while offset < m:
            take = min(eff_bs, m - offset)
            chunk = candidates[offset : offset + take]
            obs = _deploy_chunk_settle(env, chunk, settle_steps)
            oks, drops, _tip_bads = _chunk_drops_and_stable(
                chunk, obs, settle_drop_threshold, solid_polygons, tip_cfg,
            )
            for j in range(take):
                stable_out[offset + j] = oks[j]
                drops_out[offset + j] = drops[j]
                logger.info(
                    "[physics_probe] #%d drop=%.3f %s",
                    offset + j, drops[j], "stable" if oks[j] else "unstable",
                )
            offset += take
        meta["tried"] = m
        return stable_out, drops_out, meta
    finally:
        _teardown_env_and_process(env, proc)

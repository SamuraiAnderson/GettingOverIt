"""
训练配置有效性离线数值分析。

纯离线：只用 environment.json + TrainConfig，不接游戏、不需要 .pt。
复用 ClimbingEfficiencyMap / build_dynamics / climb_score / step_reward 等训练原语。

用法:
  python -m src.tests.analysis.analyze_effectiveness
  python -m src.tests.analysis.analyze_effectiveness --dt 0.02 --speed-min 1 --speed-max 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 保证可从仓库根目录 / 模块路径导入训练代码
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TESTS_CTRL = _REPO_ROOT / "src" / "tests" / "control_interaction"
if str(_TESTS_CTRL) not in sys.path:
    sys.path.insert(0, str(_TESTS_CTRL))

from src.training.config import TrainConfig
from src.training.dataset import (
    ANGLE_INDICES,
    DYNAMICS_DIM,
    REL_POS_INDICES,
    VELOCITY_INDICES,
)
from src.training.reward import (
    ClimbingEfficiencyMap,
    climb_score,
    step_reward,
)

from .env_geometry import (
    ColliderStats,
    LedgeGeometry,
    build_ledge_geometry,
    collider_stats,
    default_env_path,
    extract_polygons,
    load_environment,
    nearest_collider_gaps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    title: str
    verdict: str  # 够用 / 不足 / 失衡 / 部分够用
    evidence: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_eff_map(
    polygons: list[np.ndarray],
    config: TrainConfig,
    cache_path: Path,
    height_prior_weight: float | None = None,
) -> ClimbingEfficiencyMap:
    """构建效率图（traversable mask 缓存到 disk）。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hp = (
        config.height_prior_weight
        if height_prior_weight is None
        else height_prior_weight
    )
    t0 = time.time()
    eff = ClimbingEfficiencyMap(
        polygons,
        grid_resolution=config.grid_resolution,
        diffusion_iterations=config.diffusion_iterations,
        diffusion_alpha=config.diffusion_alpha,
        cache_path=str(cache_path),
        height_prior_weight=hp,
    )
    logger.info(
        "ClimbingEfficiencyMap 就绪: shape=%s min_g=(%d,%d) 耗时=%.1fs",
        eff.traversable.shape, eff._min_gx, eff._min_gy, time.time() - t0,
    )
    return eff


def geodesic_distance(
    traversable: np.ndarray,
    src_gy: int,
    src_gx: int,
    max_dist: int = 200,
) -> np.ndarray:
    """4-邻域 BFS 测地距离（格点数）。不可达为 -1。"""
    h, w = traversable.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    if not (0 <= src_gy < h and 0 <= src_gx < w) or not traversable[src_gy, src_gx]:
        return dist
    from collections import deque
    q = deque([(src_gy, src_gx)])
    dist[src_gy, src_gx] = 0
    while q:
        gy, gx = q.popleft()
        d = dist[gy, gx]
        if d >= max_dist:
            continue
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < h and 0 <= nx < w and traversable[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = d + 1
                q.append((ny, nx))
    return dist


def pick_seed_cells(
    traversable: np.ndarray,
    min_gx: int,
    min_gy: int,
    resolution: float,
    n: int = 5,
) -> list[tuple[int, int]]:
    """在可通行区域按高度分层挑种子格。"""
    ys, xs = np.where(traversable)
    if len(ys) == 0:
        return []
    world_y = (min_gy + ys + 0.5) * resolution
    # 排除过低（近水）与过高
    mask = (world_y > 5.0) & (world_y < 350.0)
    ys, xs, world_y = ys[mask], xs[mask], world_y[mask]
    if len(ys) == 0:
        return []
    # 按 y 分位数分层
    qs = np.linspace(0.15, 0.85, n)
    seeds = []
    for q in qs:
        target = float(np.quantile(world_y, q))
        idx = int(np.argmin(np.abs(world_y - target)))
        seeds.append((int(ys[idx]), int(xs[idx])))
    # 去重
    uniq = []
    seen = set()
    for s in seeds:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ---------------------------------------------------------------------------
# A. eff_map_range
# ---------------------------------------------------------------------------

def analyze_eff_map_range(
    eff_map: ClimbingEfficiencyMap,
    config: TrainConfig,
    ledges: LedgeGeometry,
    out_dir: Path,
) -> AnalysisResult:
    """扩散半径 vs 上升点间距。"""
    trav = eff_map.traversable
    res = config.grid_resolution
    seeds = pick_seed_cells(trav, eff_map._min_gx, eff_map._min_gy, res, n=5)
    if not seeds:
        return AnalysisResult(
            title="A. 效率图扩散半径 vs 上升点间距",
            verdict="不足",
            evidence=["无可通行种子格，无法测量扩散"],
        )

    # 对每个种子注入单位价值，跑真实 _diffuse
    max_r = 80
    # 聚合：距离 bin → 平均 |∇E|
    bin_sum = np.zeros(max_r + 1, dtype=np.float64)
    bin_cnt = np.zeros(max_r + 1, dtype=np.float64)
    value_at_r = np.zeros(max_r + 1, dtype=np.float64)
    value_cnt = np.zeros(max_r + 1, dtype=np.float64)

    for sy, sx in seeds:
        raw = np.zeros(trav.shape, dtype=np.float64)
        raw[sy, sx] = 1.0
        arr = eff_map._diffuse(raw)

        # 梯度幅值（中心差分）
        gy = np.zeros_like(arr)
        gx = np.zeros_like(arr)
        gy[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / 2.0
        gx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / 2.0
        gmag = np.sqrt(gx ** 2 + gy ** 2)

        dist = geodesic_distance(trav, sy, sx, max_dist=max_r)
        for d in range(max_r + 1):
            cells = (dist == d) & trav
            if not np.any(cells):
                continue
            bin_sum[d] += float(gmag[cells].mean())
            bin_cnt[d] += 1
            value_at_r[d] += float(arr[cells].mean())
            value_cnt[d] += 1

    with np.errstate(invalid="ignore"):
        mean_grad = np.where(bin_cnt > 0, bin_sum / bin_cnt, np.nan)
        mean_val = np.where(value_cnt > 0, value_at_r / value_cnt, np.nan)

    # 有效引导半径 r*：梯度降到种子邻域 (d=1) 的 10%
    g1 = mean_grad[1] if not np.isnan(mean_grad[1]) else np.nanmax(mean_grad)
    threshold = 0.1 * g1 if g1 and not np.isnan(g1) else 1e-4
    r_star = None
    for d in range(1, max_r + 1):
        if np.isnan(mean_grad[d]):
            continue
        if mean_grad[d] < threshold:
            r_star = d * res
            break
    if r_star is None:
        # 用价值衰减到 5% 兜底
        v0 = mean_val[0] if not np.isnan(mean_val[0]) else 1.0
        for d in range(1, max_r + 1):
            if not np.isnan(mean_val[d]) and mean_val[d] < 0.05 * v0:
                r_star = d * res
                break
    if r_star is None:
        r_star = float(max_r * res)

    # 合成轨迹回传：从低处走到高处
    backprop_reach = _synthetic_trajectory_backprop(eff_map, config, seeds)

    gaps = ledges.gaps_euclidean
    gap_p50 = float(np.median(gaps)) if len(gaps) else float("nan")
    gap_p90 = float(np.percentile(gaps, 90)) if len(gaps) else float("nan")
    frac_within = float(np.mean(gaps <= r_star)) if len(gaps) else 0.0
    frac_within_16 = float(np.mean(gaps <= 16.0)) if len(gaps) else 0.0

    # 判定
    if frac_within >= 0.7:
        verdict = "够用"
    elif frac_within >= 0.4:
        verdict = "部分够用"
    else:
        verdict = "不足"

    evidence = [
        f"扩散参数: iterations={config.diffusion_iterations}, "
        f"alpha={config.diffusion_alpha}, 4-邻域",
        f"有效引导半径 r* ≈ {r_star:.1f} m（梯度降至邻域 10%）",
        f"上升点欧氏间距: n={len(gaps)}, p50={gap_p50:.1f} m, p90={gap_p90:.1f} m",
        f"间距 ≤ r* 的比例: {frac_within:.1%}",
        f"间距 ≤ patch 视野(16m) 的比例: {frac_within_16:.1%}",
        f"合成轨迹价值回传可达距离 ≈ {backprop_reach:.1f} m",
    ]
    suggestions = []
    if r_star < gap_p50:
        suggestions.append(
            f"增大 diffusion_iterations（当前 {config.diffusion_iterations}）"
            f"或 diffusion_alpha（当前 {config.diffusion_alpha}），"
            f"使 r*（{r_star:.1f} m）覆盖至少 p50 间距（{gap_p50:.1f} m）"
        )
    elif frac_within < 0.7:
        suggestions.append(
            f"r*={r_star:.1f}m 已略超 p50={gap_p50:.1f}m，但仍有 "
            f"{1 - frac_within:.0%} 上升对超出引导半径；"
            f"可小幅增大 diffusion_iterations / alpha 以覆盖更多尾部"
        )
    if gap_p90 > r_star * 1.5:
        suggestions.append(
            f"p90 间距 {gap_p90:.1f} m 远大于 r*；远距阶段上升点仅靠效率图难以连成梯度，"
            "需配合 random_deploy / 更强 height_prior 或分层课程"
        )
    if not suggestions:
        suggestions.append("当前扩散半径对多数相邻上升点基本够用；关注远距 p90 尾部即可")

    # 图
    fig_path = out_dir / "A_eff_map_range.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    dists = np.arange(max_r + 1) * res
    ax = axes[0]
    ax.plot(dists, mean_grad, label="|grad E| mean", color="C0")
    ax.axvline(r_star, color="C3", ls="--", label=f"r*={r_star:.1f}m")
    ax.axvline(16.0, color="C2", ls=":", label="patch FOV 16m")
    ax.set_xlabel("geodesic distance (m)")
    ax.set_ylabel("mean |grad E|")
    ax.set_title("Eff-map gradient decay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(gaps):
        ax.hist(gaps, bins=40, color="C0", alpha=0.75, edgecolor="k", linewidth=0.3)
        ax.axvline(r_star, color="C3", ls="--", label=f"r*={r_star:.1f}m")
        ax.axvline(16.0, color="C2", ls=":", label="16m")
        ax.axvline(gap_p50, color="C1", ls="-.", label=f"p50={gap_p50:.1f}m")
        ax.legend(fontsize=8)
    ax.set_xlabel("ascent ledge Euclidean gap (m)")
    ax.set_ylabel("count")
    ax.set_title("Ascent gap distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    return AnalysisResult(
        title="A. 效率图扩散半径 vs 上升点间距",
        verdict=verdict,
        evidence=evidence,
        suggestions=suggestions,
        metrics={
            "r_star_m": r_star,
            "gap_p50_m": gap_p50,
            "gap_p90_m": gap_p90,
            "frac_within_rstar": frac_within,
            "frac_within_16m": frac_within_16,
            "backprop_reach_m": backprop_reach,
            "n_ascent_pairs": int(len(gaps)),
            "n_ledge_segments": len(ledges.ledge_segments),
        },
        figures=[str(fig_path.name)],
    )


def _synthetic_trajectory_backprop(
    eff_map: ClimbingEfficiencyMap,
    config: TrainConfig,
    seeds: list[tuple[int, int]],
) -> float:
    """
    构造一条从低到高的合成轨迹，调用 update()，测量价值场相对注入点的回传距离。
    """
    if len(seeds) < 2:
        return 0.0
    # 按世界 y 排序
    res = config.grid_resolution
    pts = []
    for gy, gx in seeds:
        wx = (eff_map._min_gx + gx + 0.5) * res
        wy = (eff_map._min_gy + gy + 0.5) * res
        pts.append((wx, wy, gy, gx))
    pts.sort(key=lambda p: p[1])
    # 线性插值路径
    low, high = pts[0], pts[-1]
    n_steps = max(20, int(np.hypot(high[0] - low[0], high[1] - low[1]) / res))
    xs = np.linspace(low[0], high[0], n_steps)
    ys = np.linspace(low[1], high[1], n_steps)
    # 造伪轨迹对象
    class _T:
        pass
    traj = _T()
    states = np.zeros((n_steps, 33), dtype=np.float32)
    states[:, 0] = xs
    states[:, 1] = ys
    traj.raw_states = states

    # 深拷贝后清掉 height_prior / 旧价值，只测轨迹 update 回传距离
    clean = eff_map.copy()
    h, w = clean.traversable.shape
    clean._max_arr = np.full((h, w), -np.inf, dtype=np.float64)
    clean._arr = None
    clean.diffusion_iters = config.diffusion_iterations
    clean.diffusion_alpha = config.diffusion_alpha

    clean.update([traj], config, prev_eff_map=None)
    if clean._arr is None:
        return 0.0

    # 注入点附近最大值位置
    peak_gy = int(np.floor(high[1] / res)) - clean._min_gy
    peak_gx = int(np.floor(high[0] / res)) - clean._min_gx
    peak_gy = int(np.clip(peak_gy, 0, h - 1))
    peak_gx = int(np.clip(peak_gx, 0, w - 1))

    arr = clean._arr
    peak_val = float(arr[peak_gy, peak_gx])
    if peak_val <= 1e-8:
        peak_val = float(arr.max())
    thresh = 0.05 * peak_val
    # 从 peak 做 BFS，找价值仍 > thresh 的最远测地距离
    dist = geodesic_distance(clean.traversable, peak_gy, peak_gx, max_dist=120)
    reach = 0.0
    mask = (dist >= 0) & (arr >= thresh) & clean.traversable
    if np.any(mask):
        reach = float(dist[mask].max()) * res
    return reach


# ---------------------------------------------------------------------------
# B. patch_fov
# ---------------------------------------------------------------------------

def analyze_patch_fov(
    env_data: dict,
    stats: ColliderStats,
    ledges: LedgeGeometry,
    config: TrainConfig,
    out_dir: Path,
) -> AnalysisResult:
    patch_extent = config.patch_size * config.patch_resolution  # 16m
    half = patch_extent / 2.0  # ±8m
    terrain_eff_res = config.grid_resolution  # crop_centered 先按 1m 裁再上采样

    diag = stats.diag_lengths
    edges = stats.edge_lengths
    nn_gaps = nearest_collider_gaps(stats.centers)

    gaps = ledges.gaps_euclidean
    frac_next_in_patch = float(np.mean(gaps <= half)) if len(gaps) else 0.0
    frac_next_in_full = float(np.mean(gaps <= patch_extent)) if len(gaps) else 0.0

    diag_p50 = float(np.median(diag)) if len(diag) else 0.0
    diag_p90 = float(np.percentile(diag, 90)) if len(diag) else 0.0
    edge_p50 = float(np.median(edges)) if len(edges) else 0.0
    nn_p50 = float(np.median(nn_gaps)) if len(nn_gaps) else 0.0

    # 判定：多数下一台面应在视野内，且地形特征不被 1m 分辨率糊掉
    if frac_next_in_patch >= 0.5 and diag_p50 >= terrain_eff_res:
        verdict = "够用" if frac_next_in_patch >= 0.6 else "部分够用"
    elif frac_next_in_full >= 0.5:
        verdict = "部分够用"
    else:
        verdict = "不足"

    evidence = [
        f"patch 视野: {config.patch_size}×{config.patch_size} @ "
        f"{config.patch_resolution}m → 覆盖 {patch_extent:.0f}m（±{half:.0f}m）",
        f"地形/效率通道有效分辨率 ≈ {terrain_eff_res}m（crop_centered 上采样），"
        f"部件 blob σ≈0.6m / {config.patch_resolution}m",
        f"碰撞体包围盒对角: p50={diag_p50:.2f}m, p90={diag_p90:.2f}m; "
        f"边长 p50={edge_p50:.2f}m",
        f"碰撞体最近邻间距 p50={nn_p50:.2f}m",
        f"下一上升台面落在 ±{half:.0f}m（半视野）内: {frac_next_in_patch:.1%}",
        f"下一上升台面落在 {patch_extent:.0f}m（全视野）内: {frac_next_in_full:.1%}",
    ]
    suggestions = []
    if frac_next_in_patch < 0.5:
        suggestions.append(
            f"多数下一台面超出 ±{half:.0f}m patch；远距导航完全依赖效率图 ch1。"
            f"可增大 patch_size 或 patch_resolution，或强化效率图扩散（见分析 A）"
        )
    if diag_p50 < 2 * terrain_eff_res:
        suggestions.append(
            f"地形特征 p50={diag_p50:.2f}m 接近网格分辨率 {terrain_eff_res}m，"
            "局部几何细节可能被糊掉；考虑降低 grid_resolution 或提高 patch 原生分辨率"
        )
    if not suggestions:
        suggestions.append("patch 对局部接触/勾攀够用；远距阶段依赖效率图是设计预期")

    fig_path = out_dir / "B_patch_fov.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if len(diag):
        ax.hist(diag, bins=40, color="C0", alpha=0.7, label="collider diag")
    if len(nn_gaps):
        ax.hist(nn_gaps, bins=40, color="C1", alpha=0.5, label="nearest gap")
    ax.axvline(half, color="C3", ls="--", label=f"±{half:.0f}m")
    ax.axvline(patch_extent, color="C2", ls=":", label=f"{patch_extent:.0f}m")
    ax.axvline(terrain_eff_res, color="k", ls="-.", label=f"grid {terrain_eff_res}m")
    ax.set_xlabel("length (m)")
    ax.set_ylabel("count")
    ax.set_title("Terrain feature scale vs patch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(gaps):
        ax.hist(gaps, bins=40, color="C0", alpha=0.75, edgecolor="k", linewidth=0.3)
        ax.axvline(half, color="C3", ls="--", label=f"+/-{half:.0f}m half-FOV")
        ax.axvline(patch_extent, color="C2", ls=":", label=f"{patch_extent:.0f}m FOV")
        ax.legend(fontsize=8)
    ax.set_xlabel("next ascent ledge gap (m)")
    ax.set_ylabel("count")
    ax.set_title("Ledge gap vs patch visibility")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    return AnalysisResult(
        title="B. patch 视野 / 分辨率 vs 地形尺度",
        verdict=verdict,
        evidence=evidence,
        suggestions=suggestions,
        metrics={
            "patch_extent_m": patch_extent,
            "half_fov_m": half,
            "diag_p50_m": diag_p50,
            "diag_p90_m": diag_p90,
            "nn_gap_p50_m": nn_p50,
            "frac_next_in_half_fov": frac_next_in_patch,
            "frac_next_in_full_fov": frac_next_in_full,
        },
        figures=[str(fig_path.name)],
    )


# ---------------------------------------------------------------------------
# C. reward_horizon
# ---------------------------------------------------------------------------

def analyze_reward_horizon(
    config: TrainConfig,
    ledges: LedgeGeometry,
    r_star_m: float,
    dt: float,
    speed_min: float,
    speed_max: float,
    out_dir: Path,
) -> AnalysisResult:
    gamma = config.gamma
    # 折扣到 0.05 的步数
    k_eff = int(np.ceil(np.log(0.05) / np.log(gamma))) if gamma < 1 else 10**9
    t_eff = k_eff * dt
    ctx_steps = config.context_len
    ctx_time = ctx_steps * dt
    rollout_steps = config.steps_per_rollout
    rollout_time = rollout_steps * dt

    gaps = ledges.gaps_euclidean
    gap_p50 = float(np.median(gaps)) if len(gaps) else 10.0
    gap_p90 = float(np.percentile(gaps, 90)) if len(gaps) else 30.0

    # 速度敏感性：跨越间距所需步数
    speeds = np.linspace(speed_min, speed_max, 9)
    steps_p50 = gap_p50 / (speeds * dt)
    steps_p90 = gap_p90 / (speeds * dt)

    # 创新高稀疏度合成实验
    spars = _synthetic_sparsity_experiment(config)

    # 台面间 eff_delta 稠密性：若典型间距 > r*，中间段梯度≈0
    dense_ok = gap_p50 <= r_star_m
    frac_covered = float(np.mean(gaps <= r_star_m)) if len(gaps) else 0.0

    # 判定
    mid_speed = 0.5 * (speed_min + speed_max)
    steps_at_mid = gap_p50 / (mid_speed * dt)
    horizon_ok = steps_at_mid <= k_eff
    window_ok = steps_at_mid <= ctx_steps * 2  # 宽松：窗口不必盖全间距

    if horizon_ok and dense_ok and spars["telescope_ok"]:
        verdict = "够用"
    elif horizon_ok or dense_ok:
        verdict = "部分够用"
    else:
        verdict = "不足"

    evidence = [
        f"假设 dt={dt:.4f}s（stepFrames=1）；context={ctx_steps}步≈{ctx_time:.2f}s；"
        f"rollout={rollout_steps}步≈{rollout_time:.1f}s",
        f"gamma={gamma} → 折扣至 0.05 约 {k_eff} 步 ≈ {t_eff:.2f}s",
        f"典型间距 p50={gap_p50:.1f}m / p90={gap_p90:.1f}m；"
        f"速度 [{speed_min},{speed_max}] m/s 时跨 p50 需 "
        f"{steps_p50[-1]:.0f}–{steps_p50[0]:.0f} 步",
        f"中速 {mid_speed:.1f} m/s 跨 p50 ≈ {steps_at_mid:.0f} 步 "
        f"（相对有效视野 {k_eff} 步: {'内' if horizon_ok else '外'}）",
        f"效率图覆盖典型间距: {frac_covered:.1%}（r*={r_star_m:.1f}m）→ "
        f"台面间稠密引导 {'有' if dense_ok else '弱/无'}",
        f"合成 stair 轨迹: 正高度奖励占比={spars['pos_frac']:.1%}, "
        f"望远镜和={spars['telescope_sum']:.3f} vs summit={spars['summit']:.3f} "
        f"({'一致' if spars['telescope_ok'] else '不一致'})",
        f"非创新高步仅靠 neg_reward_scale={config.neg_reward_scale} 回落惩罚 + eff_delta",
    ]
    suggestions = []
    if not horizon_ok:
        suggestions.append(
            f"中速跨台面步数 ({steps_at_mid:.0f}) > gamma 有效视野 ({k_eff})；"
            f"可提高 gamma（如 0.995）或缩短单段目标（分层/deploy）"
        )
    if not dense_ok:
        suggestions.append(
            "台面间距常超出效率图引导半径 → 非创新高段几乎无稠密信号，"
            "长轨迹信用分配失效；优先扩大扩散或引入 shaping（势场奖励）"
        )
    if steps_at_mid > ctx_steps:
        suggestions.append(
            f"跨台面步数 ({steps_at_mid:.0f}) > context_len ({ctx_steps})；"
            "策略无法在同一窗口内看到完整动作序列，依赖效率图记忆/状态压缩"
        )
    if spars["pos_frac"] < 0.15:
        suggestions.append(
            f"创新高正奖励仅占 {spars['pos_frac']:.1%}，信号极稀疏；"
            "确认 eff_delta 在扩散覆盖区内能提供稠密替代"
        )
    if not suggestions:
        suggestions.append("折扣视野与典型间距基本匹配；继续关注 p90 远距尾部")

    fig_path = out_dir / "C_reward_horizon.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ks = np.arange(0, k_eff + 50)
    ax.plot(ks, gamma ** ks, color="C0")
    ax.axhline(0.05, color="C3", ls="--", label="0.05")
    ax.axvline(k_eff, color="C3", ls=":", label=f"k_eff={k_eff}")
    ax.axvline(ctx_steps, color="C2", ls="-.", label=f"context={ctx_steps}")
    ax.set_xlabel("step k")
    ax.set_ylabel(f"gamma^k (gamma={gamma})")
    ax.set_title("Discount curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(speeds, steps_p50, "o-", label="steps for p50 gap")
    ax.plot(speeds, steps_p90, "s-", label="steps for p90 gap")
    ax.axhline(k_eff, color="C3", ls="--", label=f"gamma horizon {k_eff}")
    ax.axhline(ctx_steps, color="C2", ls=":", label=f"context {ctx_steps}")
    ax.axhline(rollout_steps, color="gray", ls="-.", label=f"rollout {rollout_steps}")
    ax.set_xlabel("climb speed (m/s)")
    ax.set_ylabel("steps needed")
    ax.set_title("Gap-crossing steps vs horizon")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = ["pos height", "neg height", "near-zero"]
    vals = [spars["pos_frac"], spars["neg_frac"], spars["zero_frac"]]
    ax.bar(labels, vals, color=["C2", "C3", "C7"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("step fraction")
    ax.set_title("New-max reward sparsity")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    return AnalysisResult(
        title="C. 奖励视野 / 创新高稀疏度 / 台面间稠密引导",
        verdict=verdict,
        evidence=evidence,
        suggestions=suggestions,
        metrics={
            "k_eff_steps": k_eff,
            "t_eff_s": t_eff,
            "context_steps": ctx_steps,
            "gap_p50_m": gap_p50,
            "steps_p50_at_mid_speed": steps_at_mid,
            "frac_gaps_within_rstar": frac_covered,
            "pos_reward_frac": spars["pos_frac"],
            "telescope_ok": spars["telescope_ok"],
        },
        figures=[str(fig_path.name)],
    )


def _synthetic_sparsity_experiment(config: TrainConfig) -> dict:
    """构造 stair-step + 噪声高度剖面，跑 step_reward（无效率图）。"""
    rng = np.random.default_rng(0)
    T = 400
    # 阶梯：每 40 步抬升 2m，中间有回落噪声
    y = np.zeros(T + 1, dtype=np.float64)
    for t in range(1, T + 1):
        stage = (t // 40) * 2.0
        noise = 0.3 * rng.normal()
        # 偶尔明显回落
        if t % 40 == 20:
            noise -= 1.5
        y[t] = max(stage + noise, 0.0)

    running_max = float(y[0])
    rewards = []
    pos_h = neg_h = zero_h = 0
    for t in range(T):
        prev = np.zeros(33, dtype=np.float32)
        curr = np.zeros(33, dtype=np.float32)
        prev[1] = y[t]
        curr[1] = y[t + 1]
        r, running_max = step_reward(
            prev, curr, None, config, running_max, normalizer=None,
        )
        rewards.append(r)
        if r > 1e-8:
            pos_h += 1
        elif r < -1e-8:
            neg_h += 1
        else:
            zero_h += 1

    summit = float(y.max() - y[0])
    # 严格望远镜：只累加创新高正增量
    rm = float(y[0])
    tel = 0.0
    for t in range(1, T + 1):
        if y[t] > rm:
            tel += y[t] - rm
            rm = y[t]
    telescope_ok = abs(tel - summit) < 1e-6

    n = T
    return {
        "pos_frac": pos_h / n,
        "neg_frac": neg_h / n,
        "zero_frac": zero_h / n,
        "summit": summit,
        "telescope_sum": tel,
        "telescope_ok": telescope_ok,
        "mean_abs_reward": float(np.mean(np.abs(rewards))),
    }


# ---------------------------------------------------------------------------
# D. norm_scale
# ---------------------------------------------------------------------------

def analyze_norm_scale(
    config: TrainConfig,
    out_dir: Path,
    y_max_map: float = 380.0,
) -> AnalysisResult:
    # --- 观测量纲 ---
    # 合成低处冷启动轨迹标定 stats
    rng = np.random.default_rng(1)
    cold_trajs = []
    for _ in range(30):
        T = 80
        states = np.zeros((T, 33), dtype=np.float32)
        y0 = float(rng.uniform(0.0, 30.0))  # 冷启动低处
        states[:, 0] = rng.normal(0, 2, T)
        states[:, 1] = y0 + np.cumsum(rng.normal(0.02, 0.05, T))
        # 速度
        for vi in VELOCITY_INDICES:
            states[:, vi] = rng.normal(0, 1.0, T)
        # 角度（度）
        for ai in ANGLE_INDICES:
            states[:, ai] = rng.uniform(-180, 180, T)
        # 相对部件（绝对坐标 = player + 相对）
        for xi, yi in REL_POS_INDICES:
            states[:, xi] = states[:, 0] + rng.normal(0, 0.8, T)
            states[:, yi] = states[:, 1] + rng.normal(0, 0.8, T)

        class _T:
            pass
        tr = _T()
        tr.raw_states = states
        cold_trajs.append(tr)

    from src.training.dataset import compute_dynamics_stats
    mean, std = compute_dynamics_stats(cold_trajs)
    abs_y_idx = DYNAMICS_DIM - 1  # 末维
    assert abs_y_idx == len(VELOCITY_INDICES) + len(ANGLE_INDICES) * 2 + len(REL_POS_INDICES) * 2
    cold_mean_y = float(mean[abs_y_idx])
    cold_std_y = float(std[abs_y_idx])

    # 登高后归一化越界：y 从 cold 到 y_max
    y_probe = np.linspace(0, y_max_map, 50)
    z_score = (y_probe - cold_mean_y) / cold_std_y
    z_at_200 = float((200.0 - cold_mean_y) / cold_std_y)
    z_at_max = float((y_max_map - cold_mean_y) / cold_std_y)

    # sin/cos 维 std
    ang_slice = slice(len(VELOCITY_INDICES), len(VELOCITY_INDICES) + 6)
    ang_std = std[ang_slice]

    # --- 奖励量纲：base_score 分解 ---
    summits = np.array([1, 2, 5, 10, 20, 40], dtype=np.float64)
    peak_steps = np.array([50, 100, 200, 400], dtype=np.float64)
    y_abs_list = np.array([10, 50, 100, 200, 300], dtype=np.float64)

    # 场景网格：固定 peak_step=100，扫 summit × y_abs
    # base_score = climb_score(summit, peak) + y_max * abs_height_weight
    # 其中 y_max = start_y + summit ≈ (y_abs 作为 max)，用 y_abs 代表 max height
    ratio_grid = np.zeros((len(summits), len(y_abs_list)))
    climb_grid = np.zeros_like(ratio_grid)
    abs_grid = np.zeros_like(ratio_grid)
    peak = 100
    for i, s in enumerate(summits):
        for j, ya in enumerate(y_abs_list):
            cs = climb_score(float(s), peak, config)
            ab = float(ya) * config.abs_height_weight
            climb_grid[i, j] = cs
            abs_grid[i, j] = ab
            ratio_grid[i, j] = ab / max(cs, 1e-8)

    # 早期典型：summit=2, y_abs=50（deploy 到中低处）
    early_cs = climb_score(2.0, 100, config)
    early_ab = 50.0 * config.abs_height_weight
    early_ratio = early_ab / max(early_cs, 1e-8)

    # height_prior vs climb_score 型 full_score
    # height_prior at y: 0.05 * y；climb_score 典型个位数~几十
    hp_at = {
        "y=50": 50 * config.height_prior_weight,
        "y=200": 200 * config.height_prior_weight,
        "y=380": 380 * config.height_prior_weight,
    }
    typical_fs = climb_score(5.0, 100, config)  # 典型单段

    # 判定
    abs_dominates = early_ratio > 1.0 or float(np.median(ratio_grid)) > 1.0
    prior_dominates = hp_at["y=200"] > typical_fs
    ood_severe = abs(z_at_200) > 5.0

    if abs_dominates or prior_dominates:
        verdict = "失衡"
    elif ood_severe:
        verdict = "部分够用"
    else:
        verdict = "够用"

    evidence = [
        f"dynamics 维={DYNAMICS_DIM}: vel={len(VELOCITY_INDICES)}, "
        f"ang_sc={len(ANGLE_INDICES)*2}, rel={len(REL_POS_INDICES)*2}, abs_y=1",
        f"冷启动标定 abs_y: mean={cold_mean_y:.2f}, std={cold_std_y:.2f} → "
        f"y=200 时 z={z_at_200:.1f}, y={y_max_map:.0f} 时 z={z_at_max:.1f}",
        f"角度 sin/cos 维 std 范围 [{ang_std.min():.3f}, {ang_std.max():.3f}]",
        f"早期场景 summit=2/peak=100: climb_score={early_cs:.2f}, "
        f"abs_height_bonus={early_ab:.2f}, 比值 bonus/climb={early_ratio:.2f}",
        f"abs_height_weight={config.abs_height_weight}; "
        f"网格中位 bonus/climb={float(np.median(ratio_grid)):.2f}",
        f"height_prior @y=50/200/380 = "
        f"{hp_at['y=50']:.2f}/{hp_at['y=200']:.2f}/{hp_at['y=380']:.2f}；"
        f"典型 climb_score(summit=5)={typical_fs:.2f}",
    ]
    suggestions = []
    if abs_dominates:
        suggestions.append(
            f"降低 abs_height_weight（当前 {config.abs_height_weight}），"
            "避免 BC top-K 退化为按投放绝对高度排序；建议使早期 bonus/climb ≲ 0.3"
        )
    if prior_dominates:
        suggestions.append(
            f"降低 height_prior_weight（当前 {config.height_prior_weight}），"
            "否则效率图近似纯高度场，eff_delta 与创新高奖励高度重叠"
        )
    if ood_severe:
        suggestions.append(
            "冷启动低处标定后 abs_y 在高海拔严重 OOD；"
            "可改为 (y - water_threshold) 再归一化、或按对数/分段标定、"
            "或扩大冷启动 deploy 高度覆盖"
        )
    if ang_std.min() < 0.05:
        suggestions.append("部分 sin/cos 维方差极低，考虑对角度特征免归一化")
    if not suggestions:
        suggestions.append("量纲大致平衡；仍建议监控 BC 筛选是否与绝对高度强相关")

    fig_path = out_dir / "D_norm_scale.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(y_probe, z_score, color="C0")
    ax.axhline(3, color="C3", ls="--", label="|z|=3")
    ax.axhline(-3, color="C3", ls="--")
    ax.axvline(cold_mean_y, color="C2", ls=":", label=f"cold mean={cold_mean_y:.1f}")
    ax.set_xlabel("world y (m)")
    ax.set_ylabel("normalized abs_y z-score")
    ax.set_title("abs_y z-score after climbing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    im = ax.imshow(ratio_grid, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(y_abs_list)))
    ax.set_xticklabels([str(int(v)) for v in y_abs_list])
    ax.set_yticks(range(len(summits)))
    ax.set_yticklabels([str(int(v)) for v in summits])
    ax.set_xlabel("y_abs (max height)")
    ax.set_ylabel("summit (m)")
    ax.set_title("abs_bonus / climb_score (peak=100)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    ys = [50, 100, 200, 300, 380]
    hp = [y * config.height_prior_weight for y in ys]
    cs_ref = [typical_fs] * len(ys)
    x = np.arange(len(ys))
    ax.bar(x - 0.2, hp, width=0.4, label="height_prior", color="C1")
    ax.bar(x + 0.2, cs_ref, width=0.4, label="climb_score(s=5)", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in ys])
    ax.set_xlabel("world y")
    ax.set_ylabel("cell value scale")
    ax.set_title("height_prior vs typical full_score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    return AnalysisResult(
        title="D. 观测归一化与奖励通道量纲平衡",
        verdict=verdict,
        evidence=evidence,
        suggestions=suggestions,
        metrics={
            "cold_mean_y": cold_mean_y,
            "cold_std_y": cold_std_y,
            "z_at_y200": z_at_200,
            "z_at_ymax": z_at_max,
            "early_bonus_over_climb": early_ratio,
            "median_bonus_over_climb": float(np.median(ratio_grid)),
            "height_prior_at_200": hp_at["y=200"],
            "typical_climb_score": typical_fs,
        },
        figures=[str(fig_path.name)],
    )


# ---------------------------------------------------------------------------
# E. summit_reachability —— 势场引导的贪心攀爬可达性（A 的端到端验证）
# ---------------------------------------------------------------------------

def _height_prior_field(eff_map: ClimbingEfficiencyMap) -> np.ndarray:
    """未训练效率图内建的 world_y*height_prior_weight 先验，扩散后（agent 训练前实际见到的场）。"""
    raw = np.where(eff_map._max_arr > -np.inf, eff_map._max_arr, 0.0)
    return eff_map._diffuse(raw)


def _oracle_geodesic_field(
    eff_map: ClimbingEfficiencyMap, summit_cell: tuple[int, int]
) -> np.ndarray:
    """
    最优基线：到山顶的空气测地距离势场（BFS，不受扩散半径截断）。

    value = -geodesic_dist_to_summit（越近山顶越高），不可达为 -inf。
    用它隔离「纯 reach/几何瓶颈」——若连这个理想场都爬不上去，问题在地形断裂而非场质量。
    """
    sy, sx = summit_cell
    dist = geodesic_distance(eff_map.traversable, sy, sx, max_dist=10_000)
    field = np.full(dist.shape, -np.inf, dtype=np.float64)
    reach = dist >= 0
    field[reach] = -dist[reach].astype(np.float64) * eff_map.resolution
    return field


def _field_value(
    arr: np.ndarray, x: float, y: float, min_gx: int, min_gy: int, res: float
) -> float:
    """floor 索引查询势场（与修复后的 reward.query 一致）。越界返回 -inf。"""
    gx = int(np.floor(x / res)) - min_gx
    gy = int(np.floor(y / res)) - min_gy
    if 0 <= gy < arr.shape[0] and 0 <= gx < arr.shape[1]:
        return float(arr[gy, gx])
    return float("-inf")


def _segment_min_distances(segments: list[np.ndarray]) -> np.ndarray:
    """
    段间最小距离矩阵 (S, S)：两条台面段任意点对的最短欧氏距离。

    比中心点距离更忠实于「能否从台面 A 跳到台面 B」——长台面中心可能相距很远，
    但端点几乎相邻。
    """
    s = len(segments)
    dmat = np.zeros((s, s), dtype=np.float64)
    for i in range(s):
        a = segments[i]
        for j in range(i + 1, s):
            b = segments[j]
            d = float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))
            dmat[i, j] = dmat[j, i] = d
    return dmat


def _greedy_climb(
    dist_matrix: np.ndarray,
    values: np.ndarray,
    start_idx: int,
    reach: float,
    max_hops: int = 500,
) -> tuple[int, int]:
    """
    在台面可达图上贪心攀爬：每步跳到 reach 内（段间最小距离）势场值更高的台面，
    直到无更高邻居（卡住）。返回 (终止台面 idx, 跳数)。
    """
    s = dist_matrix.shape[0]
    idx = np.arange(s)
    cur = start_idx
    for hop in range(max_hops):
        reachable = (dist_matrix[cur] <= reach) & (idx != cur)
        if not np.any(reachable):
            return cur, hop
        cand = np.where(reachable)[0]
        best = cand[np.argmax(values[cand])]
        if values[best] <= values[cur] + 1e-9:
            return cur, hop
        cur = best
    return cur, max_hops


def analyze_summit_reachability(
    eff_map: ClimbingEfficiencyMap,
    config: TrainConfig,
    ledges: LedgeGeometry,
    r_star_m: float,
    out_dir: Path,
    n_starts: int = 200,
    seed: int = 7,
) -> AnalysisResult:
    """
    地图采样 + 滤波的端到端验证：势场能否把 agent 从随机低处引到山顶。
    """
    centers = ledges.segment_centers
    if len(centers) < 5:
        return AnalysisResult(
            title="E. 势场引导的贪心攀爬可达性",
            verdict="不足",
            evidence=["台面段过少，无法构建可达图"],
        )

    res = config.grid_resolution
    min_gx, min_gy = eff_map._min_gx, eff_map._min_gy
    y_summit = float(centers[:, 1].max())
    y_floor = float(centers[:, 1].min())
    # 山顶种子取真实最高台面格（保证落在可通行格上，避免段中心均值落到墙内）
    cells = ledges.ledge_cells
    top_cell = cells[int(np.argmax(cells[:, 1]))]
    summit_cell = (
        int(np.floor(top_cell[1] / res)) - min_gy,
        int(np.floor(top_cell[0] / res)) - min_gx,
    )
    success_band = y_summit - 0.05 * (y_summit - y_floor)  # 到达顶部 5% 带内算成功

    # 两种势场：内建 height_prior（实际） vs 到顶测地距离（理想基线）
    fields = {
        "height_prior": _height_prior_field(eff_map),
        "oracle_geodesic": _oracle_geodesic_field(eff_map, summit_cell),
    }
    field_vals = {
        name: np.array(
            [_field_value(arr, c[0], c[1], min_gx, min_gy, res) for c in centers]
        )
        for name, arr in fields.items()
    }

    # 段间最小距离矩阵（比中心距更忠实的可达性）
    seg_dist = _segment_min_distances(ledges.ledge_segments)

    # 采样起点：低处台面（y 在底部 40%）
    rng = np.random.default_rng(seed)
    low_thresh = y_floor + 0.4 * (y_summit - y_floor)
    low_idx = np.where(centers[:, 1] <= low_thresh)[0]
    if len(low_idx) == 0:
        low_idx = np.arange(len(centers))
    starts = rng.choice(low_idx, size=min(n_starts, len(low_idx)), replace=len(low_idx) < n_starts)

    # 扫 reach 半径（连接 A/B：r*、patch 半视野 8m、全视野 16m 等）
    reaches = sorted({4.0, 8.0, round(r_star_m, 1), 16.0, 24.0, 32.0})
    sweep = {name: [] for name in fields}
    stuck_y_default = {}  # reach=16 时卡住高度分布
    default_reach = 16.0

    span = max(y_summit - y_floor, 1e-6)
    frac_default = {}  # reach=16 时爬升高度占比分布
    for name in fields:
        vals = field_vals[name]
        for R in reaches:
            reached = 0
            stuck_ys = []
            start_ys = []
            for s in starts:
                end_idx, _ = _greedy_climb(seg_dist, vals, int(s), R)
                end_y = float(centers[end_idx, 1])
                if end_y >= success_band:
                    reached += 1
                stuck_ys.append(end_y)
                start_ys.append(float(centers[int(s), 1]))
            sweep[name].append(reached / len(starts))
            if abs(R - default_reach) < 1e-6:
                stuck_y_default[name] = np.array(stuck_ys)
                # 爬升占比 = (终点 - 起点) / 总落差，clip 到 [0,1]
                gained = (np.array(stuck_ys) - np.array(start_ys)) / span
                frac_default[name] = np.clip(gained, 0.0, 1.0)

    hp_success_16 = sweep["height_prior"][reaches.index(default_reach)]
    oracle_success_16 = sweep["oracle_geodesic"][reaches.index(default_reach)]
    hp_frac = float(np.median(frac_default["height_prior"]))
    oracle_frac = float(np.median(frac_default["oracle_geodesic"]))
    # 达到 90% 成功所需 reach
    def _reach_for(target: float, name: str) -> float | None:
        for R, sc in zip(reaches, sweep[name]):
            if sc >= target:
                return R
        return None
    hp_reach90 = _reach_for(0.9, "height_prior")
    oracle_reach90 = _reach_for(0.9, "oracle_geodesic")

    # 判定以「爬升高度占比」为主（二值到顶对 460m 山过苛）：
    #   oracle 也爬不高 → 几何断裂（reach 瓶颈）；oracle 高但 hp 低 → 场质量瓶颈
    if hp_frac >= 0.7:
        verdict = "够用"
    elif oracle_frac >= 0.7:
        verdict = "部分够用"
    else:
        verdict = "不足"

    evidence = [
        f"台面可达图: {len(centers)} 段, 顶 y={y_summit:.0f}, 底 y={y_floor:.0f}; "
        f"随机低处起点 {len(starts)} 个 (y≤{low_thresh:.0f})",
        f"到顶判据: 终点 y ≥ {success_band:.0f}（顶部 5% 带，对高山偏严，仅作参考）",
        f"**爬升高度占比中位（reach=16m）: height_prior={hp_frac:.1%}, "
        f"oracle={oracle_frac:.1%}**（1.0=从起点爬到顶）",
        f"内建 height_prior @reach=16m 到顶率: {hp_success_16:.1%}"
        + (f"；到顶率达 90% 需 reach≈{hp_reach90:.0f}m" if hp_reach90 else "；扫描内到顶率未达 90%"),
        f"oracle(到顶测地距离场) @reach=16m 到顶率: {oracle_success_16:.1%}"
        + (f"；到顶率达 90% 需 reach≈{oracle_reach90:.0f}m" if oracle_reach90 else "；扫描内到顶率未达 90%"),
        f"height_prior @reach=16m 卡住高度中位: "
        f"{float(np.median(stuck_y_default['height_prior'])):.0f} (顶 {y_summit:.0f})",
        "⚠️ 方法学局限: 台面图只建模「可着陆平台」间的跳跃可达性，"
        "不表示 GOI 中靠锤子摩擦攀爬垂直墙面，故对可达性偏保守（低估）；"
        "结论应理解为「靠平台跳跃的贪心策略」的下界，而非物理上限",
    ]
    suggestions = []
    if hp_frac < 0.7 and oracle_frac >= 0.7:
        suggestions.append(
            f"oracle 场能爬到 {oracle_frac:.0%} 而 height_prior 仅 {hp_frac:.0%} → "
            "瓶颈在**场质量**而非 reach；训练早期效率图未被轨迹填充前 agent 缺乏跨台面梯度，"
            "需靠 PPO 逐步创新高 / 更强 waypoint 传播补齐"
        )
    if oracle_frac < 0.7:
        if oracle_reach90:
            need_txt = f"需 reach≈{oracle_reach90:.0f}m 才达 90% 到顶"
        else:
            need_txt = f"即便扫描上限 reach={reaches[-1]:.0f}m 仍无法稳定到顶（存在极远断裂）"
        suggestions.append(
            f"即便最优势场，reach=16m 也只爬到 {oracle_frac:.0%} → 存在**几何断裂**的孤立台面；"
            f"{need_txt}，远超 patch 半视野(8m)/r*({r_star_m:.0f}m)，"
            "提示远距阶段应靠 random_deploy 分段投放而非单条长轨迹"
        )
    if hp_reach90 and hp_reach90 > r_star_m * 1.5:
        suggestions.append(
            f"height_prior 达 90% 到顶需 reach≈{hp_reach90:.0f}m，明显大于扩散半径 "
            f"r*={r_star_m:.0f}m；扩散半径是长程引导的主要短板（呼应分析 A）"
        )
    if not suggestions:
        suggestions.append("势场在合理 reach 下基本能引导到顶；关注被判为陷阱的孤立台面尾部")

    # 图
    fig_path = out_dir / "E_summit_reachability.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for name in fields:
        ax.plot(reaches, sweep[name], "o-", label=name)
    ax.axhline(0.7, color="C3", ls="--", label="0.7")
    ax.axvline(8.0, color="C2", ls=":", label="patch half 8m")
    ax.axvline(r_star_m, color="C1", ls="-.", label=f"r*={r_star_m:.0f}m")
    ax.set_xlabel("reach radius (m)")
    ax.set_ylabel("summit success rate")
    ax.set_title("Greedy climb success vs reach")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in fields:
        if name in stuck_y_default:
            ax.hist(stuck_y_default[name], bins=30, alpha=0.55, label=name)
    ax.axvline(y_summit, color="C3", ls="--", label=f"summit {y_summit:.0f}")
    ax.axvline(success_band, color="C1", ls=":", label="success band")
    ax.set_xlabel("stuck height y (m) @reach=16m")
    ax.set_ylabel("count")
    ax.set_title("Where greedy climb gets stuck")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # 陷阱台面（height_prior, reach=16m 局部极大）散点
    vals = field_vals["height_prior"]
    idx_all = np.arange(len(centers))
    trap = np.zeros(len(centers), dtype=bool)
    for i in range(len(centers)):
        reachable = (seg_dist[i] <= default_reach) & (idx_all != i)
        if not np.any(reachable) or vals[reachable].max() <= vals[i] + 1e-9:
            trap[i] = True
    ax.scatter(centers[~trap, 0], centers[~trap, 1], s=6, c="C0", label="ledge")
    ax.scatter(centers[trap, 0], centers[trap, 1], s=16, c="C3", marker="x", label="trap (local max)")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Trap ledges @reach=16m (height_prior)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    return AnalysisResult(
        title="E. 势场引导的贪心攀爬可达性",
        verdict=verdict,
        evidence=evidence,
        suggestions=suggestions,
        metrics={
            "n_starts": int(len(starts)),
            "y_summit": y_summit,
            "success_band_y": success_band,
            "hp_climb_frac_reach16": hp_frac,
            "oracle_climb_frac_reach16": oracle_frac,
            "hp_success_at_reach16": hp_success_16,
            "oracle_success_at_reach16": oracle_success_16,
            "hp_reach_for_90pct": hp_reach90 if hp_reach90 else -1.0,
            "oracle_reach_for_90pct": oracle_reach90 if oracle_reach90 else -1.0,
            "n_trap_ledges_reach16": int(trap.sum()),
        },
        figures=[str(fig_path.name)],
    )


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def write_report(
    out_dir: Path,
    config: TrainConfig,
    dt: float,
    speed_min: float,
    speed_max: float,
    results: list[AnalysisResult],
    map_bounds: dict,
) -> Path:
    path = out_dir / "effectiveness_report.md"
    lines: list[str] = []
    lines.append("# 训练配置有效性离线数值分析报告\n")
    lines.append("## 假设与输入\n")
    lines.append(f"- 环境: `checkpoints/environment.json`")
    lines.append(
        f"- 地图范围: x∈[{map_bounds['x_min']:.1f},{map_bounds['x_max']:.1f}], "
        f"y∈[{map_bounds['y_min']:.1f},{map_bounds['y_max']:.1f}]"
    )
    lines.append(f"- **假设** `fixedDeltaTime dt = {dt:.4f} s`，`stepFrames = 1`")
    lines.append(
        f"- **假设** 攀爬速度区间 `[{speed_min}, {speed_max}] m/s`（用于跨间距步数换算）"
    )
    lines.append(
        f"- 关键配置: context_len={config.context_len}, "
        f"steps_per_rollout={config.steps_per_rollout}, gamma={config.gamma}, "
        f"gae_lambda={config.gae_lambda}"
    )
    lines.append(
        f"- 空间: patch={config.patch_size}@{config.patch_resolution}m, "
        f"grid={config.grid_resolution}m, "
        f"diffusion_iters={config.diffusion_iterations}, "
        f"alpha={config.diffusion_alpha}"
    )
    lines.append(
        f"- 奖励权重: efficiency_weight={config.efficiency_weight}, "
        f"speed_ref_steps={config.speed_ref_steps}, "
        f"waypoint_weight={config.waypoint_weight}, "
        f"abs_height_weight={config.abs_height_weight}, "
        f"height_prior_weight={config.height_prior_weight}, "
        f"reward_alpha={config.reward_alpha}, "
        f"neg_reward_scale={config.neg_reward_scale}"
    )
    lines.append("")
    lines.append("## 总览\n")
    lines.append("| 维度 | 结论 |")
    lines.append("|------|------|")
    for r in results:
        lines.append(f"| {r.title} | **{r.verdict}** |")
    lines.append("")

    for r in results:
        lines.append(f"## {r.title}\n")
        lines.append(f"**结论: {r.verdict}**\n")
        lines.append("### 数值证据\n")
        for e in r.evidence:
            lines.append(f"- {e}")
        lines.append("\n### 参数建议\n")
        for s in r.suggestions:
            lines.append(f"- {s}")
        if r.metrics:
            lines.append("\n### 指标摘要\n")
            lines.append("```")
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.4g}")
                else:
                    lines.append(f"{k}: {v}")
            lines.append("```")
        if r.figures:
            lines.append("\n### 图\n")
            for f in r.figures:
                lines.append(f"![{f}]({f})")
        lines.append("")

    lines.append("## 综合建议（按优先级）\n")
    # 收集所有 suggestions，不足/失衡优先
    prio = []
    for r in results:
        weight = {"失衡": 0, "不足": 1, "部分够用": 2, "够用": 3}.get(r.verdict, 2)
        for s in r.suggestions:
            prio.append((weight, r.verdict, s))
    prio.sort(key=lambda x: x[0])
    for i, (_, v, s) in enumerate(prio, 1):
        lines.append(f"{i}. [{v}] {s}")
    lines.append("")
    lines.append("---\n*本报告由 `python -m src.tests.analysis.analyze_effectiveness` 生成，不修改训练代码。*\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写: %s", path)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练配置有效性离线数值分析")
    p.add_argument("--dt", type=float, default=0.02, help="物理帧时长假设 (s)")
    p.add_argument("--speed-min", type=float, default=1.0, help="假设攀爬速度下限 m/s")
    p.add_argument("--speed-max", type=float, default=5.0, help="假设攀爬速度上限 m/s")
    p.add_argument(
        "--out",
        type=str,
        default=str(_REPO_ROOT / "src" / "Data" / "analysis"),
        help="输出目录",
    )
    p.add_argument(
        "--env",
        type=str,
        default=str(default_env_path()),
        help="environment.json 路径",
    )
    return p.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig()
    env_data = load_environment(Path(args.env))
    polygons = extract_polygons(env_data)
    all_pts = np.concatenate(polygons)
    map_bounds = {
        "x_min": float(all_pts[:, 0].min()),
        "x_max": float(all_pts[:, 0].max()),
        "y_min": float(all_pts[:, 1].min()),
        "y_max": float(all_pts[:, 1].max()),
    }

    cache_path = _REPO_ROOT / "checkpoints" / "cache" / "traversable_mask.npz"
    eff_map = build_eff_map(polygons, config, cache_path)

    logger.info("抽取台面几何...")
    ledges = build_ledge_geometry(
        eff_map.traversable, eff_map._min_gx, eff_map._min_gy, config.grid_resolution,
    )
    cstats = collider_stats(env_data)

    results: list[AnalysisResult] = []

    logger.info("=== A. eff_map_range ===")
    res_a = analyze_eff_map_range(eff_map, config, ledges, out_dir)
    results.append(res_a)
    logger.info("A 结论: %s", res_a.verdict)

    logger.info("=== B. patch_fov ===")
    res_b = analyze_patch_fov(env_data, cstats, ledges, config, out_dir)
    results.append(res_b)
    logger.info("B 结论: %s", res_b.verdict)

    logger.info("=== C. reward_horizon ===")
    r_star = float(res_a.metrics.get("r_star_m", 20.0))
    res_c = analyze_reward_horizon(
        config, ledges, r_star, args.dt, args.speed_min, args.speed_max, out_dir,
    )
    results.append(res_c)
    logger.info("C 结论: %s", res_c.verdict)

    logger.info("=== D. norm_scale ===")
    res_d = analyze_norm_scale(config, out_dir, y_max_map=config.y_max_cutoff)
    results.append(res_d)
    logger.info("D 结论: %s", res_d.verdict)

    logger.info("=== E. summit_reachability ===")
    res_e = analyze_summit_reachability(eff_map, config, ledges, r_star, out_dir)
    results.append(res_e)
    logger.info("E 结论: %s", res_e.verdict)

    write_report(
        out_dir, config, args.dt, args.speed_min, args.speed_max, results, map_bounds,
    )
    logger.info("完成。输出目录: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

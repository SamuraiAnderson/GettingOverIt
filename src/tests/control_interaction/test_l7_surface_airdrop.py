"""
层次 7：环境上表面空投测试 — Phase 1 几何分析 + 可视化

加载环境碰撞体和 Player 轮廓（cache 优先），计算所有可着陆表面
（包括内部表面，只要上方空间高度 > Player 包围盒 + padding），
在 matplotlib 中绘制完整地图并标注投放点。

用法：
  python src/tests/control_interaction/test_l7_surface_airdrop.py
  python src/tests/control_interaction/test_l7_surface_airdrop.py --n-drops 20 --drop-height 3.0
  python src/tests/control_interaction/test_l7_surface_airdrop.py --no-launch  # 仅使用 cache 文件
  python src/tests/control_interaction/test_l7_surface_airdrop.py --padding 0.5
  python src/tests/control_interaction/test_l7_surface_airdrop.py --physics-filter  # 游戏内过滤不稳定点
  python src/tests/control_interaction/test_l7_surface_airdrop.py --physics-filter --physics-no-launch
  python src/tests/control_interaction/test_l7_surface_airdrop.py --physics-filter-all  # 全量候选探测；
      # 稳定点写入 drop_points_all_stable.json，drop_points.json 仅保留 L8 子集（≤64 等）
  python src/tests/control_interaction/test_l7_surface_airdrop.py --physics-sequential-probes  # 逐点 2-agent（旧逻辑）
  python src/tests/control_interaction/test_l7_surface_airdrop.py --physics-l8-batch-size 16  # 批量每批最多 16 点
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.deploy_sampling import (
    build_all_deploy_candidates,
    compute_drop_points,
    sample_deploy_candidates_stratified,
)

logger = logging.getLogger(__name__)

# L8 / Unity：复制体数量上限（与 PlayerDuplicateManager.MAX_DUPLICATES 一致）
UNITY_L8_MAX_DROP_POINTS = 64
DROP_POINTS_ALL_STABLE_FILENAME = "drop_points_all_stable.json"

_CONFIG_PATH = Path(__file__).parent / "test_config.json"
_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"
_REPORT_DIR = _REPO_ROOT / "src" / "Data" / "GameResults"


# ── 数据加载 ────────────────────────────────────────────────────

def _get_game_root(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Path(cfg["game"]["executable_path"]).parent


def load_colliders(
    game_root: Path,
    env=None,
) -> tuple[dict, dict]:
    """
    加载环境和 Player 轮廓 JSON。

    environment.json 存放于 checkpoints/（仓库目录），
    player_contour.json 存放于 game_root/GoiData/Colliders/。
    若文件不存在且 env 不为 None，则通过 TCP 请求导出后复制到 checkpoints/。
    """
    import shutil

    env_path = _REPO_ROOT / "checkpoints" / "environment.json"
    colliders_dir = game_root / "GoiData" / "Colliders"
    player_path = colliders_dir / "player_contour.json"

    if not env_path.exists() or not player_path.exists():
        if env is not None:
            logger.info("碰撞体文件不存在，通过 TCP 请求导出...")
            env.export_colliders()
            import time
            time.sleep(0.5)
            # C# 导出到游戏目录，复制 environment.json 到 checkpoints/
            game_env = colliders_dir / "environment.json"
            if game_env.exists():
                env_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(game_env, env_path)
                logger.info("environment.json 已复制到 %s", env_path)
        else:
            raise FileNotFoundError(
                f"碰撞体文件不存在: env={env_path}, player={player_path}\n"
                "请先运行游戏导出，或使用 --no-launch 以外的模式"
            )

    with open(env_path, "r", encoding="utf-8") as f:
        env_data = json.load(f)
    with open(player_path, "r", encoding="utf-8") as f:
        player_data = json.load(f)

    n_colliders = len(env_data.get("colliders", []))
    n_parts = len(player_data.get("parts", {}))
    logger.info("已加载碰撞体: 环境 %d 个 collider, Player %d 个部件", n_colliders, n_parts)

    return env_data, player_data


# ── 几何计算 ────────────────────────────────────────────────────

_EXCLUDED_COLLIDERS = {"Snake"}


def extract_polygons(env_data: dict) -> list[np.ndarray]:
    """提取所有环境多边形，返回顶点数组列表，每个 shape (N, 2)。

    跳过 _EXCLUDED_COLLIDERS 中列出的碰撞体（如 Snake）。
    """
    polys = []
    skipped = 0
    for collider in env_data.get("colliders", []):
        if collider.get("name") in _EXCLUDED_COLLIDERS:
            skipped += 1
            continue
        for path in collider.get("paths", []):
            pts = np.array(path, dtype=np.float64)
            if len(pts) >= 3:
                polys.append(pts)
    if skipped:
        logger.info("已过滤 %d 个排除碰撞体: %s", skipped, _EXCLUDED_COLLIDERS)
    return polys


def compute_player_height(player_data: dict) -> float:
    """
    计算 Player 所有碰撞部件的总包围盒高度（本地坐标）。
    包含 body, tip, pot, pot_sides。
    """
    all_y = []
    for part in player_data.get("parts", {}).values():
        for path in part.get("paths", []):
            for pt in path:
                all_y.append(pt[1])
    if not all_y:
        logger.warning("Player 轮廓数据为空，使用默认高度 2.0")
        return 2.0
    height = max(all_y) - min(all_y)
    logger.info("Player 包围盒高度: %.3f (y_min=%.3f, y_max=%.3f)",
                height, min(all_y), max(all_y))
    return height


def _polygon_y_crossings(poly: np.ndarray, x: float) -> list[float]:
    """找出一个多边形在垂直线 x 处的所有 y 交点。"""
    n = len(poly)
    crossings = []
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = poly[i]
        x2, y2 = poly[j]
        dx = x2 - x1
        if abs(dx) < 1e-12:
            continue
        xlo, xhi = (x1, x2) if x1 <= x2 else (x2, x1)
        if xlo <= x <= xhi:
            t = (x - x1) / dx
            crossings.append(y1 + t * (y2 - y1))
    return crossings


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """合并重叠/相邻的 y 区间，返回按 lo 升序排列的区间列表。"""
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + 1e-4:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(a, b) for a, b in merged]


def _surfaces_at_x(polygons: list[np.ndarray], x: float,
                    min_clearance: float) -> list[float]:
    """在给定 x 处计算所有可着陆表面的 y 值。"""
    solid_intervals = []
    for poly in polygons:
        crossings = _polygon_y_crossings(poly, x)
        crossings.sort()
        for k in range(0, len(crossings) - 1, 2):
            lo, hi = crossings[k], crossings[k + 1]
            if hi > lo + 1e-6:
                solid_intervals.append((lo, hi))
    if not solid_intervals:
        return []
    merged = _merge_intervals(solid_intervals)
    surfaces = []
    for idx, (lo, hi) in enumerate(merged):
        air_gap = (merged[idx + 1][0] - hi) if idx < len(merged) - 1 else float("inf")
        if air_gap >= min_clearance:
            surfaces.append(hi)
    return surfaces


def compute_landable_surfaces(
    polygons: list[np.ndarray],
    min_clearance: float,
    resolution: float = 0.1,
    y_tolerance: float = 2.0,
    y_max_cutoff: float = float("inf"),
) -> list[np.ndarray]:
    """
    计算所有可着陆表面，返回独立的连续线段列表。

    使用链跟踪（chain tracking）：在每个采样 x 处，将当前表面 y 值
    与已有的活跃链进行最近邻匹配。匹配成功则延伸链，否则开新链。
    连续 x 处 y 差值 > y_tolerance 的不视为同一表面。
    y > y_max_cutoff 的表面点会被过滤掉。

    返回 list[np.ndarray]，每个元素 shape (N, 2) 为一条连续表面线段。
    """
    all_x_vals = np.concatenate([p[:, 0] for p in polygons])
    x_min, x_max = float(all_x_vals.min()), float(all_x_vals.max())
    sample_x = np.arange(x_min, x_max, resolution)

    # active_chains: list of [last_y, [(x,y), ...]]
    active_chains: list[list] = []
    completed: list[list] = []

    for x in sample_x:
        surfaces_y = _surfaces_at_x(polygons, x, min_clearance)
        if y_max_cutoff < float("inf"):
            surfaces_y = [y for y in surfaces_y if y <= y_max_cutoff]

        if not surfaces_y:
            for _, pts in active_chains:
                completed.append(pts)
            active_chains = []
            continue

        # Greedy nearest-neighbor matching: surface → chain
        used_chains: set[int] = set()
        used_surfaces: set[int] = set()
        matches: list[tuple[int, int]] = []

        for si, sy in enumerate(surfaces_y):
            best_ci, best_dist = -1, y_tolerance
            for ci, (last_y, _) in enumerate(active_chains):
                if ci in used_chains:
                    continue
                dist = abs(sy - last_y)
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci
            if best_ci >= 0:
                matches.append((si, best_ci))
                used_chains.add(best_ci)
                used_surfaces.add(si)

        # Extend matched chains
        for si, ci in matches:
            active_chains[ci][1].append((x, surfaces_y[si]))
            active_chains[ci][0] = surfaces_y[si]

        # Close unmatched chains
        new_active = []
        for ci, chain in enumerate(active_chains):
            if ci in used_chains:
                new_active.append(chain)
            else:
                completed.append(chain[1])

        # Start new chains for unmatched surfaces
        for si, sy in enumerate(surfaces_y):
            if si not in used_surfaces:
                new_active.append([sy, [(x, sy)]])

        active_chains = new_active

    # Flush remaining active chains
    for _, pts in active_chains:
        completed.append(pts)

    # Convert to numpy, filter tiny segments
    segments = []
    for pts in completed:
        if len(pts) >= 2:
            segments.append(np.array(pts, dtype=np.float64))

    return segments


# ── 可视化 ──────────────────────────────────────────────────────

def plot_map(
    env_data: dict,
    player_data: dict,
    segments: list[np.ndarray],
    drop_points: np.ndarray,
    drop_height: float,
    player_height: float,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # --- 环境多边形 ---
    env_patches = []
    for collider in env_data.get("colliders", []):
        if collider.get("name") in _EXCLUDED_COLLIDERS:
            continue
        for path in collider.get("paths", []):
            pts = np.array(path, dtype=np.float64)
            if len(pts) >= 3:
                env_patches.append(plt.Polygon(pts, closed=True))
    if env_patches:
        pc = PatchCollection(env_patches, facecolor="#d0d0d0", edgecolor="#555555",
                             linewidth=0.3, alpha=0.8)
        ax.add_collection(pc)

    # --- 可着陆表面（每条线段用不同颜色） ---
    cmap = plt.cm.get_cmap("tab10")
    for i, seg in enumerate(segments):
        color = cmap(i % 10)
        label = "landable surface" if i == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.5,
                zorder=3, label=label)

    # --- Player 轮廓（原点处） ---
    part_colors = {
        "body": ("#cccc00", "Body"),
        "tip": ("#ff3333", "Tip"),
        "pot": ("#00cc00", "Pot"),
        "pot_sides": ("#00cccc", "Pot Sides"),
    }
    for part_key, (color, label) in part_colors.items():
        part = player_data.get("parts", {}).get(part_key)
        if part is None:
            continue
        for path in part.get("paths", []):
            pts = np.array(path, dtype=np.float64)
            if len(pts) >= 2:
                closed = np.vstack([pts, pts[0:1]])
                ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5,
                        label=label, zorder=4)
                label = None

    # --- 投放点 ---
    if len(drop_points) > 0:
        ax.scatter(drop_points[:, 0], drop_points[:, 1], color="blue", s=40,
                   zorder=5, label=f"drop points (h={drop_height:.1f})")
        for dx, dy in drop_points:
            ax.annotate("", xy=(dx, dy - drop_height), xytext=(dx, dy),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1.0))

    ax.set_aspect("equal")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_title(f"L7: Landable Surfaces & Airdrop Points "
                 f"(clearance >= {player_height:.2f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("图片已保存: %s", save_path)

    plt.show()


# ── 主流程 ──────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="L7: 环境上表面空投测试 (Phase 1 — 可视化)")
    parser.add_argument("--n-drops", type=int, default=None, help="投放点数量")
    parser.add_argument("--drop-height", type=float, default=None, help="投放高度（上表面上方距离）")
    parser.add_argument("--padding", type=float, default=None,
                        help="在 Player 包围盒高度之上额外添加的间隙 padding")
    parser.add_argument("--height-decay", type=float, default=None,
                        help="高度衰减系数 [0,1]：0=纯弧长, 0.5=中等偏低, 0.9=强偏低")
    parser.add_argument("--y-max-cutoff", type=float, default=None,
                        help="屏蔽 y 坐标超过此值的高空表面")
    parser.add_argument("--game-root", type=str, default=None, help="游戏根目录")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--no-launch", action="store_true", help="不启动游戏，仅使用 cache 文件")
    parser.add_argument("--save-path", type=str, default=None, help="图片保存路径")
    parser.add_argument("--no-plot", action="store_true", help="不显示图片")
    parser.add_argument(
        "--physics-filter",
        action="store_true",
        help="启动游戏，按 settle_drop_threshold 过滤不稳定空投点（需 TCP；与 PPO _get_stable_agents 一致）",
    )
    parser.add_argument(
        "--physics-filter-oversample",
        type=int,
        default=3,
        help="物理过滤时候选池大小 = n_drops × 该系数（默认 3）",
    )
    parser.add_argument(
        "--physics-filter-max-candidates",
        type=int,
        default=None,
        help="最多尝试多少个候选（默认不限制，直至候选池用尽）",
    )
    parser.add_argument(
        "--physics-warmup-steps",
        type=int,
        default=None,
        help="物理过滤前 warmup 步数（默认 TrainConfig.warmup_steps）",
    )
    parser.add_argument(
        "--physics-settle-steps",
        type=int,
        default=None,
        help="每次传送后 settle 步数（默认 TrainConfig.settle_steps）",
    )
    parser.add_argument(
        "--settle-drop-threshold",
        type=float,
        default=None,
        help="竖直落差上限 teleport_y−settled_y（默认 TrainConfig.settle_drop_threshold）",
    )
    parser.add_argument(
        "--physics-no-launch",
        action="store_true",
        help="物理过滤时不启动游戏（请自行先开游戏并处于 GameRuntime 模式）",
    )
    parser.add_argument(
        "--physics-filter-all",
        action="store_true",
        help=(
            "全量 deploy 候选（顶点或弧长分层采样）物理探测；稳定点写入 "
            f"{DROP_POINTS_ALL_STABLE_FILENAME}，drop_points.json 仅 L8 子集"
        ),
    )
    parser.add_argument(
        "--physics-max-candidates",
        type=int,
        default=None,
        help="--physics-filter-all 时候选池上限（默认 800；顶点数更少则用全顶点）",
    )
    parser.add_argument(
        "--l8-max-drops",
        type=int,
        default=64,
        help=f"写入 drop_points.json 的条数上限（默认 64，且不超过 {UNITY_L8_MAX_DROP_POINTS}）",
    )
    parser.add_argument(
        "--physics-rng-seed",
        type=int,
        default=42,
        help="候选采样随机种子（预留；当前弧长采样为确定性）",
    )
    parser.add_argument(
        "--physics-sequential-probes",
        action="store_true",
        help="物理过滤用逐点 2-agent 探测；默认用 L8 批量同时传送+统一 settle",
    )
    parser.add_argument(
        "--physics-l8-batch-size",
        type=int,
        default=64,
        help="L8 批量探测时每批最多点数（1–64，默认 64）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config()
    l7 = cfg.get("l7", {})
    port = args.port or cfg.get("port", 9000)
    timeout = args.timeout or cfg.get("connect_timeout", 60)
    n_drops = args.n_drops or l7.get("n_drops", 10)
    drop_height = args.drop_height if args.drop_height is not None else l7.get("drop_height", 2.0)
    padding = args.padding if args.padding is not None else l7.get("padding", 0.3)
    height_decay = args.height_decay if args.height_decay is not None else l7.get("height_decay", 0.0)
    y_max_cutoff = args.y_max_cutoff if args.y_max_cutoff is not None else l7.get("y_max_cutoff", 380.0)
    save_path = args.save_path or str(_REPORT_DIR / "l7_surface_map.png")

    game_root = _get_game_root(args.game_root)
    logger.info("游戏根目录: %s", game_root)

    colliders_dir = game_root / "GoiData" / "Colliders"
    has_cache = (_REPO_ROOT / "checkpoints" / "environment.json").exists() and \
                (colliders_dir / "player_contour.json").exists()

    env_conn = None
    if not has_cache and not args.no_launch:
        from env.goi_env import GoiEnv
        from start.game_launcher import GameLauncher
        from start.game_mode_controller import GameModeController

        logger.info("Cache 不存在，启动游戏并请求导出...")
        GameModeController().set_game_runtime_mode()
        GameLauncher().launch(wait=False)

        env_conn = GoiEnv(port=port, num_agents=1, timeout=timeout)
        env_conn.connect()
        env_conn.reset()

    env_data, player_data = load_colliders(game_root, env=env_conn)

    if env_conn is not None:
        env_conn.close()

    player_height = compute_player_height(player_data)
    min_clearance = player_height + padding
    logger.info("最小净空高度: %.3f (player=%.3f + padding=%.3f)",
                min_clearance, player_height, padding)

    polygons = extract_polygons(env_data)
    logger.info("提取到 %d 个多边形", len(polygons))

    logger.info("y_max_cutoff: %.1f", y_max_cutoff)

    segments = compute_landable_surfaces(polygons, min_clearance, resolution=0.1,
                                         y_max_cutoff=y_max_cutoff)
    total_pts = sum(len(s) for s in segments)
    logger.info("可着陆表面: %d 条线段, %d 个点", len(segments), total_pts)

    train_cfg = TrainConfig()
    settle_thr = (
        args.settle_drop_threshold
        if args.settle_drop_threshold is not None
        else train_cfg.settle_drop_threshold
    )
    phys_warmup = (
        args.physics_warmup_steps
        if args.physics_warmup_steps is not None
        else train_cfg.warmup_steps
    )
    phys_settle = (
        args.physics_settle_steps
        if args.physics_settle_steps is not None
        else train_cfg.settle_steps
    )
    use_l8_batch = not args.physics_sequential_probes
    l8_bs = max(1, min(int(args.physics_l8_batch_size), UNITY_L8_MAX_DROP_POINTS))

    physics_meta: dict | None = None
    n_candidates_generated = 0
    physics_filter_mode: str | None = None
    stable_all_count = 0
    l8_export_count = 0
    stable_all_filename: str | None = None

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if args.physics_filter_all:
        if args.physics_filter:
            logger.warning("同时指定 --physics-filter 与 --physics-filter-all，仅执行 all 模式")
        from drop_point_physics import filter_all_drop_points_stable

        physics_filter_mode = "all"
        max_k = args.physics_max_candidates if args.physics_max_candidates is not None else 800
        rng = np.random.default_rng(args.physics_rng_seed)
        candidates = sample_deploy_candidates_stratified(
            segments, drop_height, max_k, rng, height_decay=0.0,
        )
        n_candidates_generated = len(candidates)
        logger.info(
            "physics_filter_all: 候选池 %d 点 (上限=%d)，阈值=%.2f settle=%d warmup=%d",
            n_candidates_generated, max_k,
            settle_thr, phys_settle, phys_warmup,
        )
        stable_all, physics_meta = filter_all_drop_points_stable(
            candidates,
            max_probes=None,
            settle_drop_threshold=settle_thr,
            settle_steps=phys_settle,
            warmup_steps=phys_warmup,
            port=port,
            game_root=game_root,
            timeout=timeout,
            no_launch=args.physics_no_launch,
            use_l8_batch_deploy=use_l8_batch,
            l8_batch_size=l8_bs,
        )
        stable_all_count = len(stable_all)
        stable_all_filename = DROP_POINTS_ALL_STABLE_FILENAME
        stable_path = colliders_dir / stable_all_filename
        stable_payload = {
            "params": {
                "drop_height": drop_height,
                "settle_drop_threshold": settle_thr,
                "physics_warmup_steps": phys_warmup,
                "physics_settle_steps": phys_settle,
                "physics_max_candidates": max_k,
                "physics_rng_seed": args.physics_rng_seed,
                "physics_use_l8_batch_deploy": use_l8_batch,
                "physics_l8_batch_size": l8_bs,
                "vertex_pool_count": int(sum(len(s) for s in segments)),
                "candidates_prepared": n_candidates_generated,
                "physics_filter_mode": "all",
            },
            "drop_points": [[float(x), float(y)] for x, y in stable_all],
        }
        if physics_meta is not None:
            stable_payload["physics_filter_run"] = physics_meta
        stable_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stable_path, "w", encoding="utf-8") as f:
            json.dump(stable_payload, f, indent=2, ensure_ascii=False)
        logger.info("全量稳定点: %s (%d 个)", stable_path, stable_all_count)

        unity_cap = min(max(1, args.l8_max_drops), UNITY_L8_MAX_DROP_POINTS)
        l8_export_count = int(min(n_drops, unity_cap, stable_all_count))
        drop_points = stable_all[:l8_export_count]
        if stable_all_count == 0:
            logger.warning("physics_filter_all: 无稳定点，drop_points.json 将为空")
        elif l8_export_count < stable_all_count:
            logger.info(
                "L8 子集: 写入 drop_points.json 前 %d 个稳定点 "
                "(n_drops=%d, l8_max_drops=%d, Unity≤%d)，其余见 %s",
                l8_export_count, n_drops, args.l8_max_drops,
                UNITY_L8_MAX_DROP_POINTS, stable_all_filename,
            )

    elif args.physics_filter:
        from drop_point_physics import filter_drop_points_stable

        physics_filter_mode = "quick"
        oversample = max(1, args.physics_filter_oversample)
        n_candidates_generated = max(n_drops * oversample, n_drops)
        candidates = compute_drop_points(
            segments, n_candidates_generated, drop_height, height_decay,
        )
        logger.info(
            "物理过滤(quick): 候选 %d 个 (n_drops=%d × oversample=%d)，阈值=%.2f settle=%d warmup=%d",
            len(candidates), n_drops, oversample,
            settle_thr, phys_settle, phys_warmup,
        )
        max_tries = args.physics_filter_max_candidates
        drop_points, physics_meta = filter_drop_points_stable(
            candidates,
            target_count=n_drops,
            max_tries=max_tries,
            settle_drop_threshold=settle_thr,
            settle_steps=phys_settle,
            warmup_steps=phys_warmup,
            port=port,
            game_root=game_root,
            timeout=timeout,
            no_launch=args.physics_no_launch,
            use_l8_batch_deploy=use_l8_batch,
            l8_batch_size=l8_bs,
        )
        if len(drop_points) < n_drops:
            logger.warning(
                "物理过滤后仅得到 %d/%d 个稳定点 — 已写入稳定子集；可增大 oversample、"
                "settle_drop_threshold 或 --physics-filter-max-candidates",
                len(drop_points), n_drops,
            )
    else:
        drop_points = compute_drop_points(segments, n_drops, drop_height, height_decay)

    logger.info("投放点 (%d 个):", len(drop_points))
    for i, (dx, dy) in enumerate(drop_points):
        logger.info("  #%02d  (%.2f, %.2f)", i, dx, dy)

    # 保存投放点结果到 cache
    drop_cache_path = colliders_dir / "drop_points.json"
    phys_on = bool(args.physics_filter or args.physics_filter_all)
    drop_cache = {
        "params": {
            "n_drops": n_drops,
            "drop_height": drop_height,
            "padding": padding,
            "height_decay": height_decay,
            "y_max_cutoff": y_max_cutoff,
            "min_clearance": min_clearance,
            "physics_filtered": phys_on,
            "physics_filter_mode": physics_filter_mode,
            "settle_drop_threshold": settle_thr,
            "physics_warmup_steps": phys_warmup if phys_on else None,
            "physics_settle_steps": phys_settle if phys_on else None,
            "physics_filter_oversample": (
                max(1, args.physics_filter_oversample)
                if args.physics_filter and not args.physics_filter_all
                else None
            ),
            "n_candidates_generated": n_candidates_generated if phys_on else None,
            "physics_filter_max_candidates": (
                args.physics_filter_max_candidates
                if args.physics_filter and not args.physics_filter_all
                else None
            ),
            "stable_all_path": stable_all_filename if args.physics_filter_all else None,
            "stable_all_count": stable_all_count if args.physics_filter_all else None,
            "l8_export_count": l8_export_count if args.physics_filter_all else None,
            "l8_max_drops": args.l8_max_drops if args.physics_filter_all else None,
            "physics_max_candidates": (
                (args.physics_max_candidates if args.physics_max_candidates is not None else 800)
                if args.physics_filter_all
                else None
            ),
            "physics_rng_seed": args.physics_rng_seed if args.physics_filter_all else None,
            "physics_use_l8_batch_deploy": use_l8_batch if phys_on else None,
            "physics_l8_batch_size": l8_bs if phys_on else None,
        },
        "n_segments": len(segments),
        "n_surface_pts": total_pts,
        "drop_points": [[float(x), float(y)] for x, y in drop_points],
    }
    if physics_meta is not None:
        drop_cache["physics_filter_run"] = physics_meta
    drop_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(drop_cache_path, "w", encoding="utf-8") as f:
        json.dump(drop_cache, f, indent=2, ensure_ascii=False)
    logger.info("投放点已缓存: %s", drop_cache_path)

    if not args.no_plot:
        plot_map(env_data, player_data, segments,
                 drop_points, drop_height, min_clearance, save_path)

    logger.info("=" * 60)
    logger.info("L7 Phase 1 完成 — 请检查图中表面和投放点是否正确")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

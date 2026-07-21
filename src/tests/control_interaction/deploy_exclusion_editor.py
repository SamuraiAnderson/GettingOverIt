"""
人工涂抹排除区编辑器（deploy exclusion zones）。

在 L7 地图（环境多边形 + 可着陆表面 + 已筛稳定落点）上用「圆形笔刷」涂抹，
标记不希望投放的区域。落点若落入任一圆内，离线 L7 与运行时 rollout 都会将其剔除。

结果写入本项目 checkpoints/deploy_exclusion_zones.json（随仓库版本化）：
    {"zones": [[cx, cy, r], ...], "brush_radius": <半径>, "drop_height": <高度>}

用法：
  # 优先用 cache（无需开游戏）；地图/表面/落点均来自已导出的 json
  python src/tests/control_interaction/deploy_exclusion_editor.py --no-launch
  python src/tests/control_interaction/deploy_exclusion_editor.py --brush-radius 8

交互：
  左键拖动   涂抹（沿轨迹盖圆章，当前笔刷半径）
  右键拖动   擦除（删除光标笔刷内的圆）
  中键拖动   平移视图（pan）
  滚轮        以光标为中心放大 / 缩小
  [ / ]      减小 / 增大笔刷半径
  e          切换 涂抹/擦除 模式（供无右键时使用，左键即按当前模式）
  f          复位视图到全图
  u          撤销上一笔（整段拖动为一个撤销单元）
  c          清空所有排除区
  s          保存到 checkpoints/deploy_exclusion_zones.json
  r          从磁盘重载（丢弃未保存改动）
  q / esc    退出
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
# 便于以任意工作目录 import 同目录的 test_l7_surface_airdrop
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.deploy_sampling import (  # noqa: E402
    default_exclusion_zones_path,
    load_exclusion_zones,
)

# 复用 L7 的加载/几何函数
from test_l7_surface_airdrop import (  # noqa: E402
    _get_game_root,
    _load_config,
    compute_landable_surfaces,
    compute_player_height,
    extract_polygons,
    load_colliders,
    _EXCLUDED_COLLIDERS,
)

logger = logging.getLogger(__name__)


def _load_stable_drop_points(colliders_dir: Path, drop_height: float) -> np.ndarray:
    """读取已筛稳定落点用于参考显示（优先 all_stable，回退 drop_points）。

    仅用于可视化；若都缺失则返回空数组（编辑器仍可基于表面涂抹）。
    """
    for name in ("drop_points_all_stable.json", "drop_points.json"):
        p = colliders_dir / name
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        pts = data.get("drop_points", [])
        if pts:
            logger.info("参考落点: 从 %s 载入 %d 个", name, len(pts))
            return np.asarray(pts, dtype=np.float64)
    logger.info("未找到 drop_points，落点参考图层为空（可先跑 L7 生成）")
    return np.zeros((0, 2), dtype=np.float64)


class ExclusionEditor:
    """matplotlib 圆形笔刷排除区编辑器。"""

    def __init__(
        self,
        env_data: dict,
        segments: list[np.ndarray],
        drop_points: np.ndarray,
        save_path: Path,
        brush_radius: float,
        drop_height: float,
    ) -> None:
        self.env_data = env_data
        self.segments = segments
        self.drop_points = drop_points
        self.save_path = save_path
        self.brush_radius = float(brush_radius)
        self.drop_height = float(drop_height)

        self.mode = "paint"          # "paint" | "erase"
        self._dragging = False
        self._active_erase = False
        self._last_stamp: tuple[float, float] | None = None
        self._dirty = False
        # 视图平移（中键拖动）状态：(按下像素x, 按下像素y, 按下时xlim, 按下时ylim)
        self._pan_start = None
        self._home_xlim: tuple[float, float] | None = None
        self._home_ylim: tuple[float, float] | None = None

        # 排除区圆列表（每项 [cx, cy, r]）+ 撤销快照栈
        self.zones: list[list[float]] = []
        self.undo_stack: list[list[list[float]]] = []

        existing = load_exclusion_zones(save_path)
        if existing is not None:
            self.zones = [[float(a), float(b), float(c)] for a, b, c in existing]
            logger.info("已载入现有排除区 %d 个: %s", len(self.zones), save_path)

        self._fig = None
        self._ax = None
        self._zone_artists: list = []
        self._kept_scatter = None
        self._excl_scatter = None
        self._last_excl = 0  # 缓存被排除落点数，避免涂抹时每帧重算

    # ── 构建底图 ────────────────────────────────────────────────
    def _draw_base(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection

        self._fig, self._ax = plt.subplots(1, 1, figsize=(20, 12))
        ax = self._ax

        env_patches = []
        for collider in self.env_data.get("colliders", []):
            if collider.get("name") in _EXCLUDED_COLLIDERS:
                continue
            for path in collider.get("paths", []):
                pts = np.array(path, dtype=np.float64)
                if len(pts) >= 3:
                    env_patches.append(plt.Polygon(pts, closed=True))
        if env_patches:
            pc = PatchCollection(env_patches, facecolor="#d0d0d0",
                                 edgecolor="#555555", linewidth=0.3, alpha=0.8)
            ax.add_collection(pc)

        cmap = plt.get_cmap("tab10")
        for i, seg in enumerate(self.segments):
            ax.plot(seg[:, 0], seg[:, 1], color=cmap(i % 10),
                    linewidth=1.2, zorder=3)

        ax.set_aspect("equal")
        ax.set_xlabel("X (world)")
        ax.set_ylabel("Y (world)")
        ax.grid(True, alpha=0.3)

        # 依据环境多边形 + 表面显式设定全图范围（供滚轮缩放/中键平移/复位使用）
        xs, ys = [], []
        for collider in self.env_data.get("colliders", []):
            if collider.get("name") in _EXCLUDED_COLLIDERS:
                continue
            for path in collider.get("paths", []):
                p = np.asarray(path, dtype=np.float64)
                if len(p) >= 2:
                    xs.append(p[:, 0]); ys.append(p[:, 1])
        for seg in self.segments:
            if len(seg) >= 1:
                xs.append(seg[:, 0]); ys.append(seg[:, 1])
        if xs:
            allx = np.concatenate(xs); ally = np.concatenate(ys)
            mx = (allx.max() - allx.min()) * 0.03 + 1.0
            my = (ally.max() - ally.min()) * 0.03 + 1.0
            self._home_xlim = (float(allx.min() - mx), float(allx.max() + mx))
            self._home_ylim = (float(ally.min() - my), float(ally.max() + my))
            ax.set_xlim(*self._home_xlim)
            ax.set_ylim(*self._home_ylim)
        else:
            ax.autoscale_view()
            self._home_xlim = ax.get_xlim()
            self._home_ylim = ax.get_ylim()

    # ── 增量图层（圆 + 落点着色） ───────────────────────────────
    def _excluded_mask(self) -> np.ndarray:
        if len(self.drop_points) == 0 or not self.zones:
            return np.zeros(len(self.drop_points), dtype=bool)
        z = np.asarray(self.zones, dtype=np.float64)
        dx = self.drop_points[:, 0][:, None] - z[None, :, 0]
        dy = self.drop_points[:, 1][:, None] - z[None, :, 1]
        return np.any(dx * dx + dy * dy <= (z[None, :, 2] ** 2), axis=1)

    def _make_circle(self, cx: float, cy: float, r: float):
        import matplotlib.pyplot as plt
        return plt.Circle((cx, cy), r, facecolor="#ff3b30", edgecolor="#a00",
                          alpha=0.25, zorder=6, linewidth=0.8)

    def _add_zone_patch(self, cx: float, cy: float, r: float) -> None:
        """仅新增一个圆的 patch（涂抹时的增量绘制，避免全量重建）。"""
        circ = self._make_circle(cx, cy, r)
        self._ax.add_patch(circ)
        self._zone_artists.append(circ)

    def _rebuild_zone_patches(self) -> None:
        """全量重建圆 patch（撤销/清空/擦除/重载时用）。"""
        for art in self._zone_artists:
            art.remove()
        self._zone_artists = []
        for cx, cy, r in self.zones:
            self._add_zone_patch(cx, cy, r)

    def _recompute_points(self) -> None:
        """重算落点「保留/排除」着色（较贵，仅在松手/结构性改动后调用）。"""
        ax = self._ax
        if self._kept_scatter is not None:
            self._kept_scatter.remove()
            self._kept_scatter = None
        if self._excl_scatter is not None:
            self._excl_scatter.remove()
            self._excl_scatter = None
        if len(self.drop_points) == 0:
            self._last_excl = 0
            return
        excl = self._excluded_mask()
        self._last_excl = int(excl.sum())
        kept = self.drop_points[~excl]
        gone = self.drop_points[excl]
        if len(kept) > 0:
            self._kept_scatter = ax.scatter(
                kept[:, 0], kept[:, 1], s=18, color="#1e6fff", zorder=7)
        if len(gone) > 0:
            self._excl_scatter = ax.scatter(
                gone[:, 0], gone[:, 1], s=28, color="#ff3b30", marker="x", zorder=8)

    def _redraw_all(self) -> None:
        """结构性改动后的完整重绘（圆 + 落点 + 标题）。"""
        self._rebuild_zone_patches()
        self._recompute_points()
        self._update_title()
        self._fig.canvas.draw_idle()

    def _update_title(self) -> None:
        n_excl = self._last_excl
        star = "*" if self._dirty else ""
        self._ax.set_title(
            f"Exclusion Editor{star} | mode={self.mode} | brush r={self.brush_radius:.2f} "
            f"| zones={len(self.zones)} | excluded {n_excl}/{len(self.drop_points)} pts\n"
            f"L paint  R erase  M/drag pan  wheel zoom  [/] radius  e mode  "
            f"f home  u undo  c clear  s save  r reload  q quit"
        )

    # ── 编辑操作 ────────────────────────────────────────────────
    def _push_undo(self) -> None:
        self.undo_stack.append([z[:] for z in self.zones])

    def _stamp(self, x: float, y: float) -> None:
        self.zones.append([float(x), float(y), self.brush_radius])
        self._dirty = True

    def _erase_at(self, x: float, y: float) -> None:
        r = self.brush_radius
        kept = [z for z in self.zones
                if (z[0] - x) ** 2 + (z[1] - y) ** 2 > r * r]
        if len(kept) != len(self.zones):
            self.zones = kept
            self._dirty = True

    # ── 事件回调 ────────────────────────────────────────────────
    def _on_press(self, event) -> None:
        if event.inaxes != self._ax:
            return
        # 中键 = 平移视图（pan）
        if event.button == 2:
            self._pan_start = (event.x, event.y,
                               self._ax.get_xlim(), self._ax.get_ylim())
            return
        if event.xdata is None:
            return
        # 右键 = 擦除；左键 = 当前模式
        erasing = (event.button == 3) or (self.mode == "erase")
        self._push_undo()
        self._dragging = True
        self._active_erase = erasing
        self._last_stamp = None
        self._apply_at(event.xdata, event.ydata)

    def _on_motion(self, event) -> None:
        if self._pan_start is not None:
            self._do_pan(event)
            return
        if not self._dragging or event.inaxes != self._ax or event.xdata is None:
            return
        self._apply_at(event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        if event.button == 2 and self._pan_start is not None:
            self._pan_start = None
            return
        if not self._dragging:
            return
        self._dragging = False
        # 若这一笔没有产生任何改动，回退掉刚压入的快照
        if self.undo_stack and self.undo_stack[-1] == self.zones:
            self.undo_stack.pop()
        # 松手后再重算落点着色（涂抹过程中已省略此步以保持流畅）
        self._recompute_points()
        self._update_title()
        self._fig.canvas.draw_idle()

    def _do_pan(self, event) -> None:
        if event.x is None or event.y is None:
            return
        xpix0, ypix0, (x0, x1), (y0, y1) = self._pan_start
        bbox = self._ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        # 视图沿鼠标反方向移动，保持「抓住地图拖动」的直觉
        ddx = -(event.x - xpix0) * (x1 - x0) / bbox.width
        ddy = -(event.y - ypix0) * (y1 - y0) / bbox.height
        self._ax.set_xlim(x0 + ddx, x1 + ddx)
        self._ax.set_ylim(y0 + ddy, y1 + ddy)
        self._fig.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes != self._ax or event.xdata is None:
            return
        ax = self._ax
        scale = 1.0 / 1.2 if event.step > 0 else 1.2  # 上滚放大
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        # 以光标为锚点缩放，保持光标下的世界坐标不动
        ax.set_xlim(xd - (xd - x0) * scale, xd + (x1 - xd) * scale)
        ax.set_ylim(yd - (yd - y0) * scale, yd + (y1 - yd) * scale)
        self._fig.canvas.draw_idle()

    def _apply_at(self, x: float, y: float) -> None:
        if self._active_erase:
            before = len(self.zones)
            self._erase_at(x, y)
            if len(self.zones) != before:
                self._rebuild_zone_patches()
                self._update_title()
                self._fig.canvas.draw_idle()
            return
        spacing = max(self.brush_radius * 0.4, 1e-3)
        if self._last_stamp is not None:
            lx, ly = self._last_stamp
            if (x - lx) ** 2 + (y - ly) ** 2 < spacing * spacing:
                return
        self._stamp(x, y)
        self._last_stamp = (x, y)
        # 涂抹时只增量加一个圆 + 轻量刷新标题，落点着色延迟到松手（_on_release）
        self._add_zone_patch(x, y, self.brush_radius)
        self._update_title()
        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        k = event.key
        if k == "]":
            self.brush_radius *= 1.25
            self._update_title(); self._fig.canvas.draw_idle()
        elif k == "[":
            self.brush_radius = max(self.brush_radius / 1.25, 0.1)
            self._update_title(); self._fig.canvas.draw_idle()
        elif k == "e":
            self.mode = "erase" if self.mode == "paint" else "paint"
            self._update_title(); self._fig.canvas.draw_idle()
        elif k == "f":
            if self._home_xlim is not None:
                self._ax.set_xlim(*self._home_xlim)
                self._ax.set_ylim(*self._home_ylim)
                self._fig.canvas.draw_idle()
        elif k == "u":
            if self.undo_stack:
                self.zones = self.undo_stack.pop()
                self._dirty = True
                self._redraw_all()
        elif k == "c":
            if self.zones:
                self._push_undo()
                self.zones = []
                self._dirty = True
                self._redraw_all()
        elif k == "s":
            self.save()
            self._update_title()
            self._fig.canvas.draw_idle()
        elif k == "r":
            existing = load_exclusion_zones(self.save_path)
            self.zones = ([[float(a), float(b), float(c)] for a, b, c in existing]
                          if existing is not None else [])
            self.undo_stack.clear()
            self._dirty = False
            logger.info("已从磁盘重载排除区 (%d)", len(self.zones))
            self._redraw_all()
        elif k in ("q", "escape"):
            import matplotlib.pyplot as plt
            plt.close(self._fig)

    def save(self) -> None:
        payload = {
            "zones": [[round(c, 4) for c in z] for z in self.zones],
            "brush_radius": round(self.brush_radius, 4),
            "drop_height": self.drop_height,
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._dirty = False
        logger.info("已保存 %d 个排除区: %s", len(self.zones), self.save_path)

    def run(self) -> None:
        import matplotlib.pyplot as plt

        self._draw_base()
        self._redraw_all()
        c = self._fig.canvas
        # 断开 matplotlib 默认键位（s=保存图片、f=全屏、c/v=前后、r/h=复位、g=网格…），
        # 否则会与本编辑器的 s/f/c/r 快捷键冲突、互相抢事件。
        try:
            handler_id = getattr(c.manager, "key_press_handler_id", None)
            if handler_id is not None:
                c.mpl_disconnect(handler_id)
        except Exception:  # noqa: BLE001 — 不同后端 manager 结构不一，失败则忽略
            pass
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("key_press_event", self._on_key)
        logger.info("编辑器已打开：左键涂抹 / 右键擦除 / 中键拖动平移 / 滚轮缩放 / s 保存 / q 退出")
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="人工涂抹排除区编辑器")
    parser.add_argument("--game-root", type=str, default=None, help="游戏根目录")
    parser.add_argument("--no-launch", action="store_true",
                        help="不启动游戏，仅用 cache 的 environment/player 轮廓")
    parser.add_argument("--drop-height", type=float, default=None,
                        help="表面上方投放高度（用于表面/落点显示，默认取 test_config.l7）")
    parser.add_argument("--padding", type=float, default=None,
                        help="Player 包围盒高度之上的净空 padding")
    parser.add_argument("--y-max-cutoff", type=float, default=None,
                        help="屏蔽 y 超过此值的高空表面")
    parser.add_argument("--brush-radius", type=float, default=5.0,
                        help="初始笔刷半径（世界单位，默认 5.0）")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
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
    drop_height = args.drop_height if args.drop_height is not None else l7.get("drop_height", 2.0)
    padding = args.padding if args.padding is not None else l7.get("padding", 0.3)
    y_max_cutoff = args.y_max_cutoff if args.y_max_cutoff is not None else l7.get("y_max_cutoff", 380.0)

    game_root = _get_game_root(args.game_root)
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
    polygons = extract_polygons(env_data)
    segments = compute_landable_surfaces(polygons, min_clearance, resolution=0.1,
                                         y_max_cutoff=y_max_cutoff)
    logger.info("可着陆表面: %d 条线段", len(segments))

    drop_points = _load_stable_drop_points(colliders_dir, drop_height)
    save_path = default_exclusion_zones_path()

    editor = ExclusionEditor(
        env_data=env_data,
        segments=segments,
        drop_points=drop_points,
        save_path=save_path,
        brush_radius=args.brush_radius,
        drop_height=drop_height,
    )
    editor.run()


if __name__ == "__main__":
    main()

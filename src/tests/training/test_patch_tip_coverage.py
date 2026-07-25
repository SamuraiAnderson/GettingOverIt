r"""回归测试：contact patch 窗口必须装得下任意伸展姿态的锤头。

背景见 `config.contact_patch_resolution` 注释与
`src/tests/analysis/check_tip_patch_coverage.py`：锤子几何硬上界（tip 原点 3.31m，
设计上界取 PlayerControl IL 里的 cursor 钳制常数 3.5m）+ tip 轮廓半径 0.279m
→ 需半窗 ≥ 3.78m。旧配置 0.2 m/px（半径 3.2m）会在锤子全伸时裁掉锤头接触的地形。

本测不依赖游戏进程与真实轨迹：直接构造「锤子沿 360° 各方向伸到硬上界」的合成
33D 状态，走真实 `build_patch`，断言 ch3（锤头轮廓）既非空、也未被窗口边界截断。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.dataset import build_patch

# 锤子几何链（见 check_tip_patch_coverage.py docstring）
HUB_OFFSET = 0.3546
TIP_ORIGIN_HARD_BOUND = 3.310
CURSOR_CLAMP_RADIUS = 3.5
TIP_CONTOUR_RADIUS = 0.2793

# 合成 tip 局部轮廓：按真实 player_contour.json 的尺寸造一个等效八边形
# （真实 tip: x∈[-0.0709,0.0768], y∈[-0.2791,0.1687], R_max=0.2793）
_TIP_LOCAL = np.array([
    [-0.0709, -0.2791], [0.0768, -0.2791], [0.0768, 0.1687], [-0.0709, 0.1687],
], dtype=np.float64)
_BODY_LOCAL = np.array(
    [[-0.2919, 0.0828], [0.3460, 0.0828], [0.3460, 1.0401], [-0.2919, 1.0401]],
    dtype=np.float64,
)
_POT_LOCAL = np.array(
    [[-0.3239, -0.9767], [0.3824, -0.9767], [0.3824, -0.7000], [-0.3239, -0.7000]],
    dtype=np.float64,
)


def _make_state(px: float, py: float, tip_x: float, tip_y: float) -> np.ndarray:
    """构造最小可用的 33D 状态：只填 build_patch 会读的字段。"""
    s = np.zeros(33, dtype=np.float32)
    s[0], s[1] = px, py
    # hub 在 player 上方 HUB_OFFSET（决定 body 轮廓朝向 player_hub_angle_deg）
    s[5], s[6] = px, py + HUB_OFFSET
    s[23], s[24] = tip_x, tip_y
    # pole 在 tip 与 player 之间（决定 tip 轮廓朝向 rod_angle_deg 的 pole→tip）
    s[19] = px + (tip_x - px) * 0.7
    s[20] = py + (tip_y - py) * 0.7
    return s


def _tip_channel_bbox_px(patch: np.ndarray) -> tuple[int, int, int, int] | None:
    """ch3 里非零像素的 bbox (row_min, row_max, col_min, col_max)；全空返回 None。"""
    ys, xs = np.nonzero(patch[3] > 0.0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def test_window_radius_covers_geometric_bound() -> None:
    """配置层面：窗口半径必须 >= 设计上界 + tip 轮廓半径 + 地形上下文余量。

    只装下轮廓本身还不够——锤头全伸时若正好落在最外圈像素，它接触的那块地形就在
    窗口外，而这恰是「伸到最远够远处落点」的高价值帧。故要求额外留 2 格余量。
    """
    cfg = TrainConfig()
    half_window = cfg.patch_size / 2.0 * cfg.contact_patch_resolution
    need_contour = CURSOR_CLAMP_RADIUS + TIP_CONTOUR_RADIUS
    need_context = need_contour + 2.0 * cfg.contact_patch_resolution

    assert half_window >= need_contour, (
        f"contact 窗口半径 {half_window:.2f}m < 轮廓所需 {need_contour:.3f}m "
        f"(patch_size={cfg.patch_size}, res={cfg.contact_patch_resolution})；"
        f"锤子全伸时锤头会被裁掉，见 check_tip_patch_coverage.py"
    )
    assert half_window >= need_context, (
        f"contact 窗口半径 {half_window:.2f}m 装得下轮廓但无地形上下文余量"
        f"（需 {need_context:.3f}m = 轮廓 {need_contour:.3f} + 2 格）；"
        f"锤头全伸时会落在最外圈像素，其接触的地形在窗口外"
    )


def test_tip_channel_never_clipped_at_hard_bound() -> None:
    """锤子沿 360° 各方向伸到硬上界，ch3 都必须非空且不触窗口边界。

    「不触边界」判据：ch3 非零像素的 bbox 不能贴到第 0 行/列或最后一行/列。
    一旦贴边就说明轮廓（或其紧邻地形）被窗口截断。
    """
    cfg = TrainConfig()
    size = cfg.patch_size
    px, py = 100.0, 50.0  # 任意远离原点的位置，验证平移无关性

    empty_dirs: list[float] = []
    clipped_dirs: list[float] = []
    for deg in range(0, 360, 5):
        rad = np.deg2rad(deg)
        tip_x = px + CURSOR_CLAMP_RADIUS * np.cos(rad)
        tip_y = py + CURSOR_CLAMP_RADIUS * np.sin(rad)
        state = _make_state(px, py, tip_x, tip_y)

        patch = build_patch(
            state, None, None, cfg, 0, 0,
            solid_polygons=None,
            tip_local=_TIP_LOCAL,
            body_local=[_BODY_LOCAL, _POT_LOCAL],
        )
        bbox = _tip_channel_bbox_px(patch)
        if bbox is None:
            empty_dirs.append(float(deg))
            continue
        r0, r1, c0, c1 = bbox
        if r0 == 0 or c0 == 0 or r1 == size - 1 or c1 == size - 1:
            clipped_dirs.append(float(deg))

    assert not empty_dirs, (
        f"以下方向上 ch3 完全为空（锤头整体出窗）: {empty_dirs}"
    )
    assert not clipped_dirs, (
        f"以下方向上 ch3 贴到窗口边界（锤头被截断）: {clipped_dirs}"
    )


def test_body_and_tip_coexist_in_window() -> None:
    """锤子全伸时，body(ch2) 与 tip(ch3) 必须同时可见——两者都是接触推理的依据。"""
    cfg = TrainConfig()
    px, py = 0.0, 0.0
    for deg in (0, 45, 90, 135, 180, 225, 270, 315):
        rad = np.deg2rad(deg)
        state = _make_state(
            px, py,
            px + CURSOR_CLAMP_RADIUS * np.cos(rad),
            py + CURSOR_CLAMP_RADIUS * np.sin(rad),
        )
        patch = build_patch(
            state, None, None, cfg, 0, 0,
            solid_polygons=None,
            tip_local=_TIP_LOCAL,
            body_local=[_BODY_LOCAL, _POT_LOCAL],
        )
        assert patch[2].sum() > 0.0, f"{deg}°: body/pot 通道为空"
        assert patch[3].sum() > 0.0, f"{deg}°: 锤头通道为空"


def test_old_resolution_would_have_failed() -> None:
    """反向确认：旧的 0.2 m/px 配置在硬上界处确实会裁掉锤头（证明本修复非空操作）。"""
    cfg = TrainConfig()
    cfg.contact_patch_resolution = 0.2  # 回到旧值
    px, py = 0.0, 0.0
    state = _make_state(px, py, px + CURSOR_CLAMP_RADIUS, py)  # 水平全伸

    patch = build_patch(
        state, None, None, cfg, 0, 0,
        solid_polygons=None,
        tip_local=_TIP_LOCAL,
        body_local=[_BODY_LOCAL, _POT_LOCAL],
    )
    old_half = cfg.patch_size / 2.0 * 0.2
    assert old_half < CURSOR_CLAMP_RADIUS + TIP_CONTOUR_RADIUS
    bbox = _tip_channel_bbox_px(patch)
    assert bbox is None or bbox[3] == cfg.patch_size - 1, (
        "旧 0.2m/px 配置下锤头本应出窗或贴边，若此断言失败说明几何常量或"
        "build_patch 语义已变，请重新核对 check_tip_patch_coverage.py"
    )


def main() -> None:
    test_window_radius_covers_geometric_bound()
    print("窗口半径 >= 几何上界 OK")
    test_tip_channel_never_clipped_at_hard_bound()
    print("360° 全伸姿态 ch3 无空/无截断 OK")
    test_body_and_tip_coexist_in_window()
    print("body 与 tip 同窗可见 OK")
    test_old_resolution_would_have_failed()
    print("旧 0.2m/px 确实会裁（反向确认）OK")
    print("ALL PATCH COVERAGE OK")


if __name__ == "__main__":
    main()

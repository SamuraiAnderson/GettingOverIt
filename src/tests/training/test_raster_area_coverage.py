r"""验证 `rasterize_polygon_fill` 的面积覆盖率语义正确（不只是"非空"）。

`rasterize_polygon_fill` 从二值「格心在多边形内」改为面积覆盖率（子采样估计），
是 patch ch2/ch3 的核心图元。本测确认：
  1. 覆盖率求和 × 格面积 ≈ 多边形真实面积（子采样精度内）
  2. 大多边形内部格 = 1.0、外部格 = 0.0，仅边界格取中间值
  3. 亚像素小多边形不再产生全零通道（旧二值判据的致命缺陷）
  4. `subsamples=1` 严格退化为旧二值行为（向后可比）
  5. 完全在窗口外的多边形返回全零，不越界不报错
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.dataset import PART_RASTER_SUBSAMPLES, rasterize_polygon_fill


def _polygon_area(p: np.ndarray) -> float:
    """鞋带公式求多边形面积（绝对值）。"""
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def test_coverage_sum_matches_area() -> None:
    """覆盖率总和 × 格面积 ≈ 多边形面积。"""
    res = 0.27
    size = 32
    for poly in (
        # 正方形，跨多格
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
        # 旋转 30° 的长条（考验边界格）
        np.array([[-1.5, -0.2], [1.5, -0.2], [1.5, 0.2], [-1.5, 0.2]])
        @ np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)],
                    [-np.sin(np.pi / 6), np.cos(np.pi / 6)]]),
        # 三角形
        np.array([[-1.0, -0.8], [1.2, -0.8], [0.1, 1.1]]),
    ):
        ch = rasterize_polygon_fill(poly, 0.0, 0.0, size, res)
        est = float(ch.sum()) * res * res
        true = _polygon_area(poly)
        rel = abs(est - true) / true
        assert rel < 0.06, (
            f"面积估计误差 {rel * 100:.2f}% 过大: est={est:.4f} true={true:.4f}"
        )


def test_interior_is_one_boundary_is_partial() -> None:
    """大多边形：内部格严格=1，外部格=0，存在取中间值的边界格。"""
    res = 0.27
    size = 32
    # 半径远大于格的正方形，中心对齐 patch 中心
    poly = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    ch = rasterize_polygon_fill(poly, 0.0, 0.0, size, res)

    assert ch.max() == 1.0, f"内部格应有 1.0, got max={ch.max()}"
    assert ch.min() == 0.0, f"窗口角落应为 0.0, got min={ch.min()}"
    # 中心格必然完全在内
    c = size // 2
    assert ch[c, c] == 1.0, f"中心格应=1.0, got {ch[c, c]}"
    # 必须存在部分覆盖的边界格（否则说明退化成了二值）
    partial = ch[(ch > 0.0) & (ch < 1.0)]
    assert len(partial) > 0, "应存在部分覆盖的边界格（面积覆盖率的标志）"


def test_subpixel_polygon_never_empty() -> None:
    """亚像素尺寸多边形在任意亚像素位置都不产生全零（旧二值判据的致命缺陷）。"""
    res = 0.27
    size = 32
    # 真实 tip 轮廓量级：0.148 × 0.448 m，远小于一格宽
    w, h = 0.1477, 0.4478
    base = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

    empties = 0
    trials = 0
    for ox in np.linspace(0.0, res, 7, endpoint=False):
        for oy in np.linspace(0.0, res, 7, endpoint=False):
            for deg in (0, 23, 45, 90, 137, 180, 270):
                th = np.deg2rad(deg)
                rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                poly = base @ rot.T + np.array([ox, oy])
                ch = rasterize_polygon_fill(poly, 0.0, 0.0, size, res)
                trials += 1
                if ch.sum() == 0.0:
                    empties += 1
    assert empties == 0, f"{empties}/{trials} 个亚像素姿态产生了全零通道"


def test_subsamples_one_reduces_to_binary() -> None:
    """subsamples=1 时严格退化为旧的「格心在多边形内」二值判据。"""
    res = 0.27
    size = 32
    poly = np.array([[-1.0, -0.6], [0.9, -0.6], [0.9, 0.7], [-1.0, 0.7]])
    ch1 = rasterize_polygon_fill(poly, 0.0, 0.0, size, res, subsamples=1)
    uniq = np.unique(ch1)
    assert set(uniq.tolist()) <= {0.0, 1.0}, f"subsamples=1 应只有 0/1, got {uniq}"

    # 与默认多子采样版本比较：二值版是覆盖率的粗量化，面积应接近但更差
    ch_aa = rasterize_polygon_fill(poly, 0.0, 0.0, size, res)
    true = _polygon_area(poly)
    err_bin = abs(float(ch1.sum()) * res * res - true)
    err_aa = abs(float(ch_aa.sum()) * res * res - true)
    assert err_aa <= err_bin + 1e-9, (
        f"面积覆盖率误差 {err_aa:.5f} 应不劣于二值 {err_bin:.5f}"
    )


def test_polygon_outside_window_is_empty() -> None:
    """完全在窗口外的多边形返回全零，且不抛异常/不越界写入。"""
    res = 0.27
    size = 32
    half = size / 2.0 * res
    far = np.array([
        [half + 5.0, half + 5.0], [half + 6.0, half + 5.0], [half + 6.0, half + 6.0],
    ])
    ch = rasterize_polygon_fill(far, 0.0, 0.0, size, res)
    assert ch.shape == (size, size)
    assert ch.sum() == 0.0, "窗口外多边形应全零"

    # 部分重叠：只有窗口内那部分被写入，且不越界
    edge = np.array([
        [half - 0.2, -0.5], [half + 3.0, -0.5], [half + 3.0, 0.5], [half - 0.2, 0.5],
    ])
    ch2 = rasterize_polygon_fill(edge, 0.0, 0.0, size, res)
    assert ch2.shape == (size, size)
    assert ch2.sum() > 0.0, "部分重叠应有非零覆盖"
    assert ch2[:, -1].sum() > 0.0, "越出的一侧应命中最外列"


def test_degenerate_input() -> None:
    """顶点数 < 3 的退化输入返回全零而非抛异常。"""
    for bad in (
        np.zeros((0, 2)),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    ):
        ch = rasterize_polygon_fill(bad, 0.0, 0.0, 32, 0.27)
        assert ch.shape == (32, 32) and ch.sum() == 0.0


def main() -> None:
    print(f"PART_RASTER_SUBSAMPLES = {PART_RASTER_SUBSAMPLES}")
    test_coverage_sum_matches_area()
    print("覆盖率总和 ≈ 多边形面积 OK")
    test_interior_is_one_boundary_is_partial()
    print("内部=1 / 边界部分覆盖 OK")
    test_subpixel_polygon_never_empty()
    print("亚像素多边形永不全零 OK")
    test_subsamples_one_reduces_to_binary()
    print("subsamples=1 退化为二值 OK")
    test_polygon_outside_window_is_empty()
    print("窗口外/跨界处理 OK")
    test_degenerate_input()
    print("退化输入兜底 OK")
    print("ALL RASTER COVERAGE OK")


if __name__ == "__main__":
    main()

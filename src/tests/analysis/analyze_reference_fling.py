"""只读分析:参考(人工)轨迹过树段是不是弹道甩飞。

判定依据(GOI 语义):
- 弹道甩飞 = 存在腾空段:vy 由强正→过零→强负(抛物线),该段加速度 ay≈常数(自由落体),
  且 |v| 出现尖峰;身体 y 呈抛物弧。
- 受控技法(勾枝/撬杆) = y 缓慢单调上升,|v| 全程低,ay 频繁变化(主动发力),无明显腾空弧。

输出:src/Data/analysis/reference_fling_analysis.png + 控制台文本结论。不修改任何数据。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "tests" / "control_interaction"))
PKL = _REPO / "src" / "Data" / "reference" / "reference_trajectory.pkl"
OUT = _REPO / "src" / "Data" / "analysis" / "reference_fling_analysis.png"

X, Y, VX, VY, TS = 0, 1, 2, 3, 28


def _load_polygons():
    """离线加载障碍多边形(世界坐标),失败返回 []。"""
    try:
        from training.config import TrainConfig
        from test_l7_surface_airdrop import extract_polygons, load_colliders
        cfg = TrainConfig()
        env_data, _player = load_colliders(Path(cfg.game_root))
        return extract_polygons(env_data)
    except Exception as e:  # noqa: BLE001
        print(f"[几何加载失败,跳过叠加] {e}")
        return []


def main() -> None:
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    traj = data["trajectories"][0]
    s = np.asarray(traj["raw_states"], dtype=np.float64)  # (T+1, 33)
    T = len(s)
    x, y, vx, vy = s[:, X], s[:, Y], s[:, VX], s[:, VY]
    ts = s[:, TS]
    speed = np.hypot(vx, vy)

    # dt: 优先用 timestamp 列;不可用则回退单位步
    dts = np.diff(ts)
    if np.all(np.isfinite(dts)) and np.median(dts) > 1e-6 and np.median(dts) < 5.0:
        dt = float(np.median(dts))
        dt_src = f"timestamp(中位 {dt:.4f}s)"
    else:
        dt = 1.0
        dt_src = "单位步(timestamp 不可用)"
    ay = np.gradient(vy, dt)
    ax = np.gradient(vx, dt)

    print("=" * 70)
    print(f"参考轨迹: {T} 步 (含起点)  dt 来源={dt_src}")
    print(f"起点 (x,y)=({x[0]:.2f},{y[0]:.2f})  终点 (x,y)=({x[-1]:.2f},{y[-1]:.2f})")
    print(f"y 范围 [{y.min():.2f}, {y.max():.2f}]  Δy_net={y[-1]-y[0]:.2f}  Δy_peak={y.max()-y[0]:.2f}")
    print(f"|v| 中位={np.median(speed):.3f}  |v| p95={np.percentile(speed,95):.3f}  |v| max={speed.max():.3f}")

    # 分段剖面(每 100 步):定位大攀爬段
    print("\n── 每100步剖面 (Δy=段内y净增, y_gain=段内峰-段起, |v|mid/max, |vy|max) ──")
    for a in range(0, T, 100):
        b = min(a + 100, T)
        yg = y[a:b]
        print(f"  步[{a:4d},{b:4d})  y:{yg[0]:6.2f}→{yg[-1]:6.2f}  峰{yg.max():6.2f}  "
              f"|v|mid={np.median(speed[a:b]):5.2f} max={speed[a:b].max():5.2f}  "
              f"|vy|max={np.abs(vy[a:b]).max():5.2f}")

    # ── 定位"过树段":锚定到全局最高点(越过树的那次大攀爬)──
    # secured 语义:峰=全局 argmax(y);段起点=峰前最后一次回到接近起点高度(树根发力点)。
    start_y = y[0]
    peak_idx = int(np.argmax(y))
    base_level = start_y + 0.5
    lo = peak_idx
    while lo > 0 and y[lo] > base_level:
        lo -= 1
    # 段终点:峰后 |v| 回落到中位以下(完成落地/越顶)
    hi = peak_idx
    while hi < T - 1 and speed[hi] > np.median(speed):
        hi += 1
    seg_lo, seg_hi = lo, min(hi + 3, T - 1)
    seg = slice(seg_lo, seg_hi + 1)
    print(f"\n过树段 ≈ 步[{seg_lo}, {seg_hi}]  (x:{x[seg_lo]:.1f}→{x[seg_hi]:.1f}, "
          f"y:{y[seg_lo]:.1f}→{y[seg_hi]:.1f})")

    # ── 弹道判据 ──
    seg_vy = vy[seg]
    seg_speed = speed[seg]
    seg_ay = ay[seg]
    # 1) vy 是否出现"强正→过零→强负"的发射-顶点-下落
    vy_pos_peak = seg_vy.max()
    vy_neg_peak = seg_vy.min()
    has_launch_arc = (vy_pos_peak > 0.5 * speed.std() + np.median(speed)) and (vy_neg_peak < 0)
    sign_flip = np.any((seg_vy[:-1] > 0) & (seg_vy[1:] < 0))
    # 2) 段内 |v| 尖峰相对全程
    speed_spike_ratio = seg_speed.max() / (np.median(speed) + 1e-6)
    # 3) 腾空自由落体:寻找连续窗口内 ay 近似常数(std 小)且 y 呈弧(先升后降)
    #    用 ay 的稳定度:自由飞行段 ay≈-g 常数 → 局部 std 低
    airborne_frac = float(np.mean(np.abs(seg_ay - np.median(seg_ay)) < 0.35 * (np.std(ay) + 1e-6)))
    # 4) y 在段内是否单调(受控) vs 有回落弧(弹道过冲)
    y_seg = y[seg]
    peak_i = int(np.argmax(y_seg))
    falls_after_peak = (len(y_seg) - peak_i) > 3 and (y_seg[peak_i] - y_seg[-1]) > 0.5

    print("\n── 弹道判据 ──")
    print(f"  vy 峰值: +{vy_pos_peak:.3f} / {vy_neg_peak:.3f}   发射-下落弧={has_launch_arc}  过零翻转={bool(sign_flip)}")
    print(f"  段内 |v| 尖峰 / 全程中位 = {speed_spike_ratio:.2f}×")
    print(f"  段内 ay 近常数占比(自由落体近似) = {airborne_frac:.2f}")
    print(f"  过峰后回落>0.5m(过冲) = {falls_after_peak}  (峰@段内步{peak_i}/{len(y_seg)-1})")

    # 综合评分:弹道特征数
    ballistic_signals = sum([
        bool(has_launch_arc and sign_flip),
        speed_spike_ratio > 2.5,
        airborne_frac > 0.5,
        falls_after_peak,
    ])
    print("\n── 结论 ──")
    if ballistic_signals >= 3:
        verdict = "强弹道甩飞:过树段是发射-腾空-落地的抛物线甩飞"
    elif ballistic_signals >= 2:
        verdict = "部分弹道:含腾空/甩动成分,但非纯自由飞行(可能借力+短腾空)"
    else:
        verdict = "受控技法:低速、平缓爬升,无明显腾空弧(勾枝/撬杆类)"
    print(f"  弹道信号 {ballistic_signals}/4 → {verdict}")

    # ── 作图 ──
    fig, axs = plt.subplots(3, 2, figsize=(14, 11))
    tt = np.arange(T)
    for a in axs[:, 0]:
        a.axvspan(seg_lo, seg_hi, color="orange", alpha=0.15, label="tree seg")
    axs[0, 0].plot(tt, y, "b"); axs[0, 0].set_title("y (height) vs step"); axs[0, 0].legend()
    axs[1, 0].plot(tt, vy, "g"); axs[1, 0].axhline(0, color="k", lw=0.5)
    axs[1, 0].set_title("vy (vertical vel) vs step")
    axs[2, 0].plot(tt, speed, "r"); axs[2, 0].axhline(np.median(speed), color="k", lw=0.5, ls="--")
    axs[2, 0].set_title("|v| (speed) vs step")
    axs[0, 1].plot(tt, vx, "m"); axs[0, 1].axhline(0, color="k", lw=0.5)
    axs[0, 1].axvspan(seg_lo, seg_hi, color="orange", alpha=0.15)
    axs[0, 1].set_title("vx (horiz vel) vs step")
    axs[1, 1].plot(tt, ay, "c"); axs[1, 1].axvspan(seg_lo, seg_hi, color="orange", alpha=0.15)
    axs[1, 1].set_title("ay (vert accel) vs step")
    polys = _load_polygons()
    for poly in polys:
        p = np.asarray(poly)
        if p.ndim == 2 and len(p) >= 3:
            axs[2, 1].add_patch(MplPolygon(p, closed=True, facecolor="0.6",
                                           edgecolor="0.3", alpha=0.5, zorder=0))
    sc = axs[2, 1].scatter(x, y, c=speed, cmap="viridis", s=6, zorder=2)
    axs[2, 1].scatter([x[0]], [y[0]], c="red", marker="*", s=160, zorder=3, label="start")
    # 视野裁到轨迹范围(+边距),避免整张地图缩太小
    axs[2, 1].set_xlim(x.min() - 5, x.max() + 5)
    axs[2, 1].set_ylim(y.min() - 5, y.max() + 5)
    axs[2, 1].set_title("x-y path over obstacles (color=|v|)"); axs[2, 1].legend()
    plt.colorbar(sc, ax=axs[2, 1])
    verdict_en = {"强": "BALLISTIC", "部": "PARTIAL", "受": "CONTROLLED"}.get(verdict[0], "")
    fig.suptitle(f"Reference trajectory fling analysis — signals {ballistic_signals}/4 ({verdict_en})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"\n图已保存: {OUT}")


if __name__ == "__main__":
    main()

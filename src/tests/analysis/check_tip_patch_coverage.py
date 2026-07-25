r"""核验 dualscale contact patch 窗口是否装得下锤头及其接触的地形。

## 为什么用几何硬上界，而不是统计外推

`|tip − player|` 有**几何硬上界**（锤子关节结构决定），其分布在上界处截断，
**不是高斯**。早期版本用 `dyn_mean/dyn_std` 做正态外推得到「约 2.9% 的帧 tip 出窗」，
是严重高估——正态尾部延伸到无穷，而真实分布在 3.31m 处硬截断（实测 15.2 万帧里
tip 中心从未出过 3.2m 窗口）。故本脚本改为纯几何判据，不依赖轨迹数据可得性。

## 锤子几何链（反编译 + 15.2 万真实帧交叉验证）

| 环节 | 值 | 依据 |
|------|-----|------|
| `\|hub − player\|` | 0.3546 m | 刚性偏移，σ < 1e-4 |
| slider 全伸 `\|tip − hub\|` | 2.955 m | 关节硬限位（直方图 514 帧堆积平台） |
| → tip 原点上界 | 3.31 m | hinge 可 360° 自由旋转，两段共线可达 |
| 设计上界 | 3.5 m | `PlayerControl` IL 硬编码 `\|cursor−player\| ≤ 3.5f`； |
|          |        | 与求解器软性超调实测极值 ≈3.49m 吻合 |
| tip 轮廓半径 | 0.279 m | `player_contour.json` tip 16 顶点距原点最远值 |

→ 所需半窗 = 3.5 + 0.279 ≈ **3.78 m**（装下锤头轮廓全部顶点）。
  接触判定还需看到紧邻地形，但接触发生在 `CONTACT_EPSILON = 0.03m` 内，
  故 3.78 + 0.03 ≈ 3.81 m 即绝对安全。

真实 `player_contour.json` 可得时会用文件里的实际顶点覆盖上表的 tip 半径常量。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.contact_features import CONTACT_EPSILON

_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"

# ── 锤子几何链常量（反编译 IL + 真实帧标定，见模块 docstring）──
HUB_OFFSET = 0.3546          # |hub − player|，刚性
SLIDER_MAX_EXTENSION = 2.955  # slider 全伸时 |tip − hub|，关节硬限位
CURSOR_CLAMP_RADIUS = 3.5     # PlayerControl IL: |cursor − player| <= 3.5f
TIP_CONTOUR_RADIUS_FALLBACK = 0.279  # player_contour.json 不可得时的兜底值


def _load_part_radii() -> dict[str, float] | None:
    """从真实 player_contour.json 读各部件轮廓距原点的最远顶点半径。"""
    try:
        with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
            game_root = Path(json.load(f)["game"]["executable_path"]).parent
    except (OSError, KeyError, json.JSONDecodeError):
        return None

    path = game_root / "GoiData" / "Colliders" / "player_contour.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    out: dict[str, float] = {}
    for name, part in data.get("parts", {}).items():
        paths = part.get("paths") or []
        best = 0.0
        for p in paths:
            arr = np.asarray(p, dtype=np.float64)
            if arr.ndim == 2 and len(arr) > 0:
                best = max(best, float(np.hypot(arr[:, 0], arr[:, 1]).max()))
        if best > 0.0:
            out[name] = best
    return out or None


def main() -> int:
    cfg = TrainConfig()
    half_window = cfg.patch_size / 2.0 * cfg.contact_patch_resolution
    window = cfg.patch_size * cfg.contact_patch_resolution

    print("=" * 68)
    print("contact patch 覆盖核验（几何硬上界判据）")
    print("=" * 68)
    print(f"配置: {cfg.patch_size}px × {cfg.contact_patch_resolution} m/px "
          f"= {window:.1f}m 窗口, 半径 {half_window:.2f}m")
    print(f"      一格 {cfg.contact_patch_resolution}m / CONTACT_EPSILON {CONTACT_EPSILON}m "
          f"= {cfg.contact_patch_resolution / CONTACT_EPSILON:.1f}× 粗 "
          f"(patch 只管空间布局，精确接触判定由 P2.1 向量信号承担)\n")

    radii = _load_part_radii()
    if radii:
        print("player_contour.json 实测各部件轮廓半径 (m):")
        for name in sorted(radii):
            print(f"  {name:12s} R_max = {radii[name]:.4f}")
        tip_r = radii.get("tip", TIP_CONTOUR_RADIUS_FALLBACK)
        body_r = max(
            (radii[k] for k in ("body", "pot", "pot_sides") if k in radii),
            default=1.05,
        )
    else:
        print("[!] player_contour.json 不可得，用兜底常量")
        tip_r = TIP_CONTOUR_RADIUS_FALLBACK
        body_r = 1.05
    print()

    tip_origin_bound = HUB_OFFSET + SLIDER_MAX_EXTENSION
    design_bound = max(tip_origin_bound, CURSOR_CLAMP_RADIUS)

    print("锤子几何链:")
    print(f"  |hub-player|            = {HUB_OFFSET:.4f} m  (刚性偏移)")
    print(f"  slider 全伸 |tip-hub|   = {SLIDER_MAX_EXTENSION:.4f} m  (关节硬限位)")
    print(f"  → tip 原点硬上界        = {tip_origin_bound:.3f} m")
    print(f"  PlayerControl cursor 钳制= {CURSOR_CLAMP_RADIUS:.2f} m  (IL 硬编码，取为设计上界)")
    print(f"  tip 轮廓半径            = {tip_r:.4f} m\n")

    need_tip = design_bound + tip_r
    need_contact = need_tip + CONTACT_EPSILON
    print("所需半窗:")
    print(f"  装下锤头全部顶点        = {design_bound:.2f} + {tip_r:.3f} "
          f"= {need_tip:.3f} m")
    print(f"  再含接触判定余量        = {need_tip:.3f} + {CONTACT_EPSILON} "
          f"= {need_contact:.3f} m")
    print(f"  body∪pot 轮廓 (恒在中心附近) = {body_r:.3f} m\n")

    ok = half_window >= need_contact
    margin = half_window - need_contact
    if ok:
        print(f"[PASS] 窗口半径 {half_window:.2f}m >= 所需 {need_contact:.3f}m "
              f"(余量 {margin:+.3f}m)")
        print("       锤头在任意伸展姿态下，轮廓与其接触地形均完整落在 contact patch 内。")
    else:
        print(f"[FAIL] 窗口半径 {half_window:.2f}m < 所需 {need_contact:.3f}m "
              f"(缺 {-margin:.3f}m)")
        need_res = need_contact * 2.0 / cfg.patch_size
        need_size = int(np.ceil(need_contact * 2.0 / cfg.contact_patch_resolution))
        print(f"       修复选项: contact_patch_resolution >= {need_res:.3f} m/px "
              f"(保持 patch_size={cfg.patch_size})")
        print(f"                 或 patch_size >= {need_size} px "
              f"(保持 {cfg.contact_patch_resolution} m/px，但需迁移 CNN Linear 权重)")

    # 最坏姿态枚举：tip 沿任意方向伸到设计上界，检查轮廓顶点的 Chebyshev 半径
    print("\n最坏姿态扫描（tip 沿 360° 各方向伸到设计上界，取轮廓顶点 Chebyshev 极值）:")
    worst = 0.0
    worst_deg = 0.0
    for deg in range(0, 360, 5):
        rad = np.deg2rad(deg)
        tx, ty = design_bound * np.cos(rad), design_bound * np.sin(rad)
        # 轮廓最坏顶点在远离原点方向上再加 tip_r，方形窗口取 Chebyshev
        cheb = max(abs(tx) + tip_r, abs(ty) + tip_r)
        if cheb > worst:
            worst, worst_deg = cheb, float(deg)
    print(f"  最坏 Chebyshev 半径 = {worst:.3f} m @ {worst_deg:.0f}° "
          f"(窗口半径 {half_window:.2f}m)")
    scan_ok = worst <= half_window
    print(f"  {'[PASS]' if scan_ok else '[FAIL]'} 方形窗口"
          f"{'完整覆盖' if scan_ok else '存在裁剪'}")

    return 0 if (ok and scan_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

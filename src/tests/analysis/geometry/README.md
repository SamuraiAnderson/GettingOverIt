# 观测几何

核对锤头、身体、接触和投放窗口是否与游戏一致。不改奖励，不产参考线。

- `analyze_effectiveness.py` — 用地形和训练配置检查效率图与奖励的数值行为。
- `verify_mask_aliasing.py` — 检查 1 米效率图格心采样会不会漏掉薄墙、抹掉薄台阶。
- `check_tip_patch_coverage.py` — 用锤子几何上界判断接触窗口是否装得下锤头。
- `diagnose_tip_rasterization.py` — 查明锤头通道整格为空是格心采样走样。
- `bench_patch_raster.py` — 比较面积覆盖栅格和旧二值栅格在 `build_patch` 上的开销。
- `diagnose_native_contacts.py` — 汇总原生接触审计，并量接触代理的姿态误差。
- `plot_contact_alignment.py` — 用已归档的动作前接触对动作画时序。
- `verify_body_reconstruction.py` — 核对用玩家中心和 hub 方向近似重建的身体、锅多边形。
- `verify_tip_reconstruction.py` — 核对用杆方向近似重建的锤头多边形。
- `deploy_exclusion_editor.py` — 在地图上涂圆形排除区，让落点筛选丢掉这些位置。
- `visualize_cached_drops.py` — 在地形图上分格显示投放超参和落点。
- `drop_point_physics.py` — 在游戏里测空投 settle 后的竖直落差，判定落点是否稳定。

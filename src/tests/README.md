# 测试与分析脚本总览

本目录下的脚本分三类，**区别在于要不要拉起游戏进程**——这决定了它能不能在 CI 式的
批量运行里跑，也决定了改代码后该跑哪些。

| 类别 | 目录 | 需要游戏 | 怎么跑 |
|---|---|---|---|
| 离线单元测试 | `training/test_*.py` | 否 | `python -m src.tests.run_unit_tests` |
| 离线分析 / 基准 | `analysis/` | 多数不需要 | 单独 `python -m src.tests.analysis.<name>` |
| 在环集成测试 | `control_interaction/`、`environment_interaction/` | **是** | 单独运行，需游戏可启动 |
| 历史标定 | `input_characteristics/` | 是（已完成） | 归档性质，一般不再运行 |

改动训练侧代码后的最小验证是跑一遍离线单元测试：

```bash
python -m src.tests.run_unit_tests          # 全部
python -m src.tests.run_unit_tests -k patch # 只跑名字含 patch 的
```

环境里没有安装 pytest，所有测试脚本都自带 `main()` 直接自跑；上面的运行器只是把它们
逐个以子进程拉起并汇总结果。

---

## 1. 离线单元测试 `training/`

纯计算，不碰游戏进程，秒级完成。

| 脚本 | 覆盖内容 |
|---|---|
| `test_contact_geometry_smoke.py` | 几何工具（点—线段距离、多边形最近距离、向下射线）与 5 维接触信号的提取，含 body / pot 分层各自触发 |
| `test_dynamics_dim_smoke.py` | `DYNAMICS_DIM=39` 的维度算术与 `build_dynamics` 在有/无接触信号下的行为 |
| `test_p1_ou_map_elites.py` | P1.1 OU 噪声的条件高斯与 `log_prob` 自洽、AR(1) lag-1 自相关；P1.2 MAP-Elites 的行为多样性准入与冷启动 |
| `test_p2_contact_integration.py` | P2.1 接触信号端到端：老 34D checkpoint 扩维加载、前向、向后兼容 |
| `test_patch_tip_coverage.py` | contact patch 窗口在锤子 360° 全伸时必须装得下锤头轮廓且不被边界截断 |
| `test_raster_area_coverage.py` | 面积覆盖率光栅化的数值正确性，以及亚像素轮廓不会被漏采成空通道 |

同目录下的 `diagnose_nan.py` / `repro_nan_mask.py` / `sanity_dualscale_patch.py` /
`smoke_sil.py` 是排障脚本而非回归测试，按需单独运行，不进批量运行器。

## 2. 离线分析与基准 `analysis/`

产出图表与量化结论，多数写到 `src/Data/analysis/`。

| 脚本 | 用途 |
|---|---|
| `env_geometry.py` | 环境几何的公共工具，被其余分析脚本复用 |
| `check_tip_patch_coverage.py` | 按锤子几何硬上界校验 patch 窗口半径是否够用 |
| `diagnose_tip_rasterization.py` | 扫描朝向与亚像素偏移，量化轮廓漏采率（空通道率） |
| `bench_patch_raster.py` | 面积覆盖率光栅化的性能开销基准 |
| `verify_mask_aliasing.py` | 地形 mask 栅格混叠检查 |
| `analyze_effectiveness.py` | 效率图与可达性分析 |
| `analyze_reference_fling.py` | 人工参考轨迹的挥杆分析（读 `src/Data/reference/`） |
| `calibrate_progress_wx.py` | 用参考轨迹标定 `progress_metric` 的横向权重 |
| `viz_exploration_noise.py` | 探索噪声的时序结构可视化 |
| `view_elite_trajectories.py` | **需要游戏**：精英轨迹可视化（精英池不落盘时靠现采样重建） |

## 3. 在环集成测试 `control_interaction/`

L 系列按能力分层递进，全部需要游戏进程。

| 脚本 | 验证目标 |
|---|---|
| `test_l1_l2_protocol_causality.py` | TCP 协议连通与输入—状态因果性 |
| `test_l3_repeatability.py` | 同输入序列的可复现性 |
| `test_l4_dose_response.py` | 输入幅值与响应的剂量关系 |
| `test_l5_agent_isolation.py` / `test_l5_1_parallel_input.py` | 多 agent 隔离与并行输入 |
| `test_l6_collider_visual.py` | 碰撞体收集与可视化对齐 |
| `test_l7_surface_airdrop.py` | 可着陆表面提取与空投候选点 |
| `test_l8_airdrop_deploy.py` | 批量投放与 settle 落差统计 |
| `test_l9_model_eval.py` | 模型评估（无噪声推理），结果落到 `runs/_eval/` |
| `test_l9_water_reset.py` | 落水复位行为 |
| `test_planb_premise.py` | Plan B 前提验证 |
| `test_fakecursor_state.py` / `test_mouse_detect.py` | fakeCursor 状态与鼠标注入 |
| `record_reference_trajectory.py` | 人工录制参考轨迹 → `src/Data/reference/` |
| `verify_body_reconstruction.py` / `verify_tip_reconstruction.py` | 从 33D 状态重建 body / tip 轮廓的正确性 |
| `visualize_trajectory.py` / `visualize_cached_drops.py` / `deploy_exclusion_editor.py` | 可视化与投放禁区编辑 |

`environment_interaction/` 是更早的一批环境交互测试（完全重置、状态存取、一致性），
详见该目录自带的 README。

## 4. 历史标定 `input_characteristics/`

游戏输入精度与有效范围的标定，已完成并固化为常量：精度 1.0、有效范围 [10, 100]、
饱和点 100（硬限制）。详见该目录 README，一般不需要重跑。

---

## 输出去向

| 产物 | 位置 | 是否入库 |
|---|---|---|
| 分析图表 | `src/Data/analysis/` | 是 |
| 人工参考轨迹 | `src/Data/reference/` | 是（无法重新生成） |
| 模型评估原始数据 | `runs/_eval/eval_*.json` | 否 |
| 模型评估汇总 | `runs/_eval/summary.csv` | 是 |
| 训练实验产物 | `runs/<tag>/` | 仅轻量记录，见 [`runs/README.md`](../../runs/README.md) |
| 游戏侧原始采集 | `src/Data/GameResults/` | 否 |

# 独立物理世界实验

这些入口只用于隔离游戏副本 `artifacts/reset_game`，不接入生产 PPO 启动/reset 流程。
每个 world 载入完整原生 Mian 场景，包含一个原始 Player，使用独立 PhysicsScene2D。
沿用既有控制器、输入平滑、地形、CCD 和物理参数；所有模拟仍在 Unity 主线程依次执行。

## 编译与运行

需要已配置的游戏副本及当前 `GameRuntime_v2.dll`，Python 使用
`getting-over-it-analysis` 环境。先关闭已有游戏，再开始在线实验。
下面编译命令只写 `bin/WorldIsolationExperiment`，不部署到游戏：

```powershell
& src/tests/control_interaction/world_isolation/build_probes.ps1
conda run --no-capture-output -n getting-over-it-analysis python src/tests/control_interaction/world_isolation/run_policy_isolated_8players.py --out runs/isolated_policy_new --steps 1200 --fps 30
conda run --no-capture-output -n getting-over-it-analysis python src/tests/control_interaction/world_isolation/check_isolated_policy_run.py runs/isolated_policy_new
```

策略入口使用 `runs/optimizer_momentum_fix_20261001/checkpoints/ppo_iter_0269.pt`，
复用 `training.local_evaluation`、观测构建和 TCP 协议进行实时确定性推理。
8 个 world 均从本次冷启动状态开始，不额外 reset、预热或投放；只支持一次轨迹，
不支持 RESET/TELEPORT/NEW_SNAPSHOT。正常运行后保留可见窗口；`--hidden` 会在结束后
关闭进程。窗口内左键拖动、滚轮缩放。轨迹相同时 8 个角色会重叠显示。
此脚本临时部署实验插件，结束时恢复配置、移除请求文件及插件；权重不更新。

`run_isolated_worlds.py` 是固定动作与 reset 对照，需要预先将编译好的
`IsolatedWorldExperiment.dll` 复制到副本的 `BepInEx/plugins`，完成实验后移除。
动作来源为本地归档
`runs/policy_0269_cold_8players_20261002_visual_fix/deterministic_000_agent1.json`：

```powershell
Copy-Item bin/WorldIsolationExperiment/IsolatedWorldExperiment.dll artifacts/reset_game/BepInEx/plugins/IsolatedWorldExperiment.dll
conda run --no-capture-output -n getting-over-it-analysis python src/tests/control_interaction/world_isolation/run_isolated_worlds.py --out runs/isolated_actions_new --worlds 2 --steps 1200 --resets 2
Remove-Item -LiteralPath artifacts/reset_game/BepInEx/plugins/IsolatedWorldExperiment.dll
```

`--other-zero` 改变第二世界的动作，用于检测其是否影响第一世界；`--startup-sync`
和 `--scene-sync-only` 是同步时机对照，不能当作正式修复开关。
`replay_clone_divergence.py` 在已有共享物理世界中重放相同动作；选用的
`CloneDivergenceProbe.dll` 或 `CallbackIsolationProbe.dll` 需同样单独部署/移除。
不部署探针时仅采集参考；两探针都由临时 `clone_probe.json` 激活。

## 已验证范围

- `runs/clone_divergence_step31_20261002/summary.json`：共享世界中第 31 步分歧在
  物理模拟内出现；碰撞回调屏蔽对照记录在 `callback_summary.json`。
- `runs/isolated_worlds_20261002/summary.json`：双世界同动作 1200 步、两次冷启动、
  三次 reset 的观测逐位一致，排除 timestamp；改变另一世界动作不影响第一世界。
- `runs/policy_0269_isolated_8players_20261002/consistency_check.json`：8 世界实时
  策略 1200 步，动作及观测一致，第 31/1200 步采样的刚体、控制器和关节状态一致。

独立世界能消除这些工况下的跨 player 分歧，但新增场景的初始化同步仍使其与原
单世界参考存在最多约 0.24 微米的初始刚体位置差，随后轨迹不同。
单世界起步前多做一次 SyncTransforms 可复现双世界的 1200 步结果。
这是单/多世界的初始化边界差异，不表示已经验收的同配置 reset/冷启动复现失效。
不宣称任意动作、整张地图或 Unity 隐藏求解器状态都逐位一致。

首轮 8 世界测试的轨迹完整，但旧实验接口在正常 CLOSE 时误写 `finished.json=ERROR`。
保留该原始记录，轨迹检查与运行收尾状态分别报告。当前版本显式处理 CLOSE，并由
Python 等待 `closed.json` 后再报告完成；提交前已编译，未中断用户留开的窗口重跑。

分析入口 `summarize_clone_divergence.py` / `summarize_isolated_worlds.py` 面向上述
固定归档目录，需要本机保留的完整原始数据；Git 只保存轻量结论和元数据。

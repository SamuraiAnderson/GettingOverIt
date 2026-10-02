# 通道验收

游戏进程已经起来之后，检查步进、输入和物理前提是否成立。不训练。

- `test_l1_l2_protocol_causality.py` — 检查 TCP 步进和状态变化是否一一对应。
- `test_l3_repeatability.py` — 检查相同输入是否得到相同的状态变化。
- `test_l4_dose_response.py` — 测量动作幅度和位移响应的关系。
- `test_l5_agent_isolation.py` — 检查多个复制体是否各自响应、互不串输入。
- `test_l5_1_parallel_input.py` — 检查同一帧里不同复制体能不能吃到不同动作。
- `test_l6_collider_visual.py` — 在游戏窗口里描出碰撞箱，供目视确认它跟着玩家走。
- `test_l7_surface_airdrop.py` — 从地形抽出可着陆表面并筛空投落点。
- `test_l8_airdrop_deploy.py` — 检查空投投放是否把复制体放到选定落点。
- `test_l9_water_reset.py` — 检查落水之后能否重置回起点。
- `test_fakecursor_state.py` — 检查回包里的 fakeCursor 位置和速度是否有效。
- `test_mouse_detect.py` — 零动作步进时，看状态会不会跟着真实鼠标变。
- `test_background_run.py` — 比较游戏在前台和最小化时的步进耗时。
- `test_isolated_desktop_launch.py` — 检查游戏能否启动到另一个 Win32 桌面。
- `diag_snapshot_injection.py` — 对拍快照前后，同一动作的注入是否失效。
- `visualize_trajectory.py` — 给上述通道测试画轨迹、距离和剂量响应曲线。
- `test_planb_premise.py` — 检查「多个复制体当作并行世界模型」所需的物理前提。
- `test_reset_state.py` — 独立验证同一 agent 的 reset 与冷启动参考、静止性及动作探针；跨 agent 偏离仅记录，不走 PPO。

## Reset 状态验收

```powershell
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_reset_state.py
```

默认启动 3 个独立游戏进程（依次运行），每个进程使用 2 agents：第一轮 reset
作为冷启动参照，然后依次执行随机排序的 12 个条件（无动作历史、40 步恒定动作、
200/1000 步逐 agent 独立随机动作，各配合 reset 后等待 0/0.25/1 秒）。每次 reset
回包后只执行 1 步零动作探针，暴露下一步才出现的残留冲击。共 39 个条件，全部通过
才退出 0；数值验收失败退出 1；异常、缺失条件或连接失败退出 2。
已有游戏、端口占用或输出目录已存在等预检失败也退出 2，可能尚未创建结果目录。

测试使用游戏初始化时的快照，不调用 new_snapshot、不做 100 步预热，不加载策略。
各条件在同一进程内顺序执行，历史会延续；等待差异不能单独解释成等待的因果效应。
第一次 reset 只是比较参照，也要检查静止性，不能因为所有角色一起旋转而判成功。
跨 agent 的差异不再决定 PASS：冷启动本身已有的复制体偏离留待另一项排查。
当前脚本固定使用 legacy33；比较函数也支持 35D。协议只暴露 player 的角速度，
因此不能宣称已验证所有刚体角速度、内部控制态或求解器缓存。

暂定验收阈值分别是：位置和 cursor 位置差 0.001m、各部件和 cursor 速度分量差
0.001m/s、player 角速度差 0.1°/s、角度差 0.05°。静止要求所有已观测部件的线速模长
不超过 0.05m/s，player 角速度绝对值不超过 1°/s；reset 和零动作探针均检查。
这些是可通过命令行修改的工程判据，不是已经测出的数值误差下限。state[9]、state[14]
是角度，和 state[27] 一起使用最短圆周差；state[4] 是角速度，不做环绕；timestamp 不比较。
`--allow-moving-start` 仅供诊断移动快照：仍报告静止性，但不作为 PASS 必要条件。

输出到新建的 `runs/reset_state_<UTC时间>/`：`result.json` 保存参数、阈值、插件和脚本
SHA-256、进程、配置、各条件判定和最差字段；每个条件的 NPZ 保存全部 agent 的动作历史、
历史观测、reset 回包及零动作探针，不丢弃完整数据。中途异常也保存已收到的原始回包。
脚本拒绝连接已有游戏或占用端口，只终止自己启动的进程，并恢复原 runtime_config 和模式信号。
端口默认读取 `src/config/project.json`。现有 D 诊断回包是缓存，因此未用它冒充实时空闲读数；
本脚本只能检查等待之后下一次 STEP 的可见效果。

离线校验判定器（不启动游戏、不写游戏配置）：

```powershell
conda run -n getting-over-it-analysis python -m unittest discover -s src/tests/control_interaction -p test_reset_state_checks.py -v
```

### 实验性固定起点修复（2026-10-02）

生产 DLL、生产配置和 PPO 入口没有切换。实验游戏副本在 `artifacts/reset_game/`。
三个配置开关默认均为 false：`experimentalControlledUpdates`、
`experimentalResetDiagnostics`、`experimentalSceneReset`。

编译时必须关闭部署，随后仅复制到实验副本：

```powershell
& 'C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/MSBuild/Current/Bin/MSBuild.exe' src/GameRuntime_v2/GameRuntime_v2.csproj /t:Build /p:Configuration=ResetExperiment /p:DeployPlugin=false /nologo /v:minimal
Copy-Item bin/ResetExperiment/GameRuntime_v2.dll artifacts/reset_game/BepInEx/plugins/GameRuntime_v2.dll
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_reset_state.py --game-exe artifacts/reset_game/GettingOverIt.exe --scene-reset --reset-diagnostics --probe-steps 40
```

此组合使用首次 RESET **之前**的实时 Q 读数和探针作为冷启动参考，不再用“第一次
reset”替代真实启动状态。其他条件均发送 R，比较同一 agent 的返回状态和完整探针。
`--probe-kind hold` 使用 `(40,30)`，`--probe-kind random` 使用固定种子的随机动作；
非零探针只要求起点静止，运动后仍要求与冷启动的同一 agent 轨迹一致。
Q 同时检查所有 7 个刚体、等待期间控制器与物理步进计数、自动时钟、对象数量和环境 epoch。
时间戳不参与数值或逐位比较；原始轨迹、动作和诊断均保留。

实现采用相同的固定启动配方：

- 首次启动和 R 都创建新的局部 `PhysicsScene2D`，R 卸载旧场景后重建；TCP 连接保留。
- 创建世界前统一 Script 物理、关闭自动 Transform 同步并暂停自动时钟。
- 等待原生 `PlayerControl.Start()`，补齐每个角色自己的 cursor 和控制状态，再初始化。
- 在原生 Start 前禁用 Saviour；阻止 CameraControl 的淡入结束逻辑重新开启时钟，禁用自动姿态动画。
- 动态刚体采用静态父节点，保留关节与已缓存的控制/观测引用；初始化显式同步 Transform。
- 自动 FixedUpdate 被实例级入口拦截，仅显式 STEP 调用控制器和所属物理世界。

这是**固定起点 reset**，不是任意快照恢复。N (`new_snapshot`) 明确不支持，调用会关闭
实验连接，避免返回伪成功。要求 `stepFrames=1`。原有 PPO 的 warmup/new_snapshot、空投、
可视化及暂停动画后的外观尚未完成兼容验收，因此不要把实验开关直接加到正式训练配置。
这里的冷启动指修复模式完成初始化后的启动参考，不是旧版生产插件的旧姿态。

最终在线验收使用同一实验 DLL（SHA-256
`4fade7d4cf492b757e63306b491fe0cd3424aee8a51aea585fb5c83da288bce9`）：

- `runs/reset_fix_certified_zero/result.json`：2 agents，3 次启动，39/39 条件通过；
  历史含 200/1000 步，40 步零动作探针，1599/1599 帧逐位一致。
- `runs/reset_fix_certified_hold/result.json`：2 agents，3 次启动，30/30 条件通过；
  40 步恒定动作探针，1230/1230 帧逐位一致。
- `runs/reset_fix_certified_random/result.json`：2 agents，3 次启动，30/30 条件通过；
  40 步随机动作探针，1230/1230 帧逐位一致。
- `runs/reset_fix_certified_8agents_1200/result.json`：8 agents，1 次启动，4/4 条件通过；
  1200 步随机动作探针，4804/4804 帧逐位一致。

以上逐位比较均排除 timestamp，比较对应 agent，不要求不同 agent 相同。
共 103 条件、8863 帧，未观察到 reset 相对该修复模式冷启动的额外偏差。
2-agent reset 实测 0.59–0.67 秒，8-agent 为 0.82–0.85 秒；冷启动参考的读取时间不计入。
18 项离线判定器回归测试通过。汇总及生产 DLL/配置未变的哈希记录位于
`runs/reset_fix_certified_8agents_1200/acceptance_summary.json`。
其他 `runs/reset_fix_*` 包含定位阶段的失败实验，不应当作修复成功的证据。

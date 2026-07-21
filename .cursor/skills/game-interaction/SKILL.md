---
name: game-interaction
description: 描述本项目（Getting Over It 强化学习）当前与游戏交互的方式：进程启动、模式选择、TCP 帧级步进协议、C# 服务端线程模型、输入注入与输出状态获取。当涉及 GoiEnv、TcpStepServer、StepController、RESET/STEP/TELEPORT 等命令、33D 状态回传、mouseInput 注入、Rewired 拦截、fakeCursor、多 agent 并行、GameLauncher/GameModeController/信号文件启动流程时应用。
---

# 与游戏交互的方式（Entrypoints）

## 核心形态（不可动摇）

- **Python = 训练侧 = TCP 客户端**；**C# `GameRuntime_v2`（BepInEx 插件，运行在游戏进程内）= 服务端（`TcpStepServer`）**。
- 每个 RL step = 一次阻塞式请求-响应，实现**帧级同步**。
- 端口从 `src/config/project.json` 的 `tcp_port`（C# 侧 `runtime_config.json` 的 `tcpPort`）读取，默认 9000，**禁止硬编码**。
- 三种互斥模式：`DataCollection`（导出碰撞箱后退出）/ **`GameRuntime`（核心，TCP 帧级步进）** / `GameTesting`（调试探测）。

## 启动链路

1. Python `GameModeController.set_game_runtime_mode()` 写信号文件 `GoiData/ControlSignals/mode_game_runtime.signal`。
2. `GameLauncher.launch()` 用 `subprocess.Popen` 拉起 `GettingOverIt.exe`。
3. 插件 `GameRuntimeManager.Awake()` 注册 Rewired 补丁、开后台运行；`OnSceneLoaded` 在 `Loader` 场景自动点击 NewGame，进 `Mian` 场景后 `InitializeGameRuntimeMode()`。
4. 初始化关键步骤：绑定 Player 服务 → 建复制体 → **`DisableWaterReset()`（销毁 `RestartOnContact`，否则落水会把物理改回 FixedUpdate）** → `StepController`（`STATE_DIM=33` 权威）→ `TcpStepServer.StartListening` → 激活 RL 拦截 → `TrainingLoop()` 协程等待连接。

## TCP 线路协议（小端二进制）

Python → C#：`R`(reset) / `S`(step,`[n][actions n×2×4B]`) / `N`(new_snapshot) / `C`(config 鼠标拦截) / `V`(可视化) / `E`(导出碰撞) / `T`(teleport,`[idx][x][y]`) / `F`(自由相机) / `X`(close)。

C# → Python STATE：`[n:1B]` + 逐 agent `[state 33×4B][done 1B]`，**state/done 交错**，Python 必须逐 agent 读取。

## C# 服务端

- `TcpStepServer`：后台线程阻塞收发 + 主线程协程 `TrainingLoop` 处理，`ManualResetEvent`+`volatile` 握手（Unity API 只能主线程调）。异常时回零状态空包避免 Python 死锁。
- `StepController`：初始化把 `Physics2D.simulationMode` 持久切 `Script`；`ExecuteStep` = 逐 agent `SetMouseInput` → `InvokeFixedUpdate`（**必须在 Simulate 前**，否则关节电机力缺失、复制体被弹飞）→ `Physics2D.Simulate × stepFrames` → `CollectAllStates`。另有 `Reset` / `TakeNewSnapshot` / `Teleport`（整体平移子刚体+fakeCursor）。

## 输入注入（动作 → 操作）

动作 = 2D 鼠标相对位移 `(dx,dy)`，`ClampInput` 限幅 ±100。双路注入：① 反射写 `PlayerControl.mouseInput`；② `RewiredMouseOverride` + Harmony 拦截 `GetAxis("mouseX"/"mouseY")` 返回注入值。仅在手动 `InvokeFixedUpdate`（`CurrentAgentIndex>=0`）时返回真实值，Unity 自动 FixedUpdate 返回 0（无操作）。`PlayerControl.Update()` 被 patch 跳过，防真实鼠标覆盖。

## 输出状态的获取（33D）

1. **29D**：`PlayerStateService.SampleCurrentState()` 反射读各部件 `Transform`/`Rigidbody2D` → `PlayerState.ToFloatArray()` 定序展开（0-4 player pos/vel/angVel，5-9 hub，10-14 slider，15-18 handle，19-22 pole，23-26 tip，27 hammerAngle 派生，28 timestamp）。**顺序即协议契约**。
2. **+4D fakeCursor（29-32）**：`CollectAllStates()` 反射读 `PlayerControl.fakeCursorRB` 绝对坐标+速度（未创建时回退 player 坐标+零速）。
3. 打包 `numAgents×33` flat float，`SendResponse` 小端交错编码。
4. **done 现状全 false**：C# 不判终止，落水/终止交 Python `reward.is_water`。
5. Python `GoiEnv._recv_response` 逐 agent 还原 `(num_agents,33)`。
6. 训练侧 `dataset.build_dynamics()` 才把绝对 33D 转平移等变特征（速度原样、角度→sin/cos、部件坐标取相对 player）。

## Python 客户端与生命周期

`GoiEnv`（`src/env/goi_env.py`）：`connect/close/reset/step/new_snapshot/teleport/config_rewired_mouse/set_camera_free/toggle_collider_visual/export_colliders`。`RolloutWorker`（`src/training/rollout.py`）标准流程：写 numDuplicates → 写模式信号 → 启动游戏 → connect → reset → 零动作 warmup → new_snapshot →（每轮）reset+teleport 部署 → step 采集 → close。

## 关键约束

- 端口不硬编码；状态维度权威 `StepController.STATE_DIM=33`（不信 `config.stateDimension`）。
- `InvokeFixedUpdate` 必须先于 `Simulate`；物理 Script 模式持久，只在 STEP 推进。
- 回包 state/done 交错；改状态维度/顺序会破坏 checkpoint 与轨迹缓存兼容性。

## 深入参考

完整分析、字段索引表、代码引用见 `doc/entrypoints.md`；数据流/模型/奖励见 `doc/training.md`；目录职责见 `.cursor/rules/project-standards.mdc`。

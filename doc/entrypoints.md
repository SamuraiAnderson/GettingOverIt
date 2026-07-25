# 与游戏交互的方式（Entrypoints）

本文分析**当前项目如何与《Getting Over It》游戏进行交互**：从进程启动、模式选择，到帧级别 TCP 步进控制的完整链路。

交互的根本形态：**Python 作为训练侧（TCP 客户端）**，**C# `GameRuntime_v2`（BepInEx 插件，运行在游戏进程内）作为服务端（`TcpStepServer`）**。每个 RL step 对应一次阻塞式请求-响应，实现帧级同步。端口从 `src/config/project.json` 的 `tcp_port` 读取（默认 9000），禁止硬编码。

---

## 一、总览：两个进程、三种模式、一条 TCP 通道

```
┌───────────────────────── Python 进程（训练侧 / 客户端） ───────────────────────┐
│  main.py / main_train.py / main_ppo.py                                       │
│     │                                                                        │
│     ├── GameModeController  ──写信号文件──►  GoiData/ControlSignals/*.signal  │
│     ├── GameLauncher        ──subprocess──►  启动 GettingOverIt.exe           │
│     └── GoiEnv (TCP client) ──socket──────►  127.0.0.1:9000                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ TCP（小端二进制帧）
                                    ▼
┌────────────────── 游戏进程内 BepInEx 插件 GameRuntime_v2（服务端） ─────────────┐
│  GameRuntimeManager (BepInPlugin)                                            │
│     ├── ModeManager        读信号文件 / 配置 → 决定模式                        │
│     ├── TcpStepServer      后台线程收发 + 主线程协程处理                        │
│     ├── StepController     Physics2D.Simulate 帧级推进 + 快照/重置/传送        │
│     ├── PlayerInputService 反射注入 mouseInput + Rewired GetAxis 拦截          │
│     └── PlayerStateService 反射采集 29D 状态（+fakeCursor=33D）                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

C# 插件支持三种互斥运行模式（`GameMode`），交互方式因模式而异：

| 模式 | 枚举 | 用途 | 与游戏的交互 |
|------|------|------|--------------|
| 数据采集 | `DataCollection` | 前期采集 Mountain 碰撞箱 | 无 TCP，进游戏后自动导出 `environment.json` 并退出 |
| **游戏运行** | `GameRuntime` | **AI 训练交互（核心）** | **启动 `TcpStepServer`，Python 通过 TCP 帧级步进** |
| 游戏测试 | `GameTesting` | 交互性 / 物理能力探测 | 无 TCP，`PlayerDebugTool` 键盘调试 + 物理探测日志 |

下面各节聚焦**游戏运行模式**这一核心交互路径。

---

## 二、启动链路：从 Python 到游戏内插件

### 2.1 模式选择（信号文件）

Python 侧不直接控制 Unity，而是通过**信号文件**告知插件本次以哪种模式启动：

- `GameModeController.set_game_runtime_mode()` 在 `<game_root>/GoiData/ControlSignals/` 下写入 `mode_game_runtime.signal`（先清空其它模式信号）。
- C# `ModeManager.DetermineMode()` 启动时按 **信号文件 > 配置文件** 的优先级决定模式，读取后立即删除信号文件（一次性）。

```50:54:src/start/game_mode_controller.py
    def set_mode(self, mode: GameMode):
        self.clear_all_mode_signals()
        signal_file = self.signal_dir / f"mode_{mode.value}.signal"
        signal_file.write_text("")
```

### 2.2 启动游戏进程

`GameLauncher` 从 `project.json` 读取可执行路径，用 `subprocess.Popen` 启动游戏（非阻塞）：

```40:57:src/start/game_launcher.py
    def launch(self, wait: bool = False):
        """
        启动游戏
        ...
        """
        logger.info("启动游戏: %s", self.game_exe.name)

        try:
            process = subprocess.Popen(
                [str(self.game_exe)],
                cwd=str(self.game_root),
```

### 2.3 插件初始化（游戏内）

`GameRuntimeManager`（`BaseUnityPlugin`）是插件统一入口：

- `Awake()`：注册 Rewired 输入拦截补丁、开启后台运行（`Application.runInBackground=true`、关 vSync）、加载配置、订阅场景加载事件。
- `OnSceneLoaded()`：
  - 在 `Loader` 主菜单场景，协程 `AutoLoadMainScene()` **自动查找并点击「NewGame」按钮**进入游戏（反射调用 `Button.onClick.Invoke`，最多重试 10 次）。
  - 进入主场景 `Mian` 后，按模式调用 `InitializeGameRuntimeMode()` 等初始化。

> 交互特性：整个进游戏流程无需人工点击，Python 只负责写信号 + 拉起进程，插件自动推进到可训练状态。

### 2.4 游戏运行模式初始化 `InitializeGameRuntimeMode()`

按顺序完成以下关键动作（决定后续一切交互能否成立）：

1. 初始化 `PlayerStateService` / `PlayerInputService`（绑定场景中的 `Player` 对象）。
2. 按 `config.numDuplicates` 创建复制体（多 agent 并行，见第六节）。
3. **`DisableWaterReset()`**：销毁场景内所有 `RestartOnContact` 组件。落水本会触发 `Saviour` 重置并把 `Physics2D.simulationMode` 改回 `FixedUpdate`，从而**破坏帧级控制**，因此训练模式下必须移除。
4. 创建 `StepController`（`STATE_DIM=33` 为唯一权威维度）。
5. 启动 `TcpStepServer.StartListening(config.tcpPort)`。
6. 激活 RL 输入拦截：`PlayerControlUpdatePatch.RlModeActive = true` + `RewiredMouseOverride.Active = true`。
7. 附加 `ColliderVisualizer`（默认开）与 `FreeCameraController`（默认关）。
8. 启动 `TrainingLoop()` 协程，等待 Python 连接。

---

## 三、TCP 线路协议（交互的核心契约）

Python 与 C# 之间是一套**小端二进制**、请求-响应式协议。命令定义与编解码在 `src/env/goi_env.py` 顶部注释和 `TcpStepServer` 中保持一致：

### 3.1 Python → C# 命令

| 命令 | 字节格式 | 含义 |
|------|----------|------|
| `RESET` | `['R']` | 重置所有 agent 到初始快照 |
| `STEP` | `['S'][n:1B][actions: n×2×4B]` | 注入动作并推进物理 `stepFrames` 帧 |
| `NEW_SNAPSHOT` | `['N']` | 以当前状态为新基准重拍快照（warmup 后） |
| `CONFIG` | `['C'][active:1B][mouseXId:4B][mouseYId:4B]` | 配置 Rewired 鼠标拦截（L3 可重复性测试） |
| `VISUALIZE` | `['V'][enabled:1B]` | 开/关碰撞箱描边可视化 |
| `EXPORT_COLLIDERS` | `['E']` | 导出碰撞体几何到文件 |
| `TELEPORT` | `['T'][agentIndex:1B][x:4B][y:4B]` | 传送指定 agent 到世界坐标（空投部署） |
| `CAMERA_FREE` | `['F'][enabled:1B]` | 启用/禁用自由相机 |
| `CLOSE` | `['X']` | 关闭训练循环并断开 |

### 3.2 C# → Python 回包（STATE）

```
[n: 1B]  然后对每个 agent 交错发送:  [state_i: 33×4B][done_i: 1B]
```

- **交错发送**（state 与 done 逐 agent 交替），Python 端 `_recv_response()` 必须逐 agent 读取，不能先读所有 state 再读所有 done。
- `CONFIG` / `VISUALIZE` / `EXPORT_COLLIDERS` / `CAMERA_FREE` 回空包（`n=0`）。

状态 33 维 = 基础 29 维（`PlayerState.ToFloatArray`）+ fakeCursor 4 维（`cursorX, cursorY, cursorVelX, cursorVelY`，绝对量，索引 29-32）。

---

## 四、C# 服务端：线程模型与主循环

### 4.1 `TcpStepServer` 双线程 + 事件握手

Socket IO 与 Unity 物理必须分离（Unity API 只能在主线程调用）：

- **后台线程 `BackgroundLoop()`**：阻塞 `AcceptTcpClient` → 循环读命令 → 写入 `_pendingCommand`，`_commandReady.Set()` 通知主线程 → 阻塞等 `_responseReady` → 把主线程准备好的 `_responseBytes` 写回 socket。
- **主线程协程 `TrainingLoop()`**：每帧非阻塞 `TryGetCommand()`，命中则执行物理操作并 `SendResponse()`（`_responseReady.Set()`）。

两方向通过 `ManualResetEvent` + `volatile` 变量握手，无需锁。这套机制保证**一次 STEP = Python 阻塞等待 → C# 单次物理推进 → 回包**，实现帧级同步。

```443:474:src/GameRuntime_v2/Core/GameRuntimeManager.cs
		private System.Collections.IEnumerator TrainingLoop()
		{
			while (tcpStepServer != null && tcpStepServer.IsRunning)
			{
				// 非阻塞轮询（每帧检查一次）
				if (tcpStepServer.TryGetCommand(out StepCommand cmd))
				{
					StepResponse resp;
```

> 容错：`TrainingLoop` 对每个命令处理 try/catch，异常时发送零状态空回包，避免 Python 永久阻塞。

### 4.2 `StepController`：帧级物理控制

`Initialize()` 里做了决定性一步：**将 `Physics2D.simulationMode` 持久切换为 `Script`**，使游戏物理不再自动推进，完全由 `ExecuteStep` 驱动。

`ExecuteStep(actions)` 每步流程：

1. 对每个 agent：`SetMouseInput()` 注入动作 → `InvokeFixedUpdate()`（**必须在 Simulate 前**，否则关节电机力不生效）。
2. `Physics2D.Simulate(dt)` × `stepFrames`。
3. `CollectAllStates()` 采集并打包为 flat float 数组。

```186:194:src/GameRuntime_v2/Core/StepController.cs
			// 2. 推进物理
			float dt = Time.fixedDeltaTime;
			for (int f = 0; f < stepFrames; f++)
			{
				Physics2D.Simulate(dt);
			}

			// 3. 采集状态（诊断：打印 Simulate 后的速度）
			float[] result = CollectAllStates();
```

其它 API：
- `Reset()`：所有刚体 + fakeCursorRB **+ PlayerControl 动作相关内部态**恢复初始快照，清零注入值，再零输入 `InvokeFixedUpdate` + `Simulate` 一帧稳定关节。
  - **修复1（已实施）**：`Reset` 经 `PlayerInputService.SetInternalState` 恢复 `mouseVelocityAverage`/`oldMouse`/`mw`/`oldAngle`/`pauseInputTimer`/`mouseSnap`/`inputsToSkip`，消除跨 reset 增益残留。`CaptureAllSnapshots` 同时经 `BroadcastCanonicalSnapshot` 把 agent0 权威态广播到所有复制体（修复2）。
  - **残留下限（见 `doc/planb_premise_verification.md`）**：Unity `Physics2D`（Box2D）求解器内部态（接触/关节 warm-start 冲量）不可经刚体快照恢复，形成 ~1e-4 非确定性地板。静止启动姿态下零动作跨 reset 可复现到近机器精度（P0b≈2e-7），摆动姿态下被放大到 ~0.1；激烈动作则由确定性混沌放大。故「游戏即**逐位**世界模型」不可达，方案 B 只能作近似短时域 MPC。
- `TakeNewSnapshot()`：以当前物理状态重拍快照（warmup 后调用）。
- `Teleport(agentIndex, target)`：以根刚体为基准整体平移所有子刚体 + fakeCursor，清零速度，稳定一帧。用于多 agent 空投部署。

---

## 五、输入注入与状态采集（反射 + Harmony）

### 5.1 动作如何变成游戏内操作

动作是 **2D 鼠标相对位移 `(dx, dy)`**，注入采用**双路**：

1. **反射写 `mouseInput` 字段**：`PlayerInputService` 反射拿到 `PlayerControl.mouseInput`（`NonPublic Instance`）直接写入（兼容旧路径）。
2. **Rewired 层拦截**：`RewiredMouseOverride.SetForAgent()` 记录各 agent 注入值；Harmony patch 了 `Rewired` 的 `GetAxis("mouseX"/"mouseY")`，RL 模式激活时返回注入值、屏蔽真实鼠标。

关键细节：只有在**手动 `InvokeFixedUpdate()`** 上下文里（`CurrentAgentIndex >= 0`）拦截才返回真实注入值；Unity 自动触发的 FixedUpdate（`CurrentAgentIndex=-1`）对鼠标轴返回 0，成为无操作。这保证物理推进完全由我们的 STEP 驱动。

同时 `PlayerControl.Update()` 被 Harmony prefix patch，RL 模式下整体跳过，防止游戏从真实鼠标读输入覆盖注入值。输入范围经 `ClampInput` 限制在 `[-100, 100]`。

### 5.2 状态（输出）如何获取与回传

一次 STEP 回传的观测经过 **C# 采集 → 追加 → 打包 → Python 解析 → 训练侧再加工** 五步。

#### （1）C# 采集单 agent 的 29 维

`PlayerStateService` 在初始化时 `GameObject.Find("Player")` + 深度查找绑定各部件（Hub/Slider/Handle/PoleMiddle/Tip/PotCollider）的 `Transform`/`Rigidbody2D`。每次采集由 `SampleCurrentState()` 逐部件读取位置/速度/角度填入 `PlayerState` 结构，再由 `PlayerState.ToFloatArray()` 按**固定顺序**展开为 29 维（顺序即协议契约，改动会破坏 checkpoint 兼容性）：

| 索引 | 字段 | 来源 |
|------|------|------|
| 0-1 | playerX, playerY | `playerTransform.position` |
| 2-3 | velocityX, velocityY | `playerRigidbody.velocity` |
| 4 | angularVelocity | `playerRigidbody.angularVelocity` |
| 5-6 | hubX, hubY | `hubTransform.position` |
| 7-8 | hubVelX, hubVelY | `hubRigidbody.velocity` |
| 9 | hubAngle | `hubTransform.eulerAngles.z` |
| 10-14 | slider X/Y/VelX/VelY/Angle | Slider |
| 15-18 | handle X/Y/VelX/VelY | Handle |
| 19-22 | pole X/Y/VelX/VelY | PoleMiddle |
| 23-26 | tip X/Y/VelX/VelY | Tip |
| 27 | hammerAngle | 派生量 `atan2(tip - hub)` |
| 28 | timestamp | `Time.time` |

#### （2）`StepController.CollectAllStates()` 追加 fakeCursor 4 维（29-32）

对每个 agent 取 29 维后，反射读 `PlayerControl.fakeCursorRB`（不可见的鼠标锚点，锤子控制链的隐藏状态，属观测必要组成）追加绝对坐标 + 速度。若 fakeCursorRB 尚未创建，回退用 player 坐标（相对=0）+ 零速度，避免巨大相对位移：

```356:368:src/GameRuntime_v2/Core/StepController.cs
				float cx = s[0], cy = s[1], cvx = 0f, cvy = 0f;
				Rigidbody2D fcRB = (i < inputServices.Count) ? inputServices[i]?.GetFakeCursorRB() : null;
				if (fcRB != null)
				{
					cx = fcRB.position.x;
					cy = fcRB.position.y;
					cvx = fcRB.velocity.x;
					cvy = fcRB.velocity.y;
				}
				result[baseOffset + BASE_STATE_DIM + 0] = cx;
				result[baseOffset + BASE_STATE_DIM + 1] = cy;
				result[baseOffset + BASE_STATE_DIM + 2] = cvx;
				result[baseOffset + BASE_STATE_DIM + 3] = cvy;
```

| 索引 | 字段 | 说明 |
|------|------|------|
| 29-30 | cursorX, cursorY | fakeCursorRB 绝对坐标 |
| 31-32 | cursorVelX, cursorVelY | fakeCursorRB 速度 |

产物是长度 `numAgents × 33` 的扁平 `float[]`（布局 `[s0_0..s0_32, s1_0..s1_32, ...]`）。

#### （3）打包回包（小端 + 交错）

`TcpStepServer.SendResponse()` 编码为 `[n:1B]` + 逐 agent `[33×4B state][1B done]`。每个 float 显式按小端写出（`BitConverter.GetBytes`，非小端平台 `Array.Reverse`）。state 与 done **逐 agent 交错**。

#### （4）done 标志现状

`TrainingLoop` 中 RESET / STEP 回包的 `Dones` 恒为 `new bool[NumAgents]`（**全 false**）——C# 侧当前不判定终止；落水/终止完全交给 Python 侧单一入口 `reward.is_water` 处理。

#### （5）Python 解析 `GoiEnv._recv_response()`

按 `n` 逐 agent 读 `33×4B` state + `1B` done，`np.frombuffer(raw, "<f4")` 还原，`np.stack` 成 `(num_agents, 33)`，done 为 `(num_agents,)` bool。因交错发送，**必须逐 agent 读取**，不能先读所有 state 再读所有 done。

```267:277:src/env/goi_env.py
        for _ in range(n):
            raw_state = self._recv_exact(STATE_DIM * 4)
            raw_done  = self._recv_exact(1)
            all_states.append(np.frombuffer(raw_state, dtype="<f4").copy())
            all_dones.append(bool(raw_done[0]))
```

#### （6）训练侧再加工

回传的 33D 是**绝对量**；`dataset.build_dynamics()` 才在 Python 侧把它转为**平移等变**特征（速度原样、角度 → sin/cos、部件坐标取相对 player 的差值）。这一步离线/在线共用同一实现，保证训练/推理一致（细节见 `doc/training.md`）。

**注**：模型消费的 `DYNAMICS_DIM = 39`（34 base + 5 接触信号），后 5 维为 `tip_contact` /
`tip_grip` / `body_contact` / `pot_contact` / `fall_distance_norm`，Python 端由
`contact_features.py` 复刻游戏原生 `HammerCollisions` + `PlayerSounds` 接触判断
（多边形几何 + 向下射线）；`body_contact` 与 `pot_contact` 按游戏 IL 里
`rb.GetPoint(contact).y > 0.4` 分层各自输出（body 触地 → 撞头 pain，pot 触地 →
稳定支点，语义相反）。wire 协议不变（仍 33D）。详见 P2.1 落地说明
（`doc/optimization_roadmap.md`）。

---

## 六、多 Agent 并行

一次 STEP 同时推进 N 个 agent 以提高样本效率（Option A：单进程多复制体）：

- agent 0 = 原始 `Player`；agent 1..N-1 = `PlayerDuplicateManager` 复制体，各有独立的 Input/State 服务与 `AgentIndex`。
- N 由 `config.numDuplicates` 决定；Python 侧 `rollout.py` 的 `_write_num_duplicates()` 在启动前写入 `GoiData/runtime_config.json`，与 `GoiEnv(num_agents=...)` 保持一致。
- actions 布局 `[a0_x,a0_y, a1_x,a1_y, ...]`；states 布局同序 flatten。

---

## 七、落点筛选（离线预处理，L7）

「落点筛选」是一段**独立于训练主循环的离线预处理**：先在可着陆表面上生成候选投放点，再通过**游戏内物理探测**保留能稳定停住的点，缓存供训练时的 `TELEPORT` 空投部署复用。它不在训练热路径上，但决定空投落点的质量。

**命令行入口**：`src/tests/control_interaction/test_l7_surface_airdrop.py`（argparse CLI，`main()`）。

流程分两步：

### 7.1 几何采样候选点（纯数学，无游戏依赖）

`src/training/deploy_sampling.py`：

- `build_all_deploy_candidates()`：取所有可着陆线段的折线顶点，y 加 `drop_height` 后去重，作为候选。
- `sample_deploy_candidates_stratified()` / `compute_drop_points()`：顶点过多时按**弧长权重**（`height_decay` 可偏向低处）在弧上采样定量候选点（最大余数法分配到各线段）。

### 7.2 物理稳定性筛选（需连游戏）

`src/tests/control_interaction/drop_point_physics.py`，一个候选被保留须**同时满足两个判据**：

1. **竖直稳定**：`teleport(x, y)` → 若干步零动作 settle → `drop = teleport_y - settled_y`，要求 `drop <= settle_drop_threshold`。
2. **锤头未卡进障碍物**：settle 后取锤头（tip）中心 `obs[.,23:25]`，若其落在任一实心碰撞多边形内则判为**无效落点**并计入 `rejected_tip_in_obstacle`（`deploy_sampling.tip_in_obstacle`）。`solid_polygons=None` 或 L7 `--no-tip-obstacle-check` 关闭整个检查。默认开启中心点判据。`tip_cfg`（轮廓重建配置）保留仅为向后兼容，不再参与判定。

其它要点：

- 默认 **L8 批量探测**：一次 `reset` 后多复制体（≤64）同时传送再统一 settle，与 `RolloutWorker._deploy_agents` 部署方式一致、更快；可设 `use_l8_batch_deploy=False` 退回逐点 2-agent 模式。
- 三个 API：`filter_all_drop_points_stable`（收集全部稳定点）/ `filter_drop_points_stable`（凑满 `target_count` 即停）/ `probe_drop_points_stability`（只标记不筛）。均接受 `solid_polygons` 参数。

### 7.2.1 人工涂抹排除区（可选，交互）

除自动的稳定/锤头判据外，可用**圆形笔刷**在地图上手动涂抹掉不想投放的区域（如落水口、陷阱、已知死路）。

- **编辑器**：`src/tests/control_interaction/deploy_exclusion_editor.py`（matplotlib 交互）。加载 L7 地图（环境多边形 + 可着陆表面 + 已筛稳定落点）后：左键拖动涂抹、右键拖动擦除、**中键拖动平移**、**滚轮以光标为中心缩放**、`[`/`]` 调笔刷半径、`e` 切换涂/擦、`f` 复位视图、`u` 撤销一笔、`c` 清空、`s` 保存、`r` 重载、`q` 退出。落点按「保留/排除」实时着色。
- **产物（随仓库版本化）**：`checkpoints/deploy_exclusion_zones.json`（项目内，非游戏目录；`deploy_sampling.default_exclusion_zones_path()`），格式 `{"zones": [[cx, cy, r], ...], "brush_radius", "drop_height"}`，坐标系与投放点一致（世界 x/y，y 含 `drop_height`）。
- **判据**：落点落入任一圆内（`dist ≤ r`）即剔除（`deploy_sampling.point_in_exclusion` / `filter_points_by_exclusion`）。
- **生效范围**：
  - **离线 L7**：物理探测**前**先过滤候选（省探测开销）；`--exclusion-zones <path>` 指定文件，`--no-exclusion` 关闭，默认自动读取上述项目路径。
  - **运行时 rollout**：`RolloutWorker.setup()` 加载排除区，`_sample_positions` 剔除落入圆的随机采样点并重采补足，固定回退池加载时一并过滤；由 `config.deploy_exclusion_check`（默认 True）开关控制。

### 7.3 产物与训练衔接（候选点唯一文件 → train）

L7 `--physics-filter-all` 产出**全量稳定候选集**，作为向训练传递候选点的**唯一文件**，写入本项目 `checkpoints/drop_points_all_stable.json`（随仓库版本化，`deploy_sampling.default_candidate_points_path()`）。

- `checkpoints/drop_points_all_stable.json`：**唯一候选文件**（train 读取源）。
- `<game_root>/GoiData/Colliders/drop_points.json`：L8 子集（≤64），仅供独立部署测试 `test_l8_airdrop_deploy.py` 手动使用，与 train 无关。

训练侧投放为 **pool-only**：`main_ppo` / `main_train` 读取上述唯一候选文件 → 在 train 侧做**高度过滤**（保留 `y ≤ y_max_cutoff + drop_height`，`filter_points_by_height`）→ `RolloutWorker.set_candidate_points()`（同时按人工涂抹排除区过滤、清空表面线段）。此后 `_sample_positions` 每轮从候选池无放回抽样投放（`sample_from_fixed_pool`），`_deploy_agents()` 用 `TELEPORT` 部署。**不再在训练侧重算表面线段随机采样**（`set_surface_segments` / `compute_landable_surfaces` 已从 train 路径移除）；表面线段采样仅作为无候选池时的历史回退保留在 `_sample_positions` 内。

---

## 八、Python 客户端 API（`GoiEnv`）

`GoiEnv` 是交互的唯一客户端封装（`src/env/goi_env.py`）：

| 方法 | 命令 | 说明 |
|------|------|------|
| `connect()` / `close()` | — / `X` | 连接（含重试等待）/ 断开 |
| `reset()` | `R` | 返回 `(num_agents, 33)` 初始观测 |
| `step(actions)` | `S` | 返回 `(obs, dones)` |
| `new_snapshot()` | `N` | warmup 后重拍快照 |
| `teleport(x, y, agent_index)` | `T` | 空投部署 |
| `config_rewired_mouse(...)` | `C` | 配置鼠标拦截（测试用） |
| `set_camera_free(enabled)` | `F` | 自由相机 |
| `toggle_collider_visual(enabled)` | `V` | 碰撞箱可视化 |
| `export_colliders()` | `E` | 导出碰撞几何 |

连接时先 5s 超时握手，成功后切阻塞模式（无超时），避免 C# 协程处理延迟触发 `TimeoutError`。

### 典型交互生命周期（`RolloutWorker`）

`src/training/rollout.py` 封装了完整生命周期，是训练侧与游戏交互的标准范式：

```237:256:src/training/rollout.py
    def launch_game(self) -> None:
        """启动游戏进程、TCP 连接、warmup、拍快照、开启自由相机。可多次调用。"""
        game_root = Path(self.config.game_root)
        _write_num_duplicates(game_root, self.config.num_agents)
        GameModeController(str(game_root)).set_game_runtime_mode()
        self._game_process = GameLauncher().launch(wait=False)

        self.env = GoiEnv(
            port=self.config.port,
            num_agents=self.config.num_agents,
        )
        self.env.connect()
        self.env.reset()

        _warmup_and_snapshot(
            self.env, self.config.num_agents, self.config.warmup_steps
        )
        self.env.set_camera_free(True)
```

标准流程：**写 numDuplicates → 写模式信号 → 启动游戏 → 连接 TCP → reset → 零动作 warmup → new_snapshot → （每轮）reset + teleport 部署 → step 循环采集 → close**。

---

## 九、关键约束与易错点

- **端口不可硬编码**：统一从 `project.json` 的 `tcp_port` 读取；C# 侧从 `runtime_config.json` 的 `tcpPort` 读取，二者需一致（默认 9000）。
- **状态维度权威为 `StepController.STATE_DIM=33`**：不依赖持久化的 `config.stateDimension`，避免旧配置残留 29 造成协议错位。改维度会破坏 checkpoint / 轨迹缓存兼容性。
- **`InvokeFixedUpdate` 必须在 `Simulate` 之前**：否则关节电机力缺失，复制体会被求解器弹射到 (0,0)。
- **落水不再由游戏重置**：`RestartOnContact` 已被销毁；落水判定改由 Python 侧单一入口 `reward.is_water` 或 C# `done` 决定。
- **回包是 state/done 交错**：Python 解析时必须逐 agent 读取。
- **物理为 Script 模式且持久**：初始化后游戏物理只在 STEP 时推进，不会自动前进。

---

## 十、相关文件索引

| 层 | 文件 | 职责 |
|----|------|------|
| Python 入口 | `src/main.py` | 数据采集主程序 |
| Python 入口 | `src/training/main_train.py` / `main_ppo.py` | BC / PPO 训练入口 |
| PPO 运行手册 | `doc/main_ppo_usage.md` | CLI、推荐配方、废弃清单 |
| Python 客户端 | `src/env/goi_env.py` | `GoiEnv` TCP 客户端 |
| Python 生命周期 | `src/training/rollout.py` | `RolloutWorker` 管理游戏交互全流程 |
| Python 落点筛选 | `src/tests/control_interaction/test_l7_surface_airdrop.py` | L7 落点筛选 CLI 入口（采样 + 物理筛选 + 缓存） |
| Python 落点排除 | `src/tests/control_interaction/deploy_exclusion_editor.py` | 人工涂抹排除区编辑器（圆形笔刷 → `deploy_exclusion_zones.json`） |
| Python 落点几何 | `src/training/deploy_sampling.py` | 表面候选点几何采样（纯数学） |
| Python 落点物理 | `src/tests/control_interaction/drop_point_physics.py` | 游戏内 settle 稳定性物理筛选 |
| Python 启动 | `src/start/game_launcher.py` / `game_mode_controller.py` | 启动进程 / 写模式信号 |
| C# 入口 | `src/GameRuntime_v2/Core/GameRuntimeManager.cs` | 插件入口、模式初始化、训练主循环 |
| C# 通信 | `src/GameRuntime_v2/Communication/TcpStepServer.cs` | TCP 双线程步进服务端 |
| C# 步进 | `src/GameRuntime_v2/Core/StepController.cs` | 物理推进、快照/重置/传送、状态采集 |
| C# 输入 | `src/GameRuntime_v2/PlayerControl/PlayerInputService.cs` / `MouseInputOverride.cs` | 反射注入 + Rewired 拦截 |
| C# 状态 | `src/GameRuntime_v2/PlayerControl/PlayerStateService.cs` | 29D 状态采集 |
| C# 模式/配置 | `src/GameRuntime_v2/Core/ModeManager.cs` / `Configuration/RuntimeConfig.cs` | 模式判定与运行时配置 |

> 完整数据流、模型结构、奖励/评分细节见 `doc/training.md`；PPO CLI / 配方见 `doc/main_ppo_usage.md`；目录职责与通信规范见 `.cursor/rules/project-standards.mdc`。

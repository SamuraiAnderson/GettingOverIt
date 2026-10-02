# 统一多 player 运行接口

GameRuntime 的生产后端现在固定为独立物理世界。Python 可用
`start.multi_player.launch_players(num_agents, game_root=..., port=...)` 统一配置并启动，或用
`start.runtime_config.configure_players(game_root, num_agents, port)` 配置数量，
用 `GoiEnv` 批量运行；C# 的 `GameRuntimeManager` 只调用 `IMultiPlayerRuntime`，
由 `IsolatedWorldRuntime` 实现。每个 player 载入一个完整原生 Mian 场景，拥有
独立 `PhysicsScene2D`、Player、fakeCursor、关节和地形。不存在共享世界复制体后端。

`StepController` 只负责一个世界的控制和物理步进，持有明确的物理场景句柄。
每个 STEP 对所有 player 各执行一次原生控制器和一次物理模拟，依次运行于 Unity
主线程；这不是多线程物理。保持原生输入平滑、物理参数、CCD、动作 Nx2 和
legacy33/anchor35 观测布局。当前要求 `stepFrames=1`，数量范围 1..64。

配置项改为 `numAgents`。C# 仍接受仅包含 `numDuplicates` 的旧配置；Python
配置入口会移除旧键。历史 `experimentalControlledUpdates`、`experimentalSceneReset`、
`experimentalNativeNewGame`、`experimentalNativeInputSmoothing` 不再选择生产后端。
原生动力学对照 `dynamicsProbeMode=native/native_dual` 仍是严格限制单 player 的
原生生命周期测试，不属于批量运行后端。

## 初始化、快照与 reset

所有世界完成原生 NewGame 开场后，统一执行一次 `Physics2D.SyncTransforms()`
作为可操作边界，单世界也执行相同同步。这替代过去未同步的单世界参考边界。
TCP 仅在全部世界就绪后监听，无需再固定等待 8 秒。

RESET 卸载旧场景并新建求解器，然后重放快照配方。NEW_SNAPSHOT 保存从冷启动
到当前边界的动作及投放操作记录；恢复时通过实际模拟重建接触、关节 warm-start
和输入平滑历史，不使用刚体数值覆盖来冒充完整求解器快照。
因此训练原有的“100 步预热 → NEW_SNAPSHOT → 每轮 RESET”仍可使用。
快照包含当时每个 player 自己的状态，不再向其他 player 广播 player 0 的姿态。
快照是当前进程内的记录，不是跨版本持久化文件。

TELEPORT 整体移动目标 player 的所有刚体和 fakeCursor，清零它们的速度，执行
一次零输入控制与物理步进。它只推进目标世界；其他 player 保持原状态。
自由相机的位置、缩放、开关和碰撞轮廓显示开关在 reset 后保留。
几何导出只包含一个世界的地形，以及完整 Player 物理部件。

代价是更多内存、每步需要 N 次物理模拟，reset 需要重新载入 N 个场景并重放
快照前的操作。快照离冷启动越远，reset 重放越久。旧权重仍可加载，但旧共享
求解器或不完整 reset 下的轨迹不应被视为这个后端的等价参考。

## 调用入口

PPO、BC/随机 rollout、local_evaluation、投放筛选、几何与锚点验证、reset/输入
测试、固定动作回放、8 player 权重展示均使用同一后端和配置入口。
`local_evaluation --game-root PATH` 可以选择隔离副本，实际启动路径与配置一致。
GameTesting 中 F9 进入 4 player 独立世界调试；进入后 F9 reset、F10 各世界执行
不同动作的一步、F11 打印诊断、F12 切换 1/4 世界。进入后停止旧单世界时钟热键。

HELLO 增加 `physics_backend=isolated_worlds` 和 `physics_world_count`；原有字段
保持兼容。Q 诊断 version=2 返回独立 sceneHandles、physicsWorldCount、epoch、
快照操作数与按 agent 汇总的刚体/控制器/关节状态。simulationCalls 统计实际
物理调用数：一次 N player STEP 增加 N，一次 TELEPORT 增加 1。

## 验收

在已部署当前 DLL 的隔离副本运行：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python src/tests/control_interaction/test_multiplayer_runtime.py --agents 1 2 8 --steps 1200 --out runs/multiplayer_acceptance_new
```

覆盖同动作跨世界、不同动作隔离、冷启动与两次 reset、100 步预热快照、投放
快照、投放隔离、静置不漂移、观测诊断、几何导出，以及 1/2/8 世界的共同初始化
边界和轨迹比较。观测按 float32 位模式比较，排除 timestamp；失败立即报错。
测试在自己的副本中导出碰撞几何，不应用于正在使用的生产实例。
这是指定动作与步数的验收，不声称任意地图位置或任意输入都已验证。

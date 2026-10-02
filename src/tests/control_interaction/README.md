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
- `test_native_dynamics.py` — 在独立游戏副本中采集原生、双路输入消融及修复模式的逐阶段动力学对照。

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

### 原生动力学对照

reset 对修复模式冷启动的复现性，不能证明它保留了原生游戏的动力学。
`test_native_dynamics.py` 对每组动作、模式、重复分别冷启动一个独立游戏进程，单 agent，
记录控制器调用前、调用后、物理更新后三个阶段。默认是零动作、恒定 `(40,30)`、
单步脉冲加反向脉冲，各 40 步、各模式重复 3 次，共 27 次启动。

- `native` 保留原生 Update/FixedUpdate/自动物理、Saviour 初始化、层级和动画，仅替换
  Rewired 鼠标轴及清零手柄轴。等待原生载入和输入暂停结束，再做 240 个零输入物理步。
- `native_dual` 保持上述路径，仅在探针控制器调用前追加写入 `mouseInput`，模拟现有
  RL 双路输入。这项消融用于区分已有输入接口差异与新 reset 修复的影响。
- `candidate` 使用已修复的固定启动配方和 TCP STEP；不额外添加上述原生预热。
  Y 仅启动记录，不 reset、不修改状态。两个起点若不同，不能把轨迹差异全部归因于物理算法。

探针默认关闭（`dynamicsProbeMode=""`）。原生记录器插入 Unity PlayerLoop 的
Physics2DFixedUpdate 之后；修复模式记录器位于显式 Simulate 和 SyncTransforms 之后。
记录七个刚体的位姿与速度、质量/惯量/重力/阻尼、控制器内部量、实际轴值和读取次数、
关节电机、接触点与冲量。原始 JSONL 中 bodyNames 标识刚体顺序；bodies 每体六列为
`x,y,rotation,vx,vy,angularVelocity`；motors 为 hinge 的速度/上限、slider 的速度/上限、
hinge 角度和 slider 平移量。只记录已显式列出的参数，不代表完整的隐藏求解器状态。

```powershell
& 'C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/MSBuild/Current/Bin/MSBuild.exe' src/GameRuntime_v2/GameRuntime_v2.csproj /t:Build /p:Configuration=DynamicsExperiment /p:DeployPlugin=false /nologo /v:minimal
Copy-Item bin/DynamicsExperiment/GameRuntime_v2.dll artifacts/reset_game/BepInEx/plugins/GameRuntime_v2.dll
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_native_dynamics.py --game-exe artifacts/reset_game/GettingOverIt.exe --out runs/native_dynamics_new
conda run -n getting-over-it-analysis python -m unittest discover -s src/tests/control_interaction -p test_native_dynamics_checks.py -v
```

`COMPLETE` 和退出 0 只表示采集完整、输入及计数校验通过，不表示动力学等价。
result.json 的 analysis 保留跨模式和模式内重启比较，起点不一致标为
`INITIAL_STATE_MISMATCH`，不会因模式内可复现而判等价。原生参考仍含 BepInEx 和记录钩子，
并非完全无插件的游戏。探针阻止 Saviour.SaveGameNow 写实验进度，但没有隔离其他
PlayerPrefs 修改。生产 DLL/配置哈希前后检查与游戏进程清理由脚本执行。

2026-10-02 在线结果在 `runs/native_dynamics_validation_20261002/result.json`，
结论与原始记录哈希在同目录 `validation_summary.json`。27 次启动、1080 个物理步、
3240 条阶段记录采集完整，**原生动力学保持验收未通过**：

- 所有已记录阶段的伸缩电机上限，native/native_dual 为 10000，candidate 为 4300。
  原生 `Saviour.Start()` 从 hinge.motor 构造 stillSliderMotor 并赋给 slider.motor；
  修复流程在 Start 前禁用 Saviour，漏掉这一有效参数初始化。上限降低 57%，不能解释为
  每一步实际施力都降低 57%。已记录的刚体配置和 1/120 秒物理步长一致。
- 同样输入 `(40,30)`，第一步平滑后的 mouseInput 原生为 `(48,36)`，双路输入和
  candidate 为 `(68,51)`。单步脉冲后的下一步，原生为 `(24,18)`，另外两者为零。
  双路写入在 reset 修复提交之前已经存在，不应全部归因于这次 reset 修改。
- 原生和 candidate 起点的最大坐标分量差约 0.0686m、角度差约 2.422°。
  原生自身的重复轨迹也有差异，尤其脉冲探针；这些结果不能用于把后续运动差异
  唯一归因于局部物理世界、刚体换父节点或禁用动画。尚未证明完整任务等价。

关闭记录器后的 `runs/dynamics_probe_disabled_reset_regression/result.json` 为 4/4 PASS，
冷启动恒定动作探针与上一版已验收 DLL 的结果逐位相同（排除 timestamp）。
新增 9 项离线检查、原有 18 项 reset 检查均通过。测试结束已恢复实验副本的原已验收 DLL，
生产 DLL 与配置哈希未变。本次只增加验证设施，没有修复上述新发现的差异。

### 补齐原生物理初始化

`PlayerControl/NativePhysicsInitialization.cs` 复刻 `Saviour.Start()` 的物理初始化段
（IL 000F–00C6）。仅在 controlled 初始化中执行，每次冷启动和场景 reset 都重新执行：

- 旋转电机读取原生 hinge.motor，并把初始速度置零。
- 伸缩电机同样从 **hinge.motor** 复制，遵循原生代码；不读取 prefab 上的 4300，
  也不把测到的 10000 写成新常量。
- 锅体连接关节按原生规则置零速、上限 100，光标先放到锤头位置。
- 上述操作发生在原始 PlayerControl.Start 和克隆之前。每个角色的 Start 完成后，
  校验三个关节及两个控制器电机缓存，避免下一次 FixedUpdate 写回旧值。

这一改动补齐了被禁用的 Saviour 中的物理初始化，未重新启用存档载入和自动时钟。
本次范围不包含输入平滑、起点姿态或完整游戏任务等价性。

`test_native_dynamics.py --require-native-parameters` 用实测 native 参考检查所有阶段的
三个电机上限、两个控制器缓存、刚体配置和 fixedDeltaTime。参数不符退出 1；即使参数
通过，完整动力学等价性仍显示 NOT_CERTIFIED。新增 JSONL 字段 motorParameters 的顺序为
`hinge_cap, slider_cap, hub_cap, controller_hinge_cap, controller_slider_cap`。

多 agent reset 可追加 `--native-parameter-reference <上述通过的result.json>`，要求
`--reset-diagnostics`。它在 reset 后、等待后、动作探针结束后，逐个 agent 检查三个关节
上限；缺失关节、复制体仍为旧值或动作后回退均不能通过。在线运行仍仅使用实验副本。

参数补齐后的验收使用同一 DLL，SHA-256：
`088d3f6ca23bdaacef0353489594b275e1baafa8a23ec4ff50884eb5bacc41d4`。

- `runs/native_parameters_fixed_20261002/result.json`：原生/修复模式各 3 次启动，
  零动作、恒定动作、脉冲各 40 步；18 次启动、2160 条阶段记录的参数检查通过。
  三个电机上限为 10000/10000/100，两个控制器缓存均为 10000。
- `runs/native_parameters_reset_2agents_20261002/result.json`：3 次启动、27 个条件通过，
  含 40/200/1000 步历史和 0/0.25 秒等待；40 步恒定动作探针共 1107 帧逐位匹配。
- `runs/native_parameters_reset_8agents_20261002/result.json`：4 个条件通过，1200 步
  随机动作探针共 4804 帧逐位匹配。所有 agent 在 reset、等待、探针后参数均正确。

逐位比较按对应 agent、排除 timestamp；31 个条件合计 5911 帧。2-agent reset 为
0.59–0.64 秒，8-agent 为 0.83–0.85 秒。12 项动力学判定器和 19 项 reset 判定器测试通过。
汇总与生产文件哈希位于 `runs/native_parameters_fixed_20261002/acceptance_summary.json`。
新 DLL 保留在实验副本中，生产安装未部署。输入平滑与原生起点差异仍未处理，不能将
这些参数与复现性验收结果解释为完整的原生动力学等价证明。

### 起点姿态差异专项测试

`analyze_start_pose.py` 读取零输入原生/修复模式对照，分别比较世界坐标和以 Player
刚体为原点、方向基准的坐标。后者消除整体平移、旋转，检查关节构型是否仍有差异。
它同时要求电机等原生参数检查通过，且整个记录中的 mouseInput/oldMouse 均为零，
避免把输入平滑残留当成姿态差异。位置是 Rigidbody2D 参考点，不是锅的几何中心。

```powershell
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_native_dynamics.py --game-exe artifacts/reset_game/GettingOverIt.exe --out runs/start_pose_zero_new --repeats 3 --actions zero --modes native candidate --native-warmup 0 --steps 1200 --require-native-parameters
conda run -n getting-over-it-analysis python src/tests/control_interaction/analyze_start_pose.py --run runs/start_pose_zero_new
conda run -n getting-over-it-analysis python -m unittest discover -s src/tests/control_interaction -p test_start_pose_checks.py -v
```

2026-10-02 使用参数修复后的同一 DLL（`088d3f6c…`），6 次独立启动、每次 1200 步，
合计 7200 个物理步、21600 条阶段记录。数据在 `runs/start_pose_zero_1200_20261002`，
几何分析在其中的 `start_pose_analysis.json`。各模式从自己的就绪边界开始计步：原生
不另加预热，修复模式已经完成固定姿态初始化和 120 步规范化，二者初始化配方不同。

- 起点 Player 位置相差约 5.26 厘米，角度原生约 0.0463°、修复模式约 2.7575°。
  七个刚体中最大世界位置距离差约 6.94 厘米（最大坐标分量差 6.90 厘米）。
- 锤头到 Hub 的距离原生约 1.8983m、修复模式约 1.9999m；伸缩关节平移量相差
  10.16 厘米。消除 Player 整体平移和旋转后，锤头仍相差约 11.02 厘米，构型确实不同。
- 再执行 1200 步零输入（约 10 秒模拟时间）后，Player 角度差仍约 2.711°，伸缩量
  差约 10.206–10.208 厘米，对齐后的锤头差约 10.691–10.693 厘米；这段额外静置
  没有消除差异。不能由这次有限时长实验推断无限时间后的行为。
- 三次修复模式的全部已记录刚体位姿/速度、控制器量和电机量一致。原生自身有微小
  偏差：全程最大坐标分量差约 0.055 毫米、角度差约 0.00223°，远小于跨模式差异。
  本次不继续排查原生自身偏离。

代码中固定模板来自 `Saviour.ResetPlayerButNotDialogue`（原生按 R 复位），对照则是
NewGame 启动；规范化期间还每步强制把光标移到锤头。模板和初始化流程均不同于对照，
但本实验尚未区分模板、动画、层级及调度各自的贡献，不能把全部误差唯一归因于模板。
这次只新增测试/分析，没有修正起点姿态或修改生产 DLL/配置。3 项几何判定器测试通过。

### 使用 NewGame 场景起点（实验开关）

`experimentalNativeNewGame=true` 要求同时开启 `experimentalSceneReset` 和
`experimentalControlledUpdates`；默认关闭。首次启动和每次场景 reset 都执行同一流程：

- 保留场景资源里的刚体初始位姿，使用已补齐的原生物理参数。
- 复制体的独立光标从原始光标 Rigidbody2D 的物理位姿初始化，避免复制尚未同步的
  Transform 而保留旧场景坐标；只在此实验模式启用。
- 等待各角色原生 Start 完成，调用原生 `PlayerControl.PlayOpeningAnimation()` 入口，
  由原生代码安装开场输入暂停；自动动画、自动时钟仍保持关闭。
- 在原始刚体层级下，以零动作手动调用原生控制器和物理，直到输入暂停计时结束。
  当前游戏版本为 721 步，约 6.008 秒模拟时间；代码按就绪条件停止，不硬编码 721。
- 到达就绪边界后再把动态刚体挂到静态父节点并禁用插值，防止等待期间的渲染同步改状态。
  随后保存起点；不再应用 R 复位模板、120 步规范化或额外的强制光标吸附。

这项开关取代的是起点生成规则，`normalizeInitialPose/initPoseSettleFrames` 在此模式不生效。
已有场景重建、TCP 接口和按对应 agent 比较的 reset 合同不变；仍不支持任意快照 N。
输入平滑接口和视觉效果没有在本项修改中恢复为原生，不能据此宣称完整游戏等价。

```powershell
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_native_dynamics.py --game-exe artifacts/reset_game/GettingOverIt.exe --out runs/native_start_new --repeats 3 --actions zero --modes native candidate --native-warmup 0 --steps 1200 --native-new-game --require-native-start --require-native-parameters
conda run -n getting-over-it-analysis python src/tests/control_interaction/analyze_start_pose.py --run runs/native_start_new
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_reset_state.py --game-exe artifacts/reset_game/GettingOverIt.exe --out runs/native_start_reset_new --native-new-game --scene-reset --reset-diagnostics --agents 2 --probe-steps 40 --probe-kind hold --native-parameter-reference runs/native_start_new/result.json --native-start-reference runs/native_start_new/result.json
```

`--require-native-start` 要求原生端 `--native-warmup 0`，按动作和重复编号配对原生参考。
检查所有七个刚体的起点位姿/速度及已记录控制器量：位置分量容差 1mm、角度 0.05°、
速度分量 1mm/s、角速度 0.1°/s；输入、平滑历史、增益、暂停计时和设置必须相同。
控制器缓存 mw 的坐标容差 1mm，oldAngle 容差 0.05°。没有提高原有刚体判定阈值。
缺少配对参考或超限即退出 1；通过仍区分逐位一致与容差内一致，完整动力学仍 NOT_CERTIFIED。

reset 测试的 `--native-start-reference` 逐个复制体检查复位后和等待后的七刚体状态，
包含各自的光标，使用同一组容差。这项检查与“相对自身冷启动可重复”独立；如果某个
复制体稳定地从错误的位置出发，仍会失败。定位阶段 `native_start_reset_*_20261002`
虽通过自身复现性检查，仍有复制体光标初值偏差，不属于最终原生起点验收证据。

最终 DLL SHA-256 为
`e6c6429fde0d429d0d0d99e02e701d136df3cb3ec725e3d50e9c54c44533acaf`。
`runs/native_start_final_20261002/result.json` 包含原生/修复模式各 3 次独立启动、
每次 1200 步零输入：起点状态和全程原生参数检查均通过，21600 条阶段记录完整。
起点最大世界位置距离差 0.5314mm（坐标分量差 0.5055mm）、最大角度差 0.01521°；
对齐 Player 后锤头距离差不超过 0.5681mm，伸缩量差不超过 0.2023mm。
原来约 10 厘米的构型差异已消除，但这仍是容差内匹配，不是与原生逐位相同。
全部零输入轨迹阶段的刚体误差均在原阈值内；修复模式自身的三次轨迹逐位一致。
37 项离线检查通过，包含“复制体光标稳定偏移也必须判失败”的回归用例。

同一最终 DLL 的 reset 验收：

- `runs/native_start_final_reset_2agents_20261002/result.json`：3 次启动、27/27 条件通过，
  包含 40/200/1000 步历史及 0/0.25 秒等待；1107/1107 帧逐位匹配对应冷启动参考。
- `runs/native_start_final_reset_8agents_20261002/result.json`：4/4 条件通过，包含
  1200 步随机历史和 1200 步随机探针；4804/4804 帧逐位匹配对应冷启动参考。
- 所有复制体在复位后、等待后都通过原生七刚体起点检查，并在探针后通过原生电机
  上限检查。逐位比较始终排除 timestamp，比较同一个 agent，不要求不同 agent 相同。
- reset 耗时 2 agents 为 0.72–0.78 秒、8 agents 为 1.23–1.27 秒，比原来的 120 步
  固定模板初始化更慢。开关保持默认关闭，最终 DLL 只放入实验副本，生产文件哈希未变。

汇总：`runs/native_start_final_20261002/acceptance_summary.json`，含原生起点误差、
全部复制体检查、31 条件合计 5911 帧逐位比较、DLL 与生产文件指纹。

### 原生输入平滑（实验开关）

`experimentalNativeInputSmoothing=true` 默认关闭，要求受控更新与 `stepFrames=1`。
动作只写入 Rewired 鼠标轴，原生 `PlayerControl.FixedUpdate()` 保留每个 agent 自己的
`mouseInput` 历史，执行灵敏度、原生曲线、Lerp 0.5、暂停处理和自适应增益。
零动作自然衰减，快照内部态补存 `mouseInput`，复制体输入接口也不再覆盖它。
同时屏蔽四个手柄 Stick 轴，保证鼠标动作不会被手柄替换；未增加 Python 平滑器。
复制体在原生 Start 完成后、开场模拟前，从原始角色同步灵敏度、触控板设置和两条响应
曲线（曲线另建实例，历史不复制）。Unity 克隆遗漏了运行中设置的私有灵敏度：在本机
原始角色为 2.4，而复制体原先仍为构造默认值 1；单人测试无法发现这项差异。

`test_native_dynamics.py --require-native-input` 检查原生与候选每一步的控制前、控制后、
物理后阶段：mouseInput、oldMouse、增益、暂停计时、灵敏度和触控板设置必须逐位一致。
必须覆盖 hold、pulse、reverse、random，缺少配对原生参考或任何一项不一致均失败。
`native_dual` 保留为只增加 raw 字段写入的负对照。原生与候选的位置/电机目标差异单独
报告；输入验收不代表光标目标或整段物理轨迹等价。

```powershell
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_native_dynamics.py --game-exe artifacts/reset_game/GettingOverIt.exe --out runs/native_input_new --repeats 3 --actions zero hold pulse reverse random --modes native native_dual candidate --native-warmup 0 --steps 40 --native-new-game --native-input-smoothing --require-native-input --require-native-start --require-native-parameters
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_native_input_agents.py --game-exe artifacts/reset_game/GettingOverIt.exe --reference runs/native_input_new/result.json --out runs/native_input_agents_new --agents 8 --burn-steps 1000
conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_reset_state.py --game-exe artifacts/reset_game/GettingOverIt.exe --out runs/native_input_reset_new --native-new-game --native-input-smoothing --scene-reset --reset-diagnostics --agents 2 --probe-steps 40 --probe-kind hold --native-parameter-reference runs/native_input_new/result.json --native-start-reference runs/native_input_new/result.json
```

`test_native_input_agents.py` 关闭 DynamicsProbe，在同一进程中给各 agent 分配不同
动作序列，逐步把 live Q 诊断中的输入、历史、增益对照单人原生记录。随机搅动后 reset，
再轮换各 agent 的动作序列，检查历史串用及 reset 残留。要求使用与原生对照同一 DLL。
reset 验收另外比较复位后与探针后的所有控制器内部态，避免只检查刚体而遗漏输入残留。

动作/观测维度未变，但开启此开关会改变已有策略的动作响应，需要重新评估；实验仍不
覆盖生产 DLL 或生产配置，任意快照 N 仍不属于场景 reset 的支持范围。

2026-10-02 最终验收 DLL SHA-256：
`f5ce41e1ac158e517d655d29d3c98cda85848f7d806f4132795021e33d55692b`。

- `runs/native_input_final_20261002/result.json`：5 组动作，原生/候选各独立启动一次，
  每次 40 步。候选的全部 600 条阶段记录在上述输入检查量上逐位匹配原生，最大差为 0；
  原生起点和参数检查通过。恒定输入前 3 步为 `(48,36) → (72,54) → (84,63)`；
  单步脉冲后为 `(48,36) → (24,18) → (12,9)`。
- `runs/native_input_final_agents_8_20261002/result.json`：8 agents、两轮不同动作分配，
  其中第二轮在 1000 步随机历史后 reset；640/640 个 agent 步逐位匹配各自原生输入参考。
  此项关闭 DynamicsProbe，直接检查实际运行时的 Q 诊断。
- `runs/native_input_final_reset_2agents_20261002/result.json`：3 次冷启动、27/27 条件，
  包含 40/200/1000 步历史和 0/0.25 秒等待；1107/1107 帧逐位匹配对应冷启动参考。
- `runs/native_input_final_reset_8agents_20261002/result.json`：4/4 条件，包含 1200 步随机
  历史和 1200 步随机探针；4804/4804 帧逐位匹配。两组 reset 的复位后/探针后内部态、
  所有复制体原生起点和电机参数检查均通过。逐位帧比较排除 timestamp。
- 编译与 40 项离线检查通过。最终 DLL 只部署到实验副本；生产 DLL/配置哈希未变。
  总结见 `runs/native_input_final_20261002/acceptance_summary.json`。

定位数据保留在 `runs/native_input_smoothing_20261002`（45 次启动，含旧双路负对照）
和 `runs/native_input_agents_8_20261002`（复制体灵敏度遗漏导致失败）；后者不是验收证据。
旧双路的恒定响应是 `(68,51)`，松开即归零，说明本次修复恢复了跨步历史。
完整物理轨迹/光标目标的原生等价性仍未认证，不由输入处理一致性推断。

# Getting Over It — 把封闭商业游戏改造成可编程 RL 环境

把一款没有任何官方接口的 Steam 商业游戏，改造成具备**帧级确定性步进、快照重置、多 agent 并行、可观测内部物理状态**的强化学习环境，并在其上搭建完整的训练、验证与实验记录体系。

**技术栈**：C# · BepInEx · Harmony · Unity · Python · PyTorch · TCP 二进制协议

> **当前状态**：环境侧完整可用并通过 L1–L9 分层验证；训练侧尚未突破第一段攀爬，瓶颈已定位为硬探索死锁（见第七节「已知结论」）。本项目的主体成果是**环境与验证基础设施**，训练结果是其上的一次未完成实验。

---

## 一、问题

《Getting Over It with Bennett Foddy》是一款以挫败感著称的攀爬游戏：玩家用鼠标操纵一把锤子把角色拖上山，没有检查点，摔落直接回到起点。

把它作为 RL 环境有三重困难：

- **无离散动作**：控制量是鼠标的连续 (dx, dy) 位移，策略空间庞大
- **奖励极稀疏**：唯一明确信号是"登顶"，中途没有子目标
- **无任何现成接口**：闭源 Steam 商业软件，无官方 API，无第三方 Gym wrapper

OpenAI Gym 与 Unity ML-Agents 在这里全部失效——前者需要环境主动实现接口，后者需要在工程构建期集成。**唯一入口是从 Unity 运行时下手**：注入插件、读取内部物理状态、劫持输入管线、接管物理时钟。

本项目做的就是这件事。

---

## 二、架构

```
┌──────────────── Python 进程（训练侧 / TCP 客户端） ─────────────────┐
│  main.py · main_train.py（BC） · main_ppo.py（PPO）                │
│     ├── GameModeController  ──写信号文件──►  GoiData/ControlSignals/ │
│     ├── GameLauncher        ──subprocess──►  GettingOverIt.exe      │
│     └── GoiEnv (TCP client) ──socket──────►  127.0.0.1:9000         │
└────────────────────────────────────────────────────────────────────┘
                          │  TCP（小端二进制，请求-响应）
                          ▼
┌────── 游戏进程内 BepInEx 插件 GameRuntime_v2（TCP 服务端） ──────────┐
│  GameRuntimeManager (BaseUnityPlugin)                              │
│     ├── ModeManager         信号文件 → 决定运行模式                  │
│     ├── TcpStepServer       后台线程收发 + 主线程协程处理             │
│     ├── StepController      Physics2D.Simulate 帧级推进 / 快照 / 传送 │
│     ├── PlayerInputService  反射注入 + Rewired GetAxis 拦截          │
│     └── PlayerStateService  反射采集 29D 部件状态                    │
└────────────────────────────────────────────────────────────────────┘
```

一次 RL step = 一次阻塞式 TCP 请求-响应，C# 端在回包前完成物理推进，保证时序严格对齐。

插件支持三种互斥模式：`DataCollection`（导出地图碰撞几何后退出）、`GameRuntime`（训练交互，启动 TCP 服务端）、`GameTesting`（物理能力探测）。

---

## 三、关键技术决策

让这个环境成立的，是下面这些具体决定。每一条都对应一个真实存在的约束——不这么做，环境就不成立。

### 3.1 接管物理时钟

`StepController.Initialize()` 把 `Physics2D.simulationMode` **持久切换为 `Script`**，游戏物理不再自动推进，完全由 `ExecuteStep` 驱动。

这是帧级确定性的前提——没有它，Unity 会在两次 TCP 往返之间自行推进任意帧数，采样到的 (s, a, s') 无法对齐。

### 3.2 拆掉游戏自带的落水重置

落水会触发游戏的 `Saviour` 组件，而它会把 `Physics2D.simulationMode` **改回 `FixedUpdate`**，直接摧毁上一条的成果。

因此训练模式初始化时 `DisableWaterReset()` 销毁场景内所有 `RestartOnContact` 组件，终止判定完全交给 Python 侧单一入口 `reward.is_water`。

### 3.3 输入劫持走双路

动作是 2D 鼠标相对位移 `(dx, dy)`，注入采用两条路径：

1. **反射写字段**：`PlayerControl.mouseInput`（NonPublic Instance）直接写入
2. **Rewired 层拦截**：Harmony patch 掉 `GetAxis("mouseX"/"mouseY")`，RL 模式下返回注入值、屏蔽真实鼠标

同时 Harmony prefix patch `PlayerControl.Update()` 使其在 RL 模式下整体跳过，防止游戏从真实鼠标读值覆盖注入。

关键细节：只有在**手动 `InvokeFixedUpdate()` 上下文**里（`CurrentAgentIndex >= 0`）拦截才返回注入值；Unity 自动触发的 FixedUpdate 对鼠标轴返回 0，成为无操作。这保证物理推进只由 STEP 驱动，且训练期间人手抖动不会污染轨迹。

### 3.4 `InvokeFixedUpdate` 必须在 `Simulate` 之前

顺序颠倒会导致关节电机力缺失，复制体被 Box2D 求解器弹射到 (0,0)。这是踩出来的顺序约束，不是设计偏好。

### 3.5 TCP 双线程 + 事件握手

Socket IO 与 Unity 物理必须分离（Unity API 只能在主线程调用）：

- **后台线程**：阻塞 `AcceptTcpClient` → 读命令 → 写 `_pendingCommand` → `_commandReady.Set()` → 阻塞等 `_responseReady` → 回写 socket
- **主线程协程**：每帧非阻塞 `TryGetCommand()`，命中则执行物理操作并 `SendResponse()`

两方向用 `ManualResetEvent` + `volatile` 握手，**无锁**。异常时发送零状态空回包，避免 Python 永久阻塞。

Python 侧连接时先用 5s 超时完成握手，成功后**切换为阻塞模式（无超时）**——否则 C# 协程的处理延迟会触发 `TimeoutError`。

### 3.6 快照与重置

游戏本身没有存档点。`NEW_SNAPSHOT` 以当前物理状态重拍基准，此后每次 `RESET` 回到该时刻，使 agent 可从固定起点反复尝试。

`Reset()` 除恢复所有刚体外，还经 `PlayerInputService.SetInternalState` 恢复 `mouseVelocityAverage` / `oldMouse` / `mw` / `oldAngle` / `pauseInputTimer` / `mouseSnap` / `inputsToSkip` 等**控制器内部态**，消除跨 reset 的增益残留；`CaptureAllSnapshots` 经 `BroadcastCanonicalSnapshot` 把 agent0 的权威态广播到所有复制体。

### 3.7 多 agent 并行

单进程内复制体方案：agent 0 是原始 `Player`，agent 1..N-1 由 `PlayerDuplicateManager` 复制，各自持有独立的 Input/State 服务与 `AgentIndex`。

一次 `STEP` 携带 N 组动作，游戏内同时推进，回包按 agent 顺序交错。训练默认 10 agent 并行，采样吞吐提升约 10×。

---

## 四、观测与协议

### 4.1 命令集

| 命令 | 字节格式 | 含义 |
|---|---|---|
| `RESET` | `['R']` | 重置所有 agent 到基准快照 |
| `STEP` | `['S'][n:1B][actions: n×2×4B]` | 注入动作并推进 `stepFrames` 帧 |
| `NEW_SNAPSHOT` | `['N']` | 以当前状态重拍快照 |
| `CONFIG` | `['C'][active:1B][mouseXId:4B][mouseYId:4B]` | 配置 Rewired 鼠标拦截 |
| `VISUALIZE` | `['V'][enabled:1B]` | 碰撞箱描边可视化开关 |
| `EXPORT_COLLIDERS` | `['E']` | 导出地图碰撞几何 |
| `TELEPORT` | `['T'][agentIndex:1B][x:4B][y:4B]` | 传送 agent（空投部署） |
| `CAMERA_FREE` | `['F'][enabled:1B]` | 自由相机开关 |
| `CLOSE` | `['X']` | 关闭训练循环 |

回包：`[n:1B]` + 逐 agent `[state: 33×4B][done: 1B]`。**state 与 done 按 agent 交错**，解析时必须逐 agent 读取。所有 float 显式按小端写出。

### 4.2 状态维度

**线路协议 33 维**（`StepController.STATE_DIM`，唯一权威）：

- **0–28**：`PlayerState.ToFloatArray()` 的 29 维——player / hub / slider / handle / pole / tip 各部件的位置、速度、角度，加派生的 `hammerAngle = atan2(tip - hub)` 与 timestamp
- **29–32**：`fakeCursorRB` 的绝对坐标与速度

第二段值得单独说明：`fakeCursorRB` 是游戏内不可见的鼠标锚点，是锤子控制链的隐藏状态。不观测它，策略就看不到自己上一步施加的控制究竟落在哪里——这是**补可观测性**，不是喂答案。

**模型消费 39 维**（`DYNAMICS_DIM`）：

`dataset.build_dynamics()` 在 Python 侧把绝对量转为**平移等变**特征（速度原样、角度 → sin/cos、部件坐标取相对 player 的差），得到 34 维，再叠加 5 维接触信号：

| 维度 | 字段 | 游戏原生对应（IL 反编译） |
|---|---|---|
| 34 | `tip_contact` | `HammerCollisions.collisionPoints` 含 Terrain entry |
| 35 | `tip_grip` | `HammerCollisions.slide == false`（静摩擦/钩住） |
| 36 | `body_contact` | `PlayerSounds.OnCollisionEnter2D` 中 `rb.GetPoint(contact).y > 0.4` 分支 |
| 37 | `pot_contact` | `PlayerSounds.isTouching`（pot 稳态支撑） |
| 38 | `fall_distance_norm` | `PlayerSounds.Update` 的 `CircleCast(pot, 0.5, ↓)` |

**这 5 维不是拍脑袋设计的启发式，是反编译游戏 IL 后在 Python 端复刻其原生接触判断**。阈值常量直接取自 IL：`CONTACT_EPSILON = 0.03`（`HammerCollisions.moveThreshold`）、`GRIP_VEL_THRESHOLD = 0.3`（`OnCollisionStay` 硬编码）、`FALL_CAST_RADIUS = 0.5`（`CircleCast` 半径）。

`body_contact` 与 `pot_contact` 之所以分开而不合并：游戏用 `rb.GetPoint(contacts[0].point).y > 0.4` 区分"接触点打在躯干上部"（撞头受伤，触发 `hurtThreshold=8` 的 pain sound）与"pot 底部触地"（稳定支点）。**两者物理语义完全相反**，合并会让信号互相抵消。

实现放在 Python 端而非 C# 端，是因为游戏 IL 里的接触判断本质就是"多边形相交"+"向下 CircleCast"，Python 可精确复刻；改 C# 需重编译插件、重启游戏、并破坏 wire 协议兼容性。**线路协议因此保持 33 维不变**。

---

## 五、验证体系

环境的价值取决于它是否可信。本项目的验证分四类，**区别在于要不要拉起游戏进程**——这决定了它能否进入批量运行。

### 5.1 离线单元测试（不碰游戏，秒级）

```bash
python -m src.tests.run_unit_tests            # 全部
python -m src.tests.run_unit_tests -k patch   # 按名字过滤
```

| 脚本 | 覆盖内容 |
|---|---|
| `test_contact_geometry_smoke.py` | 点—线段距离、多边形最近距离、向下射线；5 维接触信号提取（含 body / pot 分层各自触发） |
| `test_dynamics_dim_smoke.py` | `DYNAMICS_DIM=39` 维度算术与 `build_dynamics` 在有/无接触信号下的行为 |
| `test_p1_ou_map_elites.py` | OU 噪声的条件高斯与 `log_prob` 自洽、AR(1) lag-1 自相关；MAP-Elites 准入与冷启动 |
| `test_p2_contact_integration.py` | 老 34D checkpoint 扩维加载、前向、向后兼容 |
| `test_patch_tip_coverage.py` | 锤子 360° 全伸时 patch 窗口必须装得下锤头轮廓且不被边界截断 |
| `test_raster_area_coverage.py` | 面积覆盖率光栅化的数值正确性，亚像素轮廓不被漏采成空通道 |

环境未安装 pytest，每个测试脚本自带 `main()` 可独立自跑，运行器只负责逐个拉起子进程并汇总。

### 5.2 在环集成测试：L1–L9 能力阶梯

按能力分层递进，全部需要游戏进程。每一层验证的是**上一层成立之后才有意义的性质**。

| 层 | 脚本 | 验证目标 |
|---|---|---|
| L1–L2 | `test_l1_l2_protocol_causality.py` | TCP 连通性与输入—状态因果性 |
| L3 | `test_l3_repeatability.py` | 同输入序列的可复现性 |
| L4 | `test_l4_dose_response.py` | 输入幅值与响应的剂量关系 |
| L5 | `test_l5_agent_isolation.py`<br>`test_l5_1_parallel_input.py` | 多 agent 隔离与并行输入互不串扰 |
| L6 | `test_l6_collider_visual.py` | 碰撞体收集结果与游戏内实际几何对齐 |
| L7 | `test_l7_surface_airdrop.py` | 可着陆表面提取与空投候选点生成 |
| L8 | `test_l8_airdrop_deploy.py` | 批量投放与 settle 落差统计 |
| L9 | `test_l9_model_eval.py`<br>`test_l9_water_reset.py` | 无噪声推理评估；落水复位行为 |

另有 `verify_body_reconstruction.py` / `verify_tip_reconstruction.py` 验证从 33D 状态重建部件轮廓的正确性——这是接触信号能成立的几何前提。

### 5.3 离线分析与基准

产出量化结论与图表，写入 `src/Data/analysis/`：

`analyze_effectiveness.py`（效率图与可达性）、`check_tip_patch_coverage.py`（按锤子几何硬上界校验 patch 半径）、`diagnose_tip_rasterization.py`（扫描朝向与亚像素偏移，量化轮廓漏采率）、`bench_patch_raster.py`（光栅化性能基准）、`verify_mask_aliasing.py`（地形 mask 栅格混叠）、`viz_exploration_noise.py`（探索噪声时序结构）。

### 5.4 历史标定（已固化）

`input_characteristics/` 下是游戏输入精度与有效范围的标定，结论已固化为常量：**精度 1.0、有效范围 [10, 100]、饱和点 100（硬限制）**。归档性质，一般不再运行。

---

## 六、实验记录体系

每组训练实验在 `runs/<tag>/` 下留一份 `RESULT.md`，固定记录**意图 / 状态 / 时间 / 完整命令 / 客观结果 / 结论**。目的是让负面结果也可复用——多数实验的价值在于排除一个假设，而不是证实一个假设。

例（`runs/start_sil/RESULT.md`，节选）：

> **意图**：固定起点 + SIL 精英池（190 轮）
> **客观结果**：mean_reward 末轮 -0.5411，峰值 1.051
> **结论**：跑满 190 轮，效率图价值分布仍只覆盖起点盆地，与 `start_long`（同配方无 SIL，60 轮）相比没有质变。SIL 精英池在上游探索从未产出正样本的前提下，只能反复强化"起点乱晃"——这是把 SIL 归入"探索的下游机制"的直接证据。

---

## 七、已知结论

以下都是本项目跑出来的、可复现的结论，其中相当一部分是负面结果。

### 7.1 环境确定性存在 ~1e-4 的地板

Unity `Physics2D`（Box2D）求解器的内部态——接触与关节的 warm-start 冲量——**无法经刚体快照恢复**。实测：

| 场景 | 跨 reset 复现误差 |
|---|---|
| 静止启动姿态 + 零动作 | ≈ 2e-7（近机器精度） |
| 摆动姿态 | 放大至 ~0.1 |
| 激烈动作 | 由确定性混沌进一步放大 |

**推论**：「把游戏当作逐位精确的世界模型」不可达；基于游戏本身的规划只能作**近似短时域 MPC**。详见 `doc/planb_premise_verification.md`。

### 7.2 训练瓶颈：硬探索死锁

**证据**：精英轨迹 top-10 全部在起点附近横向游走，峰值 y ≈ 0~1，**没有任何一条翻越第一个障碍**（Deadtree, x ≈ -30）；效率图跑到 190–260 轮 PPO 后价值仍只在起点盆地。

**直接死因**：`stepFrames=1` 即 50Hz 控制，PPO 逐帧 iid 高斯采样；翻越第一个台阶需要跨约 25 帧的**时序连贯"勾—拉"动作**，50 次独立高斯采样几乎不可能构造出来。

**系统性死因**：塑形（效率图 PBRS）、模仿（SIL）、评分都位于**探索的下游**。探索从未撞对第一段真攀爬 → 上游无正样本 → 所有下游机制退化为强化"起点乱晃"。这是稀疏奖励 + 硬探索的经典死锁。

### 7.3 地图存在几何断裂

`src/Data/analysis/effectiveness_report.md` 的离线可达性分析（A–E 五项）中，E 项结论为**不足**：

即便使用 oracle（到顶测地距离场）作为势场，reach=16m 的贪心攀爬爬升高度占比中位数也只有 **6.0%**，到顶率 **0%**；内建 `height_prior` 为 14.7% / 0%。扫描到 reach=32m 仍无法稳定到顶。

> ⚠️ 该分析的方法学局限已在报告中标注：台面图只建模「可着陆平台」之间的跳跃可达性，不含游戏中靠锤子摩擦攀爬垂直墙面的能力，因此对可达性**偏保守**。结论应理解为「靠平台跳跃的贪心策略」的下界，而非物理上限。

### 7.4 动作流形是低频的

频域分析显示，能完成攀爬的目标动作住在**低频薄流形**上：4Hz 以下占约 100% 功率。

对比探索噪声的时序结构：iid 高斯 lag-1 自相关 ≈ -0.02，功率平摊到 25Hz Nyquist，与目标结构几乎不重叠；pink 噪声 lag-1 ≈ 0.79、谱斜率 ~1/f，与目标基本重叠。

这是把探索噪声从 iid 高斯改为 OU / AR(1) 的直接依据（见 `src/training/config.py` 中 OU 段注释与 `doc/optimization_roadmap.md` P1.1），也提示 50Hz 的逐帧决策频率极大浪费了搜索预算。

> 注：该频谱结论的原始分析脚本（`ideal_action.py`）当前不在仓库内，`viz_exploration_noise.py` 只覆盖噪声侧的时序结构可视化。公开前应把目标动作侧的频谱分析补进 `src/tests/analysis/`，使结论可从仓库复现。

---

## 八、训练侧设计原则

所有改进项都经过同一条准则筛选：

> **缩搜索空间 ≠ 缩解空间。**

| | 做法 | 举例 |
|---|---|---|
| ✅ 缩搜索空间 | 让同样的解更容易被找到 | 加接触信号；连贯探索先验；相位加权损失；行为多样性池 |
| ❌ 缩解空间 | 直接规定解长什么样 | 硬编码"接触时沿法向施力"；喂人类演示做 MSE 模仿 |

判断方法：

1. 这个改动是让策略更容易找到好解，还是替策略决定好解？
2. 如果最优策略跟我猜的不一样，这个改动会不会阻止它出现？

据此**明确拒绝**了喂人类演示做模仿学习——即便它大概率能更快出结果。完整的优化项、依赖关系、已否决方案及其理由见 `doc/optimization_roadmap.md`。

已实施项中，两处值得单独提：

- **OU 探索噪声**：为在时序相关噪声下保持 PPO 重要性采样比无偏，buffer 中额外存 `noise_prev`，新旧策略对同一 `z_{t-1}` 计算条件高斯 `raw_t | z_{t-1} ~ N(μ + σφz_{t-1}, σ²(1-φ²))`，`ratio = exp(new_lp - old_lp)` 无系统偏差。`ou_phi=0` 完全退化为原 iid 行为。
- **MAP-Elites 精英池**：原 top-K% 是**相对**门槛，整批都烂时仍会收入"最不烂的乱晃"并自我强化。改为按行为描述子（secured peak 的 x, y）离散化到网格、每格保留格内最优，天然回避阈值调参。

---

## 九、快速开始

### 前置

- 《Getting Over It》（Steam 版）已安装
- 游戏目录中已安装 **BepInEx**（`BepInEx\core\BepInEx.dll` 存在）
- **Visual Studio 2019 Build Tools**（含 MSBuild）
- **Python 3.9+**

### 配置

游戏路径需在两处保持一致：

1. `src/GameRuntime_v2/GameRuntime_v2.csproj` 的 `<GameRoot>` 属性
2. `src/config/project.json` 的 `executable_path`

```json
{
  "game": {
    "executable_path": "C:\\...\\Getting Over It\\GettingOverIt.exe",
    "max_execution_time": 300
  },
  "tcp_port": 9000
}
```

```powershell
pip install -r requirements.txt
```

### 编译插件

```powershell
$msbuild = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
& $msbuild src\GameRuntime_v2\GameRuntime_v2.csproj /p:Configuration=Release
```

产物 `GameRuntime_v2.dll` 输出到 `bin\Release\`，PostBuild 自动复制到游戏的 `BepInEx\plugins`。

### 采集地图几何

```powershell
cd src
python main.py
```

流程：设为数据采集模式 → 启动游戏 → 自动导出碰撞体 → 退出。

### 训练

从**仓库根目录**运行，脚本会自动完成启动游戏、TCP 连接、采集、更新、关闭：

```powershell
# 迭代行为克隆
python -m src.training.main_train --num-agents 10 --max-iterations 50

# PPO
python -m src.training.main_ppo --num-agents 10 --steps-per-rollout 500 --max-iterations 200
python -m src.training.main_ppo --resume checkpoints/ppo_iter_0050.pt
```

完整 CLI 与推荐配方见 `doc/main_ppo_usage.md`。

---

## 十、目录结构

```
src/
├── GameRuntime_v2/          # C# BepInEx 插件（游戏进程内）
│   ├── Core/                #   插件入口、模式管理、帧级步进、事件总线、反射工具
│   ├── Communication/       #   TCP 双线程步进服务端
│   ├── PlayerControl/       #   反射注入 + Rewired 拦截、状态采集、复制体管理
│   ├── ColliderCollection/  #   碰撞几何导出与可视化
│   └── Camera/              #   自由相机
├── env/goi_env.py           # Python TCP 客户端（唯一环境封装）
├── training/                # 训练框架：模型、PPO、buffer、rollout、奖励、数据集
├── tests/                   # 四类验证：离线单测 / 离线分析 / 在环 L 系列 / 历史标定
├── start/                   # 进程启动与模式信号
└── Data/                    # 采集数据、参考轨迹、分析图表

doc/                         # entrypoints / training / optimization_roadmap /
                             # hammer_physics / planb_premise_verification
runs/<tag>/RESULT.md         # 每组实验的意图、命令、客观结果、结论
checkpoints/                 # 模型、地图几何、候选投放点、人工排除区
```

---

## 十一、当前局限

- **训练未达成目标**：PPO 跑到 260 轮仍未稳定翻越第一个障碍，根因见 7.2
- **游戏路径硬编码两处**（`.csproj` 的 `<GameRoot>` 与 `project.json` 的 `executable_path`），换机需手动同步
- **游戏进程、训练脚本、TensorBoard 需分别启动**，无统一编排
- **多 agent 并发依赖游戏内同时存在 N 个角色实例**，游戏本身未设计为多实例场景，agent 数量过多时部分物理交互会互相干扰
- **`src/GameRuntime_v2/FILE_OVERVIEW.md` 已过时**（其中标注为"待实现"的 PlayerControl 与通信服务实际已实现，且通信为 TCP 而非 UDP），以 `doc/entrypoints.md` 为准

---

## 十二、说明

本项目为个人研究性质，用于探索"在没有现成环境接口的商业软件上构建 RL 环境"的工程路径。不包含任何游戏本体资源；运行需自行拥有游戏正版及 BepInEx。

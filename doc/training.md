# 训练模型与数据流

本目录实现 Getting Over It 的强化学习训练。Python 作训练侧（TCP 客户端），C# `GameRuntime_v2` 作服务端（`TcpStepServer`，端口从 `src/config/project.json` 的 `tcp_port` 读取，默认 9000）。每个 RL step 对应一次阻塞式请求-响应，实现帧级同步。

## 一、整体架构

存在两条并行训练管线，共享同一套环境接口、观测构建逻辑与投放机制。观测构建（dynamics 特征、4ch patch、定长窗口填充/mask）在离线数据集与在线采集之间**收敛为单一实现**（`dataset.build_dynamics`/`build_patch`/`left_pad_sequence`），从结构上保证训练与推理逐窗口一致，不依赖文档手动对齐：

| 管线 | 入口 | 算法 |
|------|------|------|
| BC 行为克隆迭代微调 | `main_train.py` | 监督学习（MSE 回归） |
| PPO 在线强化学习 | `main_ppo.py` | Clipped Surrogate + GAE |

当前**没有 SAC/DQN**，也没有 off-policy 持久 replay buffer。

### 文件职责

| 文件 | 职责 |
|------|------|
| `config.py` | 集中超参 `TrainConfig`（环境、模型、BC、PPO、投放、奖励） |
| `main_train.py` | BC 迭代训练入口：冷启动随机采集 → 效率图 → 筛选轨迹 → fine-tune |
| `main_ppo.py` | PPO 训练入口：启动游戏 → collect_ppo → PPO update → checkpoint |
| `rollout.py` | 数据采集核心：`RolloutWorker` 管理游戏生命周期、deploy、多种采集模式 |
| `deploy_sampling.py` | 空投几何采样（纯数学，无游戏依赖）：弧长加权、固定点池、分层分配 |
| `model.py` | `ActionPredictor` + `SpatialEncoder`（BC 策略网络） |
| `actor_critic.py` | `ActorCritic`（PPO Actor-Critic，共享 backbone） |
| `dataset.py` | `Trajectory` 数据结构、`TrajectoryDataset`，以及**离线/在线共享的观测构建原语**：`build_dynamics`、`build_patch`（单样本 4ch patch）、`left_pad_sequence`（定长窗口左填充 + valid_mask） |
| `trainer.py` | BC 监督训练器（MSE + AdamW + early stopping） |
| `ppo_trainer.py` | PPO 更新器（clipped surrogate + value clip + entropy + KL 早停） |
| `ppo_buffer.py` | PPO on-policy buffer（GAE + minibatch，非持久 replay） |
| `reward.py` | 攀升门控进度势 `progress_metric`（secured dwell + 门控横向 reach）、轨迹评分（`base_score`/`score_trajectory`，共享核心 `climb_score`）、逐步奖励 `step_reward` + 确认式创新高 `NewHighConfirmer`、落水判定单一入口 `is_water`/`is_water_trajectory`、攀爬效率图 `ClimbingEfficiencyMap` |

相关模块：
- `src/env/goi_env.py` — `GoiEnv` TCP 客户端（reset/step/teleport/new_snapshot 等）
- `src/start/` — `GameLauncher` 启动游戏、`GameModeController` 切换 GameRuntime 模式

## 二、模型网络结构（BC 与 PPO 共享 backbone）

三模态输入，上下文长度 `context_len=32`：

- **dynamics** `(B, T, 39)` — 由 `dataset.build_dynamics()` 从原始 33D 状态构建的特征（水平方向平移等变），组成：
  - 15 维速度/角速度（player + 各部件线速度、player 角速度、**cursor 线速度**），原始值，见 `VELOCITY_INDICES`
  - 3 个角度（hubAngle/sliderAngle/hammerAngle）→ **(sin, cos)** 共 6 维；角度在 C# 侧为「度」，编码前 `deg2rad`，消除环绕不连续，见 `ANGLE_INDICES`
  - 6 个部件（hub/slider/handle/pole/tip/**cursor**）相对 player 的坐标 `(part - player)` 共 12 维，保留相对几何精度，见 `REL_POS_INDICES`
  - **1 维绝对高度 `player_y`**（`ABS_HEIGHT_INDEX`）：唯一刻意保留的非平移等变项。水平(x)无绝对偏好故对 x 等变；但重力、落水阈值、"爬得越高越好"均以世界 y 为绝对参照，缺失会让策略无法感知海拔/离水距离。归一化后绝对 y 与「离水距离 (y-threshold)」等价。
  - 以上 34 维为 `BASE_DYNAMICS_DIM`。
  - **5 维接触信号**（`contact_features.CONTACT_DIM`，roadmap P2.1）：`tip_contact` / `tip_grip` / `body_contact` / `pot_contact` / `fall_distance_norm`。Python 端以多边形几何复刻游戏原生 `HammerCollisions` + `PlayerSounds` 判断（阈值 `CONTACT_EPSILON=0.03m`、`GRIP_VEL_THRESHOLD=0.3`、向下射线 clip[0,20]/20 均对齐反编译 IL），**wire 协议不变（C# 仍只回传 33D）**。`body_contact` 与 `pot_contact` 按游戏 IL 里 `rb.GetPoint(contact).y > 0.4` 分层各自输出（body 触地=撞头 pain，pot 触地=稳定支点，语义相反，合并会互相抵消）。详见 `doc/optimization_roadmap.md` P2.1。
  - 维度常量 `DYNAMICS_DIM = BASE_DYNAMICS_DIM + CONTACT_DIM = 39`，**与 `config.state_dim=33`（原始状态维度）解耦**：模型 `dynamics_proj`/归一化 buffer 用 `DYNAMICS_DIM`，`state_dim` 仅指 C# 回传的原始 33 维。
- **patches** `(B, T, 4, 32, 32)` — 局部空间图，由 `config.patch_mode` 决定语义（离线 `dataset.build_patch` 与在线 `rollout._build_patches_batch` 共用同一实现）：
  - **`dualscale`（默认，双分支 · 强调碰撞几何）**：通道切分为 wide `[0]` + contact `[1:4]`，编码器 `DualScaleSpatialEncoder` 分两路 CNN 编码后 concat 融合。两分支均以 player 为中心，仅尺度不同：
    - `[0]` wide 攀爬效率图 — player 中心、**大范围低精度**（`wide_patch_resolution=2.0` m/px → 64 m 窗口，降采样）。效率图本就从 traversable 地形 mask 扩散而来，已隐含全局地形结构，故 wide 只放它做全局引导。
    - `[1]` contact 高精度地形实心 mask — player 中心、**中高精度**（`contact_patch_resolution=0.27` m/px → 8.64 m 窗口，半径 4.32 m），由 `rasterize_solid_local` 逐格点在多边形内判定（二值；地形特征米级，远大于一格，不需抗锯齿）。
    - `[2]` contact 锅/body 碰撞轮廓 — `body∪pot` 局部轮廓按 `player→hub` 方向 + `body_angle_offset` 重建为世界多边形后，按**面积覆盖率**栅格化（与地形同栅格 co-registered）。
    - `[3]` contact 锤头碰撞轮廓 — tip 局部轮廓按 `pole→tip` 方向 + `tip_angle_offset` 重建后同样按面积覆盖率栅格化（同一 player 中心网格）。
    - **部件轮廓必须抗锯齿（`PART_RASTER_SUBSAMPLES=4`）**：Player 部件远小于一格——真实 tip 轮廓 bbox 仅 `0.148 × 0.448` m，在 0.2 m/px 下宽 0.74 px、0.27 m/px 下 0.55 px，**都不足 1 像素**。旧的「格心在多边形内」二值判据下，多边形可整体落在相邻格心之间而一个格心都不含 → **整帧通道全零、锤头对 CNN 完全不可见**：实测旧配置（0.2 m/px 二值）空通道率 **8.7%**，平均仅 1.28 个像素。改为面积覆盖率（每格取被多边形覆盖的面积比例，`subsamples²` 个子采样点估计）后空通道率归 **0%**，且**亚像素位置与朝向由灰度强度承载**，信息量高于同分辨率的二值。子采样只在轮廓 bbox（通常几格）内做，故图元开销仅 +16%（44.4 → 51.4 μs），`build_patch` 整体 0.445 ms/次不变量级。诊断/基准：`python -m src.tests.analysis.diagnose_tip_rasterization`、`bench_patch_raster`；正确性单测 `src/tests/training/test_raster_area_coverage.py`（覆盖率总和 ≈ 多边形面积、内部格=1、`subsamples=1` 严格退化为旧二值）。
    - 三个 contact 通道 co-registered，使 CNN 直接看到锅/锤头各自与地形、以及彼此的接触关系。身体姿态另由 dynamics 向量承载。
    - **刻意不在 patch 里另加"接触边缘/接触点"通道**，理由是分辨率与职责分工：
      - **分辨率失配 6.7×**：contact 栅格 `0.2` m/px，而游戏接触阈值 `CONTACT_EPSILON=0.03` m。一格比阈值粗 6.7 倍 → "贴着接触"与"悬空 0.15m"落在同一格，patch 上长得一样；且栅格化是「格中心点在多边形内」判定（`MplPath.contains_points`），两多边形相切时边界格标 0 还是 1 完全取决于亚格子舍入 → **在 patch 分辨率下标出来的接触是走样的噪声，不是 ground truth**。
      - **CNN 已能从现有通道取到接触的空间信息**：ch1/ch2/ch3 co-registered，故"重叠区域"= 逐像素 AND = `relu(ch1+ch2-1)`，一个 1×1 卷积 + ReLU 就精确表达；接触法向同理（地形 mask 的局部梯度即法向，conv 第一层天生是梯度算子）。这与 P2.1 里"多边形距离查询是 O(V·E) 符号几何算法、CNN 学不动"是**完全不同量级**的难度，不需要替它算。
      - **职责分工**：patch 管**空间布局**（哪里有什么形状，格子级精度够用）；dynamics 向量管**精确物理谓词**（亚格子接触判定、grip 状态机、离地距离）——后者正是 P2.1 的 5 维接触信号，用精确几何算、不受栅格分辨率限制。在 patch 里再标一遍，等于用低精度媒介重复表达已用高精度媒介表达好的东西。
    - **窗口半径由锤子几何硬上界反推（已修复出窗，勿下调）**：contact 半径需 ≥ **3.78 m**，来自 `|hub−player| = 0.3546`（刚性偏移）+ slider 全伸 `|tip−hub| = 2.955`（关节硬限位）= tip 原点上界 **3.31 m**，取 `PlayerControl` IL 里硬编码的 `|cursor−player| ≤ 3.5f` 作设计上界（与求解器软性超调实测极值 ≈3.49 m 吻合），再加 tip 轮廓半径 **0.279 m**；另留 2 格地形上下文（锤头全伸时也要看得到它接触的那块地形）→ ≈**4.28 m**。旧配置 `32 × 0.2` = 半径 3.2 m 不足，约 **0.46%** 的帧里锤头贴窗边、接触的地形被裁掉，而这些恰是「锤子伸到最远够远处落点」的高价值帧。现配置 `32 × 0.27` = 半径 **4.32 m**，余量 +0.51 m。
      - **注意**：`|tip−player|` 有几何硬上界，其分布在上界处**截断而非高斯**。早期用 `dyn_mean/dyn_std` 正态外推得出的「2.9% 的帧 tip 中心出窗」是严重高估（正态尾延伸到无穷）——15.2 万真实帧里 tip 中心从未出过 3.2 m 窗口，真正的问题只是贴边裁掉周围地形。故核验一律走几何判据，勿用统计外推。
      - **为何改分辨率而非 `patch_size`**：保持 patch shape `(4,32,32)` 不变 → checkpoint 完全兼容（`SpatialEncoder` 里 `Linear(128*4*4, d)` 的 flatten 维度锁死 2048，改 size 需做空间对齐的权重迁移）、wide 分支不被迫从 64 m 变 80 m、每帧栅格化格点数不变。分辨率变粗的信息代价已被上面的面积覆盖率栅格化抵消。
      - 核验：`python -m src.tests.analysis.check_tip_patch_coverage`（几何硬上界 + 360° 最坏姿态扫描）；回归单测 `src/tests/training/test_patch_tip_coverage.py`（走真实 `build_patch`，断言 360° 全伸姿态下 ch3 既非空也不贴窗边，并反向确认旧 0.2 m/px 确实会裁）。
    - **ch2 仍是 `body∪pot` 合并**（与向量端 P2.1 把 `body_contact`/`pot_contact` 拆开**有意不对称**）：patch 是空间图，body 与 pot 本就空间分居上下，CNN 从形状上能区分上半团/下半团；而向量端若合并成单个标量 0/1 则是真正的信息丢失（两者语义相反会抵消）。故只拆向量端。
    - **TODO（轮廓朝向为粗标定）**：锤头 `tip_angle_offset=-90°`（`pole/-90/scale=1`，经 `verify_tip_reconstruction` 粗扫描）与 body `body_angle_offset=-90°`（`player→hub / -90 / scale=1`，经 `verify_body_reconstruction` 粗扫描确认躯干朝上、锅贴地）均已在 90° 粒度确认，但仍为近似；精确值待细粒度扫描或 C# 回传部件世界顶点真值。
  - **`legacy`（旧单尺度 4ch，A/B baseline）**：以 player 为中心 0.5 m/px 的 4 通道 — ch0 地形可通行 mask、ch1 攀爬效率图、ch2 身体部件（player/hub/slider）高斯 blob、ch3 锤子部件（handle/pole/tip）高斯 blob。
- **actions** `(B, T, 2)` — 历史动作。
- **valid_mask** `(B, T)` — 1=有效 / 0=左填充；用于 episode 起步阶段历史不足时的 padding（见第 2.1、四节）。

Backbone：

```
dynamics 观测归一化 (dyn_mean/dyn_std buffer)
DualScaleSpatialEncoder:
  dualscale → wide CNN(1ch→d/2) + contact CNN(3ch→d/2) → concat → Linear(d)
  legacy    → 单 CNN(4ch→32→64→128 → Linear 2048→128)
dynamics_proj(Linear 39→128)
  → concat → fusion(Linear 256→128) = obs_embed
action_proj(Linear 2→128) = act_embed
  → 交错序列 [obs_0, act_0, obs_1, act_1, ...] 长度 2T=64
  → pos_embed + TransformerEncoder(3 层 / 4 头 / d=128, causal mask + key_padding_mask)
  → 取最后一个 token
```

输出头：

| 模型 | 输出 | 动作空间 |
|------|------|----------|
| `ActionPredictor` (BC) | `Linear(128→2) → tanh × 100` | 确定性回归 |
| `ActorCritic` (PPO) | Actor 高斯 (mean, std) + tanh squash × 100；Critic MLP → V(s) | 随机策略 + log_prob 修正 |

默认超参：`d_model=128`、`nhead=4`、`num_layers=3`、`action_scale=100.0`、`context_len=32`、`state_dim=33`（原始状态维度）、`DYNAMICS_DIM=39`（模型消费的动力学特征维 = 34 base + 5 接触信号）。

### 2.1 观测预处理与归一化

- **观测归一化**：`_DynamicsNormMixin` 以 `dyn_mean`/`dyn_std`/`dyn_stats_ready` 三个 buffer 随 `state_dict` 持久化，归一化在 `forward` 内部完成，训练/推理天然一致。标定时机：
  - BC — 首次训练前用 `compute_dynamics_stats` 从冷启动轨迹标定；
  - PPO — 从 BC 迁移（`--resume-bc`，`load_from_bc` 自动复制统计量）或 PPO checkpoint 继承；全新启动则先跑一轮标定 rollout。
  - 标定后冻结，不随迭代变化，避免 Critic 目标漂移。
- **padding mask**：历史窗口左填充零（起步阶段）。定长窗口的左填充与 `valid_mask` 生成由**唯一原语 `dataset.left_pad_sequence`** 完成，离线 `__getitem__` 与在线 `rollout._build_history_window` 共用它（各自把「右对齐的真实步序列」交给它填充），从结构上消除历史上的 off-by-one。随后 `build_key_padding_mask` 将 `valid_mask` 按交错序列展开为 `(B, 2T)` 加性掩码（填充位 `-inf`，与 causal mask 同为浮点类型），使 Transformer 忽略填充 token；最后一个时间步恒有效，不会出现整行 `-inf` 的 NaN。

> 兼容性：原始状态由 29 维扩为 **33 维**（新增 fakeCursor 绝对坐标+速度，索引 29-32），`state_dim=33`。**动力学特征维 `DYNAMICS_DIM` 现为 39**（34 base 含绝对高度 `player_y`，+ 5 维 P2.1 接触信号），与 `state_dim` 解耦；`dynamics_proj` 输入维度与归一化 buffer 随之为 39。
>
> **旧 checkpoint 走 partial-load 无缝迁移**：`model.expand_state_dict_for_dynamics_dim(ckpt_state, current_state)` 把 `dynamics_proj.weight` 前 34 列原值保留、后 5 列初始化 0，`dyn_mean`/`dyn_std` 前 34 保留、后 5 pad 0/1 → 新特征"起手无效应"，训练中逐步被学到。`main_ppo` 的 `--resume` 路径已自动调用；已用真实 iter=260 ckpt（`dynamics_proj.weight (128,34) → (128,39)`）端到端验证（`src/tests/training/test_p2_contact_integration.py`）。
>
> 原始 33 维状态本身不变，故只含 33 维原始状态的轨迹缓存仍可用（`build_dynamics` 会在读取时重新产出 39 维特征；缺 `contact` 序列的老轨迹尾 5 维零填充，等价旧 34D 行为）；`environment.json`、效率图 npz 等环境级产物照常复用。

## 三、Observation / Action

原始 33D 状态（索引 0-28 来自 C# `PlayerState.ToFloatArray()`，索引 29-32 由 `StepController.CollectAllStates()` 从 `fakeCursorRB` 追加）：

| 索引 | 字段 |
|------|------|
| 0-1 | playerX, playerY |
| 2-4 | velocityX, velocityY, angularVelocity |
| 5-9 | hubX/Y, hubVelX/Y, hubAngle |
| 10-14 | sliderX/Y, sliderVelX/Y, sliderAngle |
| 15-18 | handleX/Y, handleVelX/Y |
| 19-22 | poleX/Y, poleVelX/Y |
| 23-26 | tipX/Y, tipVelX/Y |
| 27 | hammerAngle |
| 28 | timestamp |
| 29-30 | cursorX, cursorY（fakeCursor 绝对坐标） |
| 31-32 | cursorVelX, cursorVelY（fakeCursor 线速度） |

> **为何加 cursor**：hammer 由不可见的 `fakeCursorRB` 牵引（实现为 hinge/slider 关节速度伺服而非弹簧，锤子物理模型与力传递的数学表示见 `doc/hammer_physics.md`），其位置/速度是控制链的隐藏状态。原始 29D 缺失 cursor 后观测非马尔可夫（同一部件状态下施加相同 action 可得不同响应）。补入 cursor 4 维后恢复马尔可夫性。teleport 时 cursor 随 player 平移、速度清零，保证 reset 一致性。

- **Action**：2D 鼠标相对位移 `(dx, dy)`，范围 `[-100, 100]`，经 C# `PlayerInputService` 反射注入 `mouseInput`。
- **Done**：C# 侧 `done` 标志，或 Python 侧落水判定。落水阈值判定收敛为单一入口 `reward.is_water(y, config)`（`y < water_y_threshold`，默认 -10.0）；step 级 done、`step_reward` 落水扣分、轨迹级过滤 `is_water_trajectory`（最低点触水）均由其派生。

## 四、完整数据流链路

```
GameLauncher 启动游戏 → GoiEnv.connect(9000)
  → warmup 100 步零动作 → new_snapshot() 固定 reset 基准
  → (可选) 采样表面坐标 → teleport → settle 30 步 → 稳定性过滤
循环每 step:
  33D raw → build_dynamics(33D 特征) + build_patch(4ch) + left_pad_sequence(32 步窗口左填零 + valid_mask)
  → 模型内部观测归一化 → 模型推理 2D action
  → env.step(actions[N,2]) → TCP STEP → C# 注入 mouseInput
  → Physics2D.Simulate × stepFrames → 回传 N×33D 状态 + dones
  → step_reward() → buffer.add()
更新: PPO 当轮 buffer 做 GAE+minibatch 更新 / BC 轨迹评分筛选后 fine-tune
```

多 agent 并行：一次 STEP 同时推进 N 个复制体（agent 0 = 原始 Player，1..N-1 = Duplicate），提高样本效率。PPO 每轮迭代启动/关闭游戏一次以防内存泄漏。

## 五、数据缓冲差异

- **PPO**：`PPORolloutBuffer`，on-policy。每 agent 独立 GAE 后 merge，每轮更新完即丢弃。步级额外存 `noise_prev = z_{t-1}`（OU AR(1) 前一步归一化噪声，(2,)），供 PPO 更新用条件式 log_prob 复算 → ratio 数学自洽；iid 模式（`ou_enabled=False`）下退化为零向量、无影响。
- **BC**：`TrajectoryDataset` 只存原始 `(33D states, 2D actions)`（约 70KB/条），dynamics/patch 在 `__getitem__` 即时构建；双层过滤——`add_trajectories` 按 score top-K% 筛选入库 + `trim_oldest` FIFO 滑动窗口（最多 300 条）；`secured peak 截断`只取到 `progress_metric` 的 secured 峰 `τ ∈ [0, min(peak_idx, T-1)]`（dwell 驻留过滤甩飞尖峰后的守住峰），排除越峰跌落段与瞬时尖峰段。类似 filtered replay。
- **SIL 精英池（PPO 侧）**：默认走 MAP-Elites 行为多样性准入（`add_trajectories_map_elites`，P1.2）——bd = secured peak (x,y) → 5m×5m 网格；每 cell 保留分数最高一条；`is_water_trajectory` 硬拒；冷启动 |pool| < `sil_cold_start_min` 阶段允许同 cell 覆盖。回退到旧 top-K% 相对门槛需设 `sil_map_elites_enabled=False`（BC 路径 `main_train` 无关，仍走原 `add_trajectories`）。

### 5.1 窗口对齐（训练/推理一致性）

数据集窗口与在线推理**逐窗口对齐**：预测动作步 `τ` 时，
- 观测窗口 `obs_{τ-ctx+1 .. τ}`（**含当前观测 `obs_τ`**）
- 动作历史 `act_{τ-ctx .. τ-1}`（当前步之前）
- 监督目标 `act_τ`

早期步（`τ < ctx-1`）历史不足时左填充零并由 `valid_mask` 标记，与推理起步阶段完全一致（因此 `τ` 从 0 开始枚举，也让 BC 训练覆盖起步阶段，并恢复利用 `peak_t < ctx` 的短轨迹）。此对齐修复了此前"数据集用 `(obs_k, act_k)` 同索引、推理用 `(obs_{k+1}, act_k)`"的 off-by-one 错位。

**一致性实现**：离线与在线均先取「右对齐的真实步序列」（离线按索引切片、在线取历史尾部），再统一交给 `dataset.left_pad_sequence(entries, ctx)` 左填充到定长并生成 `valid_mask`（obs 驱动）。填充/掩码只有一份实现，故对齐由代码结构保证而非人工约定；已用同一轨迹对拍离线 `__getitem__` 与在线 `_build_history_window`，在 `τ ∈ [0, ctx)` 填充区与 `τ ≥ ctx` 满窗区逐元素完全一致。

## 六、Airdrop / Deploy 采样机制（候选点唯一文件 → train）

1. **L7 几何**（`src/tests/control_interaction/test_l7_surface_airdrop.py`）：加载 `environment.json` + `player_contour.json`，算 `min_clearance = player_height + surface_padding(0.3)`，扫描找净空足够的可着陆表面线段。
2. **几何采样 + 过滤**（`deploy_sampling.py`）：弧长加权在表面上采候选，y += `drop_height(2.0)`；`--physics-filter-all` 做物理探测 + 锤头中心点判据 + 人工涂抹排除区过滤。
3. **物理过滤**（`src/tests/control_interaction/drop_point_physics.py`）：批量 teleport → settle，`drop = teleport_y - settled_y <= settle_drop_threshold(5.0)` 才算稳定。
4. **唯一候选文件**：全量稳定候选集写入项目 `checkpoints/drop_points_all_stable.json`（`default_candidate_points_path()`，随仓库版本化），作为向 train 传递候选点的**唯一通道**。`<game>/GoiData/Colliders/drop_points.json`（L8 子集）仅供 `test_l8_airdrop_deploy.py`。
5. **训练侧 deploy（pool-only）**（`main_ppo`/`main_train` + `rollout.py`）：读唯一候选文件 → train 侧**高度过滤**（`y ≤ y_max_cutoff + drop_height`）→ `set_candidate_points()`（再按排除区过滤）；每轮 `_sample_positions` 从候选池无放回抽样，`_get_stable_agents` 里 settle 落差 / 锤头中心 / 排除区兜底再筛。agent 0 始终留初始位置。

管线差异：
- `TrainConfig.random_deploy` 默认 `True`（候选池投放）。**当前最小任务（roadmap P0.1）应显式传 `--no-random-deploy`**，详见 [`main_ppo_usage.md`](main_ppo_usage.md)。**不再在训练侧重算表面线段随机采样**——候选点全部来自 L7 唯一文件。
- 高度上限 `y_max_cutoff` 现作为 **train 侧候选点集的高度过滤阈值**（不再用于 L7 表面裁剪的训练路径）。

验证工具：`test_l8_airdrop_deploy.py` 批量 deploy + settle 落差统计；`visualize_cached_drops.py` 空投超参可视化。

## 七、评分、奖励与效率图

### 7.0 攀升门控进度势（价值核心：取代单一纵向 summit）

价值核心由 `progress_metric(states, config)` 给出，**取代旧的单一纵向 `summit=y_max-y0`**，
解决「爬树死路刷高分、右侧真实路径压不过」与「甩到顶又滑回的甩飞刷分」：

- `held = _secured_held(y)`：每位置折算为「dwell 步窗口内守得住的高度」（窗口跨度 ≤ `dwell_drop_tol`
  取窗口最高点，否则取最低点）→ 甩飞尖峰被压低。secured 峰 `peak_idx = argmax held`。
- `secured_dy = held[peak_idx] - y0`（守住的净爬升）；`reach = x[peak_idx] - x0`（到 secured 峰的向右伸展）。
- `progress = secured_dy + progress_wx · max(0, reach)`，**仅当 secured_dy > 0**（不爬升不计横向、向左不计）。

BC 与 PPO 的进度信号**故意分化，且已锁定对应关系，勿误合并**：

- **BC 轨迹评分** `base_score`/`score_trajectory`：对**完整轨迹算一个标量**用于 top-K% 筛选，
  = `climb_score(progress)`（相对攀爬核心，量纲自洽 = `progress + efficiency_weight·(progress/peak_step·speed_ref_steps)`，两项均为米，`efficiency_weight` 为无量纲权重）
  + `abs_height_bonus`（绝对高度**轨迹级**排序加成）。**waypoint 默认不进 BC 排序**（`bc_waypoint_weight=0.0`）：效率图是 value-to-go，端点差 `E(peak)-E(start)` 沿真实攀爬多为负会压后好轨迹，且势函数塑形（PBRS）只适用于 RL 回报口径、不适用于监督排序；效率图对 BC 仅经 patch wide 通道输入生效。
- **PPO 逐步奖励** `step_reward`：见 7.1，是上述 `progress` 的**逐步望远镜分解**（确认式创新高累加 → secured_dy；前沿门控横向累加 → 门控 reach）。BC 因能直接对完整轨迹取 secured 峰用标量即可；PPO 逐步优化需可加的差分形式做信用分配。

对应关系（三通道方向一致，故 BC→PPO 迁移不互相推翻）：

| 通道 | BC（轨迹级标量） | PPO（逐步差分） |
|------|------------------|-----------------|
| 纵向攀爬 | `secured_dy`（dwell 守住的净爬升） | 确认式创新高（`NewHighConfirmer`，望远镜和 = secured_dy） |
| 横向进度 | `progress_wx·max(0,reach)`（到 secured 峰） | 前沿门控 `progress_wx·max(0,Δx)·1[前沿]` |
| 效率 | 不进 BC 排序（`bc_waypoint_weight=0`，仅作 patch 输入） | PBRS 势场塑形 `γ·Φ(s')−Φ(s)`（Φ=效率图，不改最优解） |
| 绝对高度 | `abs_height_bonus`（每条轨迹，轻微 tie-break） | 无（`height_prior_weight` 默认 0.0） |

> `climb_score` 是 `base_score` 与 `ClimbingEfficiencyMap.update` 共享的唯一相对攀爬公式，输入统一为 `progress`，**刻意不含绝对高度**；绝对高度仅由 `abs_height_bonus`（轨迹级 tie-break）承担。`height_prior_weight` 默认 0.0（旧值 0.02 会给死路树顶白送 floor，与门控进度势冲突）。

### 7.1 奖励与效率图（PPO）

- **逐步奖励** `step_reward`（`running_max_y` 必传，可选 `confirmer`/`normalizer`）：
  - 纵向分量：提供 `confirmer`（`NewHighConfirmer`）时用**确认式创新高**——新高需在 `dwell_steps` 步窗口内守住才结算发奖（防甩飞，望远镜和 = secured_dy）；未提供时退化为即时创新高，回落按 `-neg_reward_scale × log(1+drop)`（`neg_reward_scale` 默认 **0.0** → 跌落零显式惩罚，时间隐式受罚）。
  - 横向分量：`progress_wx · max(0, Δx)`，**仅在攀升前沿**（`curr_y ≥ running_max_y - dwell_drop_tol`）计——低处平地向右滑/向左不计，与 BC 门控口径一致。
  - 效率分量：**势函数塑形 PBRS**（Ng et al. 1999）`F = γ·Φ(s') − Φ(s)`，`Φ=效率图`（value-to-go）。加到主奖励**不改变最优策略**，只提供稠密引导。旧式无 γ 的 `Φ'−Φ`（=γ=1 特例）在折扣 RL 里会偏置最优解；引入 γ 后把「未来可达价值」正确贴现回当前步（value-to-go 沿好路虽递减，但 γΦ'−Φ 方向与主进度奖励一致），这正是 value-to-go 作势场的正确用法。
  - 归一化：`RewardNormalizer`（Welford 在线 σ，约若干轮后冻结）；进度通道（纵向+横向）与 PBRS 塑形通道各自归一。σ 冻结后 `α`、`σ_h/σ_e` 皆常数 → 等价「主奖励 + 常数·PBRS」再乘全局正标量，**策略不变性在冻结后严格成立**（常数倍势场仍合法、全局缩放不改 argmax）。
  - 落水：`is_water(curr_y)` 命中则覆盖为 `config.water_penalty`（默认 -10.0，与坐标阈值 `water_y_threshold` 语义解耦，勿混用）。
- **攀爬效率图** `ClimbingEfficiencyMap`：网格分辨率 1.0m，从碰撞多边形构建 traversable mask；跨迭代沿轨迹路径传播 secured-progress 的价值（`waypoint_weight` 为传播系数，后缀 O(T)），每轮内做地形感知邻域扩散（不穿墙，静态邻域结构预计算复用），构成 value-to-go 势场 `Φ`。两处消费：① 作为 patch wide 通道输入模型；② 对 PPO 以 **PBRS 势场塑形** `γΦ'−Φ` 生效（不改最优解）。BC 排序默认不消费效率图（`bc_waypoint_weight=0`）。`height_prior_weight` 默认 0.0：格值完全由 secured-progress 沿轨迹回传决定（死路因未来无处可去而贬值），远距连通靠轨迹价值回传 + random_deploy 分层课程。
  - **mask 走样核验**：`python -m src.tests.analysis.verify_mask_aliasing`（1m 中心采样漏判薄墙约 1.57%，路线拓扑保留；如需可加 3×3 子格采样，无须降到 0.5m）。
  - **contact patch 覆盖核验**：`python -m src.tests.analysis.check_tip_patch_coverage`（锤子几何硬上界 vs 窗口半径 + 360° 最坏姿态扫描）；`diagnose_tip_rasterization`（部件轮廓栅格化命中率/空通道率）；`bench_patch_raster`（栅格化开销）。见第二节 contact 分支说明。
  - **progress_wx 标定**：`python -m src.tests.analysis.calibrate_progress_wx <过树轨迹.json>`（从 reach/secured_dy 反推，默认 0.5）。

## 八、TCP 通信协议（`GoiEnv` ↔ `TcpStepServer`）

| 命令 | 含义 | 载荷 |
|------|------|------|
| `R` | RESET | 所有 agent 恢复 snapshot |
| `S` | STEP | `[n][n×2×float32 actions]` |
| `N` | NEW_SNAPSHOT | 以当前状态为新基准 |
| `T` | TELEPORT | `[agentIndex][x][y]` |
| `F` | CAMERA_FREE | 自由相机 |
| `E` | EXPORT_COLLIDERS | 导出碰撞体 |
| `C` | CONFIG | Rewired 鼠标 override |
| `X` | CLOSE | 断开 |

响应：`[n:1B]` → 对每个 agent `[state: 33×4B][done: 1B]`（按 agent 交错）。

## 九、运行入口速查

PPO 的 CLI 说明、推荐配方与废弃清单见 **[`main_ppo_usage.md`](main_ppo_usage.md)**（勿再把 `--random-deploy` 当默认主线）。

```bash
# BC 迭代训练
python -m src.training.main_train

# PPO：当前最小任务（关 deploy、短 episode；详见 main_ppo_usage.md）
python -m src.training.main_ppo \
  --no-random-deploy --steps-per-rollout 400 --persist-game \
  --resume-bc checkpoints/model_iter_0010.pt

# PPO：基座过树后的低处课程投放（非当前默认）
python -m src.training.main_ppo --random-deploy --y-max-cutoff 125 --steps-per-rollout 500

# L7 生成候选点唯一文件（写 checkpoints/drop_points_all_stable.json，train 读取源）
python src/tests/control_interaction/test_l7_surface_airdrop.py --physics-filter-all

# L8 验证 deploy
python src/tests/control_interaction/test_l8_airdrop_deploy.py

# 模型评估
python src/tests/control_interaction/test_l9_model_eval.py --checkpoint checkpoints/ppo_iter_0050.pt
```

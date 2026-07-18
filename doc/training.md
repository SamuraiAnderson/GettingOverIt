# 训练模型与数据流

本目录实现 Getting Over It 的强化学习训练。Python 作训练侧（TCP 客户端），C# `GameRuntime_v2` 作服务端（`TcpStepServer`，端口从 `src/config/project.json` 的 `tcp_port` 读取，默认 9000）。每个 RL step 对应一次阻塞式请求-响应，实现帧级同步。

## 一、整体架构

存在两条并行训练管线，共享同一套环境接口、观测构建逻辑与投放机制：

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
| `dataset.py` | `Trajectory` 数据结构、4ch patch 即时构建、`TrajectoryDataset` |
| `trainer.py` | BC 监督训练器（MSE + AdamW + early stopping） |
| `ppo_trainer.py` | PPO 更新器（clipped surrogate + value clip + entropy + KL 早停） |
| `ppo_buffer.py` | PPO on-policy buffer（GAE + minibatch，非持久 replay） |
| `reward.py` | 轨迹评分、`step_reward`、攀爬效率图 `ClimbingEfficiencyMap` |

相关模块：
- `src/env/goi_env.py` — `GoiEnv` TCP 客户端（reset/step/teleport/new_snapshot 等）
- `src/start/` — `GameLauncher` 启动游戏、`GameModeController` 切换 GameRuntime 模式

## 二、模型网络结构（BC 与 PPO 共享 backbone）

三模态输入，上下文长度 `context_len=32`：

- **dynamics** `(B, T, 29)` — 由 `dataset.build_dynamics()` 从原始 29D 状态构建的**平移等变**特征（不含绝对坐标），组成：
  - 13 维速度/角速度（player + 各部件线速度、player 角速度），原始值，见 `VELOCITY_INDICES`
  - 3 个角度（hubAngle/sliderAngle/hammerAngle）→ **(sin, cos)** 共 6 维；角度在 C# 侧为「度」，编码前 `deg2rad`，消除环绕不连续，见 `ANGLE_INDICES`
  - 5 个部件（hub/slider/handle/pole/tip）相对 player 的坐标 `(part - player)` 共 10 维，保留相对几何精度，见 `REL_POS_INDICES`
  - 维度常量 `DYNAMICS_DIM = 29`（= `config.state_dim`）
- **patches** `(B, T, 4, 32, 32)` — 以 player 为中心的 4 通道局部空间图：
  - ch0 地形可通行 mask
  - ch1 攀爬效率图
  - ch2 身体部件（player/hub/slider）高斯 blob
  - ch3 锤子部件（handle/pole/tip）高斯 blob
- **actions** `(B, T, 2)` — 历史动作。
- **valid_mask** `(B, T)` — 1=有效 / 0=左填充；用于 episode 起步阶段历史不足时的 padding（见第 2.1、四节）。

Backbone：

```
dynamics 观测归一化 (dyn_mean/dyn_std buffer)
SpatialEncoder(CNN 4→32→64→128 → Linear 2048→128)
dynamics_proj(Linear 29→128)
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

默认超参：`d_model=128`、`nhead=4`、`num_layers=3`、`action_scale=100.0`、`context_len=32`、`state_dim=29`。

### 2.1 观测预处理与归一化

- **观测归一化**：`_DynamicsNormMixin` 以 `dyn_mean`/`dyn_std`/`dyn_stats_ready` 三个 buffer 随 `state_dict` 持久化，归一化在 `forward` 内部完成，训练/推理天然一致。标定时机：
  - BC — 首次训练前用 `compute_dynamics_stats` 从冷启动轨迹标定；
  - PPO — 从 BC 迁移（`--resume-bc`，`load_from_bc` 自动复制统计量）或 PPO checkpoint 继承；全新启动则先跑一轮标定 rollout。
  - 标定后冻结，不随迭代变化，避免 Critic 目标漂移。
- **padding mask**：历史窗口左填充零（起步阶段），`build_key_padding_mask` 将 `valid_mask` 按交错序列展开为 `(B, 2T)` 加性掩码（填充位 `-inf`，与 causal mask 同为浮点类型），使 Transformer 忽略填充 token；最后一个时间步恒有效，不会出现整行 `-inf` 的 NaN。

> 兼容性：`state_dim` 由旧的 16 改为 29 并新增归一化 buffer，`dynamics_proj` 输入维度随之改变，**旧的 `.pt` checkpoint 不再兼容，需重新训练**。而 `warmup_state.pkl`、`cache/`、`environment.json`、效率图 npz 等均为环境级产物（基于原始 29D 状态），仍可复用。

## 三、Observation / Action

原始 29D 状态（C# `PlayerState.ToFloatArray()`）：

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

- **Action**：2D 鼠标相对位移 `(dx, dy)`，范围 `[-100, 100]`，经 C# `PlayerInputService` 反射注入 `mouseInput`。
- **Done**：C# 侧 `done` 标志，或 Python 侧 `y < water_y_threshold(-10.0)` 判落水弃轨。

## 四、完整数据流链路

```
GameLauncher 启动游戏 → GoiEnv.connect(9000)
  → warmup 100 步零动作 → new_snapshot() 固定 reset 基准
  → (可选) 采样表面坐标 → teleport → settle 30 步 → 稳定性过滤
循环每 step:
  29D raw → build_dynamics(29D 特征) + 4ch patch + 32 步历史窗口(左填零) + valid_mask
  → 模型内部观测归一化 → 模型推理 2D action
  → env.step(actions[N,2]) → TCP STEP → C# 注入 mouseInput
  → Physics2D.Simulate × stepFrames → 回传 N×29D 状态 + dones
  → step_reward() → buffer.add()
更新: PPO 当轮 buffer 做 GAE+minibatch 更新 / BC 轨迹评分筛选后 fine-tune
```

多 agent 并行：一次 STEP 同时推进 N 个复制体（agent 0 = 原始 Player，1..N-1 = Duplicate），提高样本效率。PPO 每轮迭代启动/关闭游戏一次以防内存泄漏。

## 五、数据缓冲差异

- **PPO**：`PPORolloutBuffer`，on-policy。每 agent 独立 GAE 后 merge，每轮更新完即丢弃。
- **BC**：`TrajectoryDataset` 只存原始 `(29D states, 2D actions)`（约 61KB/条），dynamics/patch 在 `__getitem__` 即时构建；双层过滤——`add_trajectories` 按 score top-K% 筛选入库 + `trim_oldest` FIFO 滑动窗口（最多 300 条）；`peak 截断`只取到最高点 `τ ∈ [0, min(peak_t, T-1)]`，排除跌落段。类似 filtered replay。

### 5.1 窗口对齐（训练/推理一致性）

数据集窗口与在线推理**逐窗口对齐**：预测动作步 `τ` 时，
- 观测窗口 `obs_{τ-ctx+1 .. τ}`（**含当前观测 `obs_τ`**）
- 动作历史 `act_{τ-ctx .. τ-1}`（当前步之前）
- 监督目标 `act_τ`

早期步（`τ < ctx-1`）历史不足时左填充零并由 `valid_mask` 标记，与推理起步阶段完全一致（因此 `τ` 从 0 开始枚举，也让 BC 训练覆盖起步阶段，并恢复利用 `peak_t < ctx` 的短轨迹）。此对齐修复了此前"数据集用 `(obs_k, act_k)` 同索引、推理用 `(obs_{k+1}, act_k)`"的 off-by-one 错位。

## 六、Airdrop / Deploy 采样机制（四级）

1. **L7 几何**（`src/tests/control_interaction/test_l7_surface_airdrop.py`）：加载 `environment.json` + `player_contour.json`，算 `min_clearance = player_height + surface_padding(0.3)`，扫描找净空足够的可着陆表面线段，`y_max_cutoff=380` 过滤过高面。
2. **几何采样**（`deploy_sampling.py`，纯数学无游戏依赖）：弧长加权随机选线段 + 弧上均匀采样，y += `drop_height(2.0)`；或从固定点池 `drop_points.json` 无放回抽取；`compute_drop_points` 用最大余数法按弧长×高度权重分配。
3. **物理过滤**（`src/tests/control_interaction/drop_point_physics.py`）：批量 teleport → settle，判据 `drop = teleport_y - settled_y <= settle_drop_threshold(5.0)` 才算稳定，结果写入 `drop_points.json` / `drop_points_all_stable.json`。
4. **训练侧 deploy**（`rollout.py` 的 `_reset_and_deploy` / `_get_stable_agents`）：agent 0 始终留初始位置，复制体只有稳定的才参与采集。

管线差异：
- BC 管线默认启用随机 deploy（每轮随机表面投放）。
- PPO 管线默认 `random_deploy=False`（所有 agent 留初始位置），需 `--random-deploy` 才开启表面随机投放。

验证工具：`test_l8_airdrop_deploy.py` 批量 deploy + settle 落差统计；`visualize_cached_drops.py` 空投超参可视化。

## 七、奖励与效率图（PPO）

- **逐步奖励** `step_reward`：
  - 高度分量（创新高模式）：超过 `running_max_y` 给正奖励；回落给 `-neg_reward_scale × log(1+drop)` 轻微惩罚。
  - 效率分量：`eff_map.query(curr) - eff_map.query(prev)`。
  - 归一化：`RewardNormalizer`（Welford 在线 σ，约若干轮后冻结）。
  - 落水：`-10.0`。
- **攀爬效率图** `ClimbingEfficiencyMap`：网格分辨率 1.0m，从碰撞多边形构建 traversable mask；跨迭代沿轨迹路径传播 waypoint 奖励，每轮内做地形感知邻域扩散（不穿墙）。既作为 patch ch1 输入模型，也用于 waypoint 奖励引导。

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

响应：`[n:1B]` → 对每个 agent `[state: 29×4B][done: 1B]`（按 agent 交错）。

## 九、运行入口速查

```bash
# BC 迭代训练
python -m src.training.main_train

# PPO 训练（可选随机 deploy）
python -m src.training.main_ppo --random-deploy --num-agents 10

# 从 BC 初始化 PPO
python -m src.training.main_ppo --resume-bc checkpoints/model_iter_0010.pt

# L7 生成投放点
python src/tests/control_interaction/test_l7_surface_airdrop.py --physics-filter

# L8 验证 deploy
python src/tests/control_interaction/test_l8_airdrop_deploy.py

# 模型评估
python src/tests/control_interaction/test_l9_model_eval.py --checkpoint checkpoints/ppo_iter_0050.pt
```

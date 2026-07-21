---
name: project-task
description: 描述本项目（Getting Over It 强化学习）的核心任务、领域背景与关键约束，用于在任何涉及训练、环境交互、奖励设计、观测建模、TCP 通信或游戏控制的工作中锚定主题。当讨论 GoiEnv、GameRuntime_v2、BC/PPO 训练、rollout、reward、dataset、airdrop/deploy、锤子/cursor 控制等内容时应用。
---

# 项目任务：用强化学习训练 AI 玩《Getting Over It》

## 一句话定义

训练一个策略网络，**仅通过输出「鼠标相对位移」间接操控一把锤子**，让《Getting Over It》（掘地求升）中的角色尽可能往高处攀爬。

## 领域背景（AI 必须记住的前提）

- 《Getting Over It》是高难度物理攀爬游戏：角色坐在缸中，玩家用鼠标操纵锤子勾/撑/推地形把自己撑上山。操作极难、极不连续，一步失误可能滑落到起点。
- 攀爬是**稀疏、长时程、易回退**的信用分配问题，这是奖励设计的根本难点。
- 锤子由**不可见的 `fakeCursorRB`** 通过弹性约束牵引，其位置/速度是控制链的隐藏状态，属于观测的必要组成。原始状态为 **33D**（索引 29-32 为 cursor 坐标+速度）。

## 系统架构（不可动摇的约束）

- **Python = 训练侧 = TCP 客户端**；**C# `GameRuntime_v2`（BepInEx 插件）= 游戏内 = TCP 服务端（`TcpStepServer`）**。
- 每个 RL step = 一次阻塞式请求-响应，实现**帧级同步**。
- 端口从 `src/config/project.json` 的 `tcp_port` 读取（默认 9000），**禁止硬编码**。
- 多 agent 并行：一次 STEP 同时推进 N 个复制体（agent 0 = 原始 Player，1..N-1 = Duplicate）以提高样本效率。

## 观测 / 动作 / 终止

- **Observation**：33D 原始状态 → `build_dynamics`（平移等变特征）+ `build_patch`（以 player 为中心的 4ch 局部空间图）+ `left_pad_sequence`（定长窗口左填充 + valid_mask）。离线与在线**共享同一套观测构建原语**，训练/推理逐窗口一致。
- **Action**：2D 鼠标相对位移 `(dx, dy) ∈ [-100, 100]`，经 C# `PlayerInputService` 注入。
- **Done**：C# 侧 `done`，或 Python 侧落水判定（单一入口 `reward.is_water`）。

## 两条训练管线

| 管线 | 入口 | 算法 |
|------|------|------|
| BC 行为克隆迭代微调 | `main_train.py` | 监督学习（MSE），轨迹评分 top-K% 筛选后 fine-tune |
| PPO 在线强化学习 | `main_ppo.py` | Clipped Surrogate + GAE，可从 BC 迁移初始化 |

当前**没有 SAC/DQN，也没有 off-policy 持久 replay buffer**。

## 工作时的取向

- 任何改动都要维持**训练/推理观测一致性**（观测构建只有一份实现，勿手动对齐）。
- 奖励相关改动要遵守 **BC 排序信号 vs PPO 逐步奖励的分工**，`climb_score` 刻意不含绝对高度，勿并入。
- 落水判定、观测构建、窗口填充等均已收敛为**单一入口/原语**，扩展时优先复用而非新增分支。
- 修改数据结构（如状态维度）会破坏 checkpoint / 轨迹缓存兼容性，需评估重训与重标定成本。

## 深入参考

- 完整数据流、模型结构、TCP 协议、评分/奖励细节见 `doc/training.md`。
- 目录职责与通信规范见 `.cursor/rules/project-standards.mdc`。

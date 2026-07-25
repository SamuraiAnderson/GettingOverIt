# 方案 B 前提验证：游戏能否当作并行世界模型

**方案 B**（Expert Iteration / 游戏即模型的 CEM-MPC 规划）设想：用「NEW_SNAPSHOT + RESET + N 复制体」
把游戏本身当作完美世界模型——快照当前状态 → N 个 agent 从同一状态并行尝试不同动作序列 →
按 secured progress 挑最优 → 提交推进 → 再快照。规划纯搜索即可发现「过树」这类动作，
发现的精英轨迹再蒸馏进 BC/SIL。

本文记录该设想四条物理前提的**实测验证**结论。验证脚本：
`src/tests/control_interaction/test_planb_premise.py`（conda 环境 `getting-over-it-analysis`）。

## 一、验证方法

从同一快照出发，用 33D 完整状态（剔除 timestamp 列）逐位比较，量化以下断言：

| 断言 | 含义 | 失败意味着 |
|------|------|-----------|
| **P0a** 跨 agent（零动作） | reset 后所有 agent 零动作，agent 间是否保持一致 | 起点未对齐 / 串扰 / 非确定 |
| **P0b** 跨 reset（零动作） | 单 agent 零动作序列跑两次是否可复现 | 物理引擎非确定 或 reset 不完整 |
| **P1** 同起点 | reset 后 N 复制体完整状态是否逐位一致 | 复制体起点不对齐 |
| **P2** 确定性+零串扰 | 所有 agent 执行**同一**动作序列，轨迹是否逐位相同 | 非确定 或 复制体互相干扰 |
| **P3** 隔离性 | agent0 序列固定，其他 agent 零 vs 随机，agent0 是否不受影响 | agent 间状态泄漏 |
| **P4** 跨 reset 可复现 | 同起点+同序列跑两次是否一致 | reset 未完全恢复 或 物理非确定 |

**关键设计：零动作诊断（P0）**。零动作把混沌降到最弱，用来把「物理混沌」与「基础设施 bug」
解耦——否则暴力动作下一切都会分叉，无法归因。判定阈值 `IDENTICAL_EPS = 1e-2`（世界单位 m / m·s⁻¹），
并始终报告分歧随步数的增长曲线以区分「指数增长=混沌」与「恒定偏置=状态未恢复」。

## 二、实测结果（3 次会话，6 agent，horizon 150）

### 决定性对照

| 指标 | 值 | 判读 |
|------|-----|------|
| **P0b**（零动作跨 reset，单 agent） | **3e-8 ~ 3e-4** | 物理引擎逐位确定，刚体快照恢复正确 |
| **P0a**（零动作跨 agent） | 8e-3（对齐时）/ 0.21（不对齐时），**全程平坦** | 隔离正确、零串扰；但起点对齐不稳定 |
| **P4**（暴力动作跨 reset，单 agent） | **191 ~ 221** | 与 P0b 矛盾 → 见下「障碍 1」 |
| **P1**（reset 后跨 agent） | 9e-3 ~ **7.68**（跨会话跳变） | 复制体起点不稳定 → 见下「障碍 2」 |
| **P2**（同动作跨 agent） | 分歧 s1=0.03 → s5=5.8 → s20=10.6，数步爆炸 | 混沌 + 起点差污染，当前不可判读 |

### 分歧增长 vs 动作风格（可用规划时域）

| 动作风格 | 分歧行为 | 结论 |
|----------|----------|------|
| `hold`（恒定中等） | s10 尖峰 12.7 → **s50 收敛回 0.008** | 稳定动作下动力学**收敛**（contractive），非混沌 |
| `smooth`（低频正弦，大幅） | 数步内 >10，持续 | 大幅连贯动作仍强混沌 |
| `violent`（每步 ±100 白噪声） | s1=0.03 → s5 已 >100 | 最坏情况，e 折 < 1 步 |

## 三、根因诊断

### ✓ 核心设施可用

- **物理引擎确定**（P0b = 3e-8）：`Physics2D.Simulate` + 刚体快照逐位可复现，即使 6 agent 同时步进。
  这是方案 B 的根本基石——「游戏即世界模型」原理上成立。
- **复制体隔离正确**（P0a 零动作全程平坦）：`PlayerDuplicateManager.SetupCollisionIgnore`
  的 `Physics2D.IgnoreCollision` 生效，复制体两两及与原始 Player 互不碰撞、无串扰。
- **reset 恢复刚体正确**：位置/旋转/速度/角速度 + fakeCursor 恢复无误。

### ✗ 障碍 1（关键）：reset 未恢复 `PlayerControl` 的动作相关内部状态

**证据**：P0b（零动作跨 reset）= 3e-8，但 P4（暴力动作跨 reset）= 191。唯一变量是动作幅度。
既然刚体恢复逐位可复现，191 的不可复现只能来自一个**动作相关、reset 没恢复**的状态。

**定位**：`StepController.Reset()` 只恢复刚体 + fakeCursor + `RewiredMouseOverride.Reset()`，
**未恢复 `PlayerControl` 的内部累积字段**（见 `doc/hammer_physics.md`）：
- `mouseVelocityAverage`（自适应增益 g，EMA 累积）——动作相关，跨 episode 残留；
- `oldMouse`（上一帧 mouseInput，用于 `Lerp` 平滑）——动作相关；
- 次要：`mw`、`oldAngle`（D 项系数 ×0 故不影响）、`pauseInputTimer`、`mouseSnap`、`inputsToSkip`。

零动作下这些字段一帧内衰减归零，故 P0b 看不出；非零动作下跨 episode 残留 → 起点增益不同 →
混沌放大到 191。**这会同时破坏串行与并行规划的可复现性**。

### ✗ 障碍 2：复制体起点不对齐且跨会话不稳定

**证据**：P1 在 9e-3 ~ 7.68 之间跨会话跳变。

**定位**：`StepController.CaptureAllSnapshots()` 让每个 agent 快照**自身漂移后的状态**，
而非共享同一权威基准。warmup 结束时各 agent 已有微小差异，快照把差异固化下来。

### ⚠ 固有约束：确定性混沌，强烈依赖动作风格

分歧增长证明系统 Lyapunov 指数为正，但**混沌率取决于动作**：`hold` 稳定动作收敛、
`violent`/`smooth-大幅` 动作数步内爆炸。零动作下 λ≈0。

## 四、对方案 B 的结论

**前提当前不成立，但可修复，且验证暴露了正确的架构方向。**

1. **最干净的形态是「单 agent + 快照/回放串行评估候选」**，而非依赖 N 复制体逐位相同。
   P0b 证明单 agent 同起点同动作可复现到 1e-8——每个候选从**完全相同的快照**出发，
   排序精确，混沌不伤害排序（所有候选同起点）。**前提是先做修复 1**，否则串行评估会被
   上一候选的增益残留污染。

2. **并行 N 复制体（提速）**需额外做**修复 2**。修复后加上已验证的确定性 + 隔离，
   同动作应逐位相同，届时 P2/P3 才能真正检验残余串扰（当前被起点差 + 混沌污染，不可判读）。

3. **规划必须短时域滚动重规划（receding-horizon MPC）**，并优先采样连贯/稳定基元
   （呼应动作重参数化，见 `doc/training.md`「搜索空间压缩」讨论）——不要指望单条长开环
   rollout 有意义。`hold` 动作的收敛性说明存在低混沌的稳定基元区，规划应偏向这些区域。

## 五、修复清单与实施结果

| 编号 | 内容 | 位置 | 状态 |
|------|------|------|------|
| **修复 1** | 快照/恢复纳入 `PlayerControl` 内部字段（`mouseVelocityAverage`/`oldMouse`/`mw`/`oldAngle`/`pauseInputTimer`/`mouseSnap`/`inputsToSkip`） | `PlayerInputService.GetInternalState`/`SetInternalState`（反射）+ `StepController.CaptureAllSnapshots`/`Reset` | ✅ 已实施，实测有效 |
| **修复 2** | 广播 agent0 权威快照到所有 agent（刚体值 + fakeCursor + 内部态，保留各自 rb 引用） | `StepController.BroadcastCanonicalSnapshot`（`AlignAgentsToCanonical=true`） | ✅ 已实施，实测有效 |
| ~~修复 3~~ | ~~多帧 settle / 关节 warm-start 冲刷（切 `hj/sj.enabled` 重建 b2Joint）~~ | — | ❌ 实测无效，已回退（见下「障碍 3」） |

### 实施验证（Fix 1+2，多次会话，6 agent）

**结论：修复 1、2 确实生效，但结果强烈依赖启动亚稳姿态；strict 逐位测试无法全绿，属固有物理限制而非缺陷。**

- **静止启动姿态**下，干净零动作诊断**全绿**（直接证明两修复有效）：
  - `P0b`（零动作跨 reset，单 agent）= **9.2e-5，且 s1=2.4e-7（近机器精度）** → 修复 1 恢复内部态 + 刚体快照逐位可复现。
  - `P0a`（零动作跨 agent）= **2.3e-4** → 修复 2 广播使 N 复制体起点逐位对齐、零串扰。
- **摆动启动姿态**下，同样的零动作 P0a/P0b **退化到 ~0.1–0.15**（分歧集中在 state[4] 角速度 / state[27] hammerAngle）。
  分歧曲线呈**拍频（beat）**特征（如 s10=0.13→s50=1.3e-4→s100=0.13），非指数爆炸——
  说明存在一个 **~1e-4 的非确定性种子**，在摆动模态里表现为微小相位差，瞬时值可达摆幅（~0.14）。
- `P1/P2/P3/P4`（暴力动作或 reset-from-极端态）**始终红**：见障碍 2/3。

### ✗ 障碍 3（新发现，固有）：Box2D 求解器内部态不可经刚体快照恢复 → ~1e-4 非确定性地板

**证据**：静止姿态 P0b 收敛到 2.4e-7，但摆动姿态 P0b 达 0.14；且**多帧 settle（40 帧）与关节
warm-start 冲刷（切 `hj/sj.enabled` 重建 b2Joint）均无法消除**（实测 P0b 仍 0.14）。

**定位**：Unity `Physics2D`（Box2D）的**求解器内部态**——接触流形 / 关节 warm-start 累积冲量——
不随 `rb.position/velocity` 写入而恢复，也不由公开 API 暴露清除。它在两次 reset 间因**上一 episode
历史不同**而不同，构成一个 ~1e-4 的种子。静止姿态下无放大（P0b≈机器精度）；摆动姿态下被拍频/
混沌放大到 O(0.1)。这是「游戏即**逐位**世界模型」不可逾越的下限。

### ⚠ 障碍 2 复述：确定性混沌是 P2/P3/P4 全红的根因（与任何 reset 修复无关）

`violent` 动作 e 折 <1 步：即使把种子压到机器精度 1e-7，150 步（乃至 5 步）激烈动作也必然放大到 O(0.1)。
故 **P2/P3/P4 在长时域 + 激烈动作下物理上不可能逐位复现**，`IDENTICAL_EPS=1e-2` 阈值不可达——
这不是 bug，原文档第三节已预告。`P1` 因紧跟混沌 regime、reset-from-极端态叠加障碍 3，同样红。

## 六、最终结论（修订）

1. **修复 1、2 是必要且有效的**：它们消除了**系统性大误差**（内部增益残留、复制体起点漂移），
   把干净诊断从 O(0.1–1) 压到 O(1e-4~1e-7)。这是「游戏即世界模型」可用的前提基石。
2. **「逐位相同的世界模型」不可达**：受障碍 3（Physics2D 求解器态 ~1e-4 地板）+ 障碍 2（确定性混沌）
   双重限制，strict 测试（阈值 1e-2、长时域/激烈动作）无法全绿。
3. **方案 B 应作「近似」世界模型用**：短时域滚动重规划（receding-horizon MPC）、优先稳定/连贯基元、
   候选排序（对 1e-4 种子鲁棒），而非依赖长开环逐位复现。
4. **待办（若要提高 P0/P1 稳定转绿率）**：让 `NormalizeInitialPose` 可靠收敛到**静止**姿态
   （消除启动摆动抽签），可使零动作诊断稳定全绿；但对 P2/P3/P4 的混沌红无能为力。

## 七、复现命令

```bash
# 长时域（观察混沌）
conda run -n getting-over-it-analysis python -u \
  src/tests/control_interaction/test_planb_premise.py --agents 6 --horizon 150
# 短时域（多跑几次可捕捉静止姿态，验证 Fix1/2：P0a/P0b 应达 1e-4~1e-7）
conda run -n getting-over-it-analysis python -u \
  src/tests/control_interaction/test_planb_premise.py --agents 6 --horizon 30
# 连接已运行的游戏（跳过启动）：追加 --no-launch
```

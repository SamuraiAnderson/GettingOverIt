# 优化路线图

本文记录当前项目待推进的优化项，按依赖关系与性价比排序。诊断与设计原则同步给出，
所有优化项均经"缩搜索空间 vs 缩解空间"筛过，属于前者。

## 一、当前瓶颈判断

**证据链（详见 `doc/planb_premise_verification.md` 与训练产物）：**
- 精英轨迹（`logs_treefix2/elite_trajectories.png`，top-10 by base_score）全部在起点
  附近横向游走，峰值 `y ≈ 0~1`，**没有任何一条翻越第一个障碍（Deadtree, x≈-30）**。
- 效率图（`logs_start_sil/ppo_effmap_iter_0190.png`、`logs_treefix2cont/ppo_effmap_iter_0260.png`）
  的价值只在起点盆地。跑 190~260 轮 PPO 后仍未出现纵向价值传播。

**直接死因：**
- `stepFrames = 1` → 50Hz 控制；PPO 采样为逐帧 iid 高斯（`explore_noise_std ≈ 0.2`，
  action ±20 单位）。翻越第一个台阶需要跨 ~25 帧的**时序连贯"勾—拉"动作**，
  50 个独立高斯采样几乎不可能构造出来（呼应 `doc/hammer_physics.md` 的 hold 收敛 /
  violent 混沌结论）。

**系统性死因：**
- 塑形 (效率图 PBRS)、模仿 (SIL)、评分 (secured 峰) 都在**探索的下游**。探索从未撞对
  第一段真攀爬 → 上游无正样本 → 所有下游机制退化为强化"起点乱晃"。
  这是稀疏奖励 + 硬探索的经典死锁。

## 二、设计原则

**唯一准则：缩搜索空间 ≠ 缩解空间。**

| 类别 | 做法 | 举例 |
|------|------|------|
| ✅ 缩搜索空间（做） | 让同样的解更容易被找到 | 加接触信号；连贯探索先验；相位加权损失；行为多样性池 |
| ❌ 缩解空间（不做） | 直接规定解长什么样 | 硬编码"接触时沿法向施力"；喂人类演示做 MSE 模仿；奖励只在指定动作序列上给 |

**判断准则**：
1. 这个改动是让策略更容易找到好解，还是替策略决定好解？
2. 如果最优策略跟我猜的不一样，这个改动会不会阻止它出现？

## 三、优化项（按优先级）

### P0 · 思路层（不写代码）

#### P0.1 · 锁定最小任务

- **内容**：固定起点、短 episode（300–500 步）、只考核"是否翻过 Deadtree"。
  暂停 `random_deploy`，全部下游实验先在最小任务上迭代。
- **动机**：deploy 姿态不稳定 + 基座控制未通 = 双重未解叠加。**先让 agent 学会从
  起点开始的第一段攀爬**，再谈采样效率。
- **改动量**：0 代码，改 CLI（`--random-deploy` 关掉、缩短 `steps_per_rollout`）。
- **风险**：无。

---

### P1 · 便宜且高杠杆（可并行、几乎立刻能试）

#### P1.1 · 探索噪声改为时序连贯（pink 噪声）

- **内容**：`ActorCritic.sample()` 里把 iid 高斯改为 pink 噪声（推荐）或 OU / action-repeat。
  可视化对比证据见 `C:\Users\Symbol\AppData\Local\Temp\noise_demo.py`：
  - pink 噪声 lag-1 自相关 ~0.79，功率谱 ~1/f 斜率，与目标动作（低频、准周期笔画）
    的功率结构基本重叠。
  - iid 高斯 lag-1 ~-0.02，功率平摊到 25Hz Nyquist，与目标动作的低频结构几乎不重叠。
- **动机**：目标策略输出住在低频薄流形上（见 `ideal_action.py`：4Hz 以下 ~100% 功率）。
  白噪声探索几乎在这个流形外 → 撞对概率 ~0。
- **实施要点**：
  - 保持 PPO 的 `log_prob` 数学正确。pink 噪声不是逐步独立采样，需 reparameterize：
    整个 episode 的噪声在开始时一次性从 pink 分布采出，作为**外部先验加到策略均值上**，
    `log_prob` 仍按当前时刻边际分布计算；或改成 OU/AR(1) 保持 Markov（每步 `log_prob`
    可由条件高斯写出）。**OU 是最容易正确落地的第一步**。
  - 探索 std 保持当前初值即可，先只改时间结构。
- **改动量**：小（1-2 天）。
- **风险**：低（不改解空间）。
- **验收信号**：`elite_trajectories.png` 上 top-10 里出现 secured peak y > 2 的轨迹。

#### P1.2 · SIL 精英池改为"行为多样性"（MAP-Elites 风格）

- **内容**：取消现有 `add_trajectories` 的 top-K% 百分位准入，改为：
  ```
  1. 硬质量地板：is_water_trajectory 直接拒
  2. 行为描述子 bd = (secured peak x, secured peak y)   # 复用 progress_metric
  3. 离散化到网格（默认 5m × 5m）→ cell_id
  4. 若该 cell 空 或 新分数 > cell 内最优 → 替换
  5. 池 = 所有 occupied cells
  6. 冷启动兜底：|pool| < 5 时直接入池，避免全部挤在同一 cell 时池只有 1 条
  ```
- **动机**：
  - 现状 top-K% 是**相对**门槛：整批都烂时仍会入池"最不烂的乱晃"→ 自我强化平台。
  - 绝对阈值门槛需要手动调参且长期不达标会让 SIL 失效。
  - 行为多样性准入**天然回避阈值调参**，鼓励行为覆盖，多个 cell 分别保留自己内最优 →
    多样性 + 质量正交约束。QD (Quality-Diversity) 家族在硬探索问题上被反复验证有效。
- **行为描述子选择**：`secured peak (x, y)` 而非 `endpoint` 或 `mean`——前者已被
  dwell 过滤，不会被甩飞尖峰或落水回落污染。
- **改动量**：极小（`dataset.py` 的 `add_trajectories` 换实现，`trim_oldest` 弃用或
  改为按 cell iteration 老化）。
- **风险**：低（不改解空间）。
- **验收信号**：SIL 池占据的 cell 数量随迭代增长；不再出现"全池都在起点几个 cell"
  的塌缩。

---

### P2 · 中等改动、关键杠杆（P3 的前置）

#### P2.1 · 观测里加入接触信号（33D → 39D dynamics）✅ **已实施（Python 端）**

- **落地方式**：**不改 C# 端 wire 协议**，直接在 Python 端复刻游戏原生
  `HammerCollisions` + `PlayerSounds` 的接触判断（`src/training/contact_features.py`），
  从 33D 原始状态 + player_contour.json + 环境多边形几何 → 5 维派生接触信号：
  | 索引 | 字段 | 游戏原生对应（IL 反编译） | Python 复刻 |
  |------|------|--------------------------|-------------|
  | 34 | `tip_contact` (0/1) | `HammerCollisions.collisionPoints` 有 Terrain entry | tip 世界多边形 ↔ Terrain 最小距离 < 0.03m |
  | 35 | `tip_grip` (0/1) | `HammerCollisions.slide == false`（静态摩擦/钩住） | `tip_contact` ∧ v_tip<0.3 ∧ 连续 2 帧 tip 移动 < 0.03m |
  | 36 | `body_contact` (0/1) | `PlayerSounds.OnCollisionEnter2D` 里 `rb.GetPoint(contact).y > 0.4` 分支（对应 `hurtThreshold=8` pain sound） | body（`parts.body`，躯干局部 +y 在上）世界多边形 ↔ Terrain 距离 < 0.03m |
  | 37 | `pot_contact` (0/1) | `PlayerSounds.isTouching`（OnCollisionStay Terrain，pot 稳态支撑） | pot（`parts.pot`，锅底）世界多边形 ↔ Terrain 距离 < 0.03m |
  | 38 | `fall_distance_norm` | `PlayerSounds.Update` 的 `CircleCast(pot, 0.5, ↓)` | player 中心向下几何射线到 Terrain 距离，clip[0,20]/20 |
- **常量对齐游戏 IL**：`CONTACT_EPSILON = 0.03`（HammerCollisions.moveThreshold），
  `GRIP_VEL_THRESHOLD = 0.3`（OnCollisionStay 硬编码），`FALL_CAST_RADIUS = 0.5`
  （PlayerSounds.CircleCast 半径）—— 不臆造启发式，全部反编译得到（见
  `contact_features.py` 头部注释与 IL dump）。
- **body / pot 为什么要分开算**：游戏 `PlayerSounds.OnCollisionEnter2D` 里存在
  **局部 y 分层**——`rb.GetPoint(contacts[0].point).y > 0.4` 用于挑出"接触点打在
  躯干上部"分支（触发 `hurtThreshold=8` 时的 pain sound）；`y ≤ 0.4` 侧则对应 pot
  底部触地 = 稳定支撑。两者物理语义完全相反：**body 触地 → 撞头/受伤**（负面），
  **pot 触地 → 支点**（正面）。合并成单一 `pot_contact`（老 4 维方案）会让这两个
  语义相反的信号相互抵消，网络更难学；分开后模型能同时读到"挨打"与"支撑"两种
  截然不同的力学上下文。
- **`tip_grip` 是最关键的信号**：区分"锤子搁在斜坡上滑动"和"锤子钩住岩石"，
  对应游戏物理引擎 `slide` 内部状态；策略据此可判"拉力是否有效"，此前 33D 里完全缺失。
- **动机**：
  - 接触信号**从原始 33D 推不出**（网络学不到"多边形 A ↔ 多边形 B 距离查询"这类
    O(V·E) 几何算法，需替它算好）。
  - **不用 C# 端加**的理由：游戏 IL 里的接触判断本质就是"多边形是否相交"+
    "向下 CircleCast"，Python 端能精确复刻；改 C# 需重编译插件、重启游戏、破坏
    wire 协议 → 成本远大于精度收益（首版失配可再引入 C# ground truth，见下）。
  - 与当年 29D→33D 补 fakeCursor 同类改动：**补可观测性、不是喂答案**。
  - 是 P3.1（相位加权）的前置。
- **老 34D checkpoint 兼容**：`model.expand_state_dict_for_dynamics_dim`
  自动扩展 `dynamics_proj.weight` 前 34 列保留、后 5 列初始化 0；`dyn_mean/std`
  前 34 保留、后 5 pad 0/1。新特征"起手无效应"，训练中逐步被学到。已在
  `test_p2_contact_integration.py` 用 iter=260 真实 ckpt 验证通过（真实 ckpt
  dynamics_proj.weight (128,34) → (128,39) 完美迁移）。
- **可选后续**：若发现 Python 复刻的 `tip_contact` 与实际接触时刻明显偏差（视频比对），
  可在 C# 侧加独立 debug 通道 dump 真实 `collisionPoints` 作 ground truth 校准。首版无需。
- **改动量**：中（Python `contact_features.py` + `dataset.build_dynamics` +
  `rollout._compute_contact_batch` + `model.expand_state_dict_for_dynamics_dim` +
  三个单测，无 C# 改动）。
- **风险**：低（旧 checkpoint 通过 partial-load 无缝迁移；wire 协议不变；
  单测 `test_p2_contact_integration.py::test_real_checkpoint_partial_load` 用真实
  iter=260 ckpt 端到端跑通）。
- **验收信号**：加入后不做其他改动，让 PPO 跑一段，观察策略在接触帧（`tip_grip=1`
  或 `pot_contact=1`）的动作方差是否显著低于空中帧（说明模型开始利用接触信号做
  条件决策）；同时 `body_contact` 高的轨迹应快速收敛到负奖励，与 pot 支撑轨迹
  分岔（对应游戏"撞头" vs "站稳"两个物理相反状态）。

#### P2.2 · 接触条件的双输出头（Contact-conditional actor heads）

- **内容**：`ActorCritic` 保留共享 backbone（perception 层，接触/空中都要用），将 actor
  单头拆为两个：
  - `head_contact`（接触帧）：独立 `log_std`（初值更小 → 精细控制先验）
  - `head_airborne`（空中相）：独立 `log_std`（初值更大 → 粗糙探索先验）
  - 硬 gate 混合（推荐首版）：
    ```
    use_contact = (tip_contact >= threshold)
    dist = Normal(mean_contact, std_contact) if use_contact
           else Normal(mean_airborne, std_airborne)
    ```
    `tip_contact` 本身是外部离散信号，硬 gate 天然干净，`log_prob` 就是标准高斯。
  - 若后续需要平滑过渡，可换软 gate（混合高斯 `log_prob` 需显式写）。
- **动机**：
  - 接触帧与空中帧的最优动作分布物理上截然不同（精细施力方向 vs 姿态调整），
    单头共享参数 → 数量占优的空中帧梯度稀释接触帧学习。
  - 独立 `log_std` 直接编码"接触精细 / 空中粗糙"先验（缩搜索空间、不缩解空间）。
  - 与 P3.1 相位加权协同：P3.1 从损失权重推动分化，P2.2 从架构上分开参数——两者叠加。
- **依赖**：P2.1（无接触信号无法 gate）。
- **改动量**：小（`ActorCritic.forward` / `sample` / `log_prob` / `evaluate_actions`
  统一走 gate 分派）。
- **风险**：低（架构层面小改，不缩解空间）。
- **验收信号**：训练后 `log_std_contact` < `log_std_airborne`（模型学到"接触时收紧
  探索、空中时放开"的先验）；接触帧策略熵显著低于空中帧策略熵。

---

### P3 · 中等改动、依赖 P2

#### P3.1 · 相位加权的训练信号

- **内容**：把 PPO `step_reward` 与 SIL `MSE loss` 都按相位重要度加权：
  - `relevance(t) = high` 在接触前后 K 帧（K ~= 5-10）
  - `relevance(t) = low` 在纯空中飘荡帧
  - PPO：`reward(t) *= relevance(t)`（PBRS 势场保持不变，只调进度通道）
  - SIL：`_rebuild_index` 里过滤或 loss 权重降低空中帧
- **动机**：GOI 物理上极端不均匀——接触相是高杠杆决策，空中相基本由脱离瞬间的动量决定。
  当前 50 帧均匀加权把接触帧信号稀释在空中噪声里。
- **实施顺序**：先用**启发式相位定义**验证概念（例如 tip 距最近地形距离 < 阈值判为接触
  相位），再换成 P2.1 的真接触信号。
- **改动量**：中（`reward.py` + `dataset.py` + `ppo_trainer.py` 各改一处）。
- **风险**：低。
- **验收信号**：训练后期策略在接触帧的动作幅度更大/更结构化，空中帧的动作幅度自然衰减
  （模型学会"接触相发力、空中调姿态"）。

#### P3.2 · SIL 轨迹内 advantage 加权

- **内容**：现状 `_rebuild_index` 里 `for tau in range(peak+1)` 每步等权模仿。改为按
  PPO 的 advantage（或简化的 return-to-secured-peak）给每步加权，只对高优势子段做强
  模仿。
- **动机**：即便撞对一段真攀爬，其前面的乱晃前缀也被复刻。等权模仿稀释了信号。
- **改动量**：中（`ppo_trainer.sil_update` 里改 MSE 权重）。
- **风险**：低。
- **验收信号**：SIL 池里同一条轨迹的高 advantage 步比低 advantage 步在模型输出上被
  更准确复现。

---

### P4 · 大改动、长期方向（前置实验站住脚后再考虑）

#### P4.1 · 动作空间重参数化到"笔画基元"

- **内容**：网络输出从每帧 `(dx, dy)` 改为每 K 帧一组笔画参数（频率/幅度/相位/朝向），
  中间由参数化过程生成 K 帧动作。
- **动机**：目标动作住在低频薄流形上（`ideal_action.png`：4Hz 以下 ~100% 功率）。
  50Hz 决策频率极大浪费搜索预算。
- **前置条件**：P1.1（连贯噪声）需先看到收益，证实"低频动作"方向有效。
- **改动量**：大（动作接口 + 训练循环 + 观测窗口对齐都要改）。
- **风险**：中——若基元集不够丰富，可能滑向缩解空间。**要严格验证：任何最优策略仍可
  在基元空间内表达。**
- **验收信号**：在最小任务上，笔画参数化的样本效率显著高于逐帧动作。

#### P4.2 · 方案 B 短时域 MPC 搜索作为探索引擎

- **内容**：利用已验证的近似世界模型（P0b ≈ 1e-4，短时域可用；见
  `doc/planb_premise_verification.md`）做 CEM / random shooting 在游戏里搜索发现
  第一段真攀爬。搜索出的精英轨迹进 SIL 池。
- **动机**：单纯 PPO 探索点不着火时的兜底。**属于自发现（不是人类演示），
  完全符合"探索学习、不靠模仿加数据"的哲学。**
- **前置条件**：P1（连贯探索）+ P2/P3（信号对齐）跑通后如仍未点火。
- **改动量**：大（造 planner + 集成到 rollout）。
- **风险**：中（需要短时域策略/成本函数设计）。
- **验收信号**：搜索能发现 secured peak y > 3 的轨迹，且蒸馏后策略可复现。

---

## 四、已考虑并暂缓的方向

以下想法讨论过但**当前决定不做**，此处记录以避免后续重复论证。

- **完整 Feudal HRL（高层目标网络 + 低层执行网络）**：经典鸡生蛋——低层未训好时
  高层收到的是乱码，反之亦然；两头稀疏奖励叠加。**P4.1（笔画参数化）已用"确定性
  参数化过程"替代可学习低层**，占据了纵向抽象的安全形式。等 P4.1 落地后仍有能力
  上限问题再评估完整 HRL。
- **Mixture-of-Experts 技能拆分（K 个技能 + gate 网络）**：MoE 的 gating 训练本身
  是独立难题（模式塌缩、负载失衡）；当前不是模型容量瓶颈，是探索/信号问题。
- **学习式 world model 做规划**：被 P4.2（游戏本身当近似世界模型）严格支配——游戏是
  ground truth，学出来的模型不可能比它更准。
- **多 seed 并行挑选**：不是拆分，是 diversification；等价于当前多 agent 并行采样，
  边际收益有限。
- **每相位独立 Critic（V_contact / V_airborne）**：是 P2.2 的小弟版；可作为 P2.2 的
  ablation 验证，但不作为主线。

---

## 五、建议的第一个 sprint（本周）

按依赖与性价比：

1. **P0.1**（不做 deploy 的最小任务锁定）——0 代码，规则性变化。
2. **P1.1**（OU 或 pink 噪声）——最贴合诊断；先从 OU 起（`log_prob` 好推），再看 pink。
3. **P1.2**（行为多样性 SIL 池）——极小改动，防止 P1.1 有一点点起色时被 SIL 拉回平台。

跑 20-50 轮 iter，主观测：
- `elite_trajectories.png` 里三角形（peak）分布——是否开始出现 y > 2 的样本？
- SIL 池占据的 cell 数量——是否随迭代增长？

**如果 peak 开始出现 y > 2** → 第一个信号灯亮，P2/P3 沿路开展。
**如果 peak 依然全在 y ≈ 0** → 连贯探索本身不够，直接进 P4（重参数化或搜索引导）。
这个信号本身就是有价值的答案。

## 六、进度追踪（更新时补此表）

| 项 | 状态 | 备注 |
|---|---|---|
| P0.1 | ⬜ 未开始 | CLI 侧改动，跑时 `--no-random-deploy --steps-per-rollout 400` |
| P1.1 | ✅ 已实施 | OU AR(1)，`config.ou_enabled/ou_phi`；PPO ratio 数学自洽（buffer 存 `noise_prev`，new/old 用同一条件式）；episode 起手 `z_prev~N(0,1)` 保首步 marginal |
| P1.2 | ✅ 已实施 | MAP-Elites，`config.sil_map_elites_enabled/sil_cell_size/sil_cold_start_min`；bd=secured peak (x,y)；冷启动用 extras 允同 cell 填池防塌缩；`main_ppo` 日志 `[SIL/MAP-Elites] cells=` |
| P2.1 | ✅ 已实施 | Python 端复刻游戏 `HammerCollisions`+`PlayerSounds`，`DYNAMICS_DIM 34→39`；5 维派生：`tip_contact`/`tip_grip`/`body_contact`/`pot_contact`/`fall_distance_norm`（body / pot 按游戏 IL 里 `rb.GetPoint().y > 0.4` 分层各自输出，语义相反）；老 34D ckpt 自动 partial-load；wire 协议不变；见 `contact_features.py` + `test_p2_contact_integration.py` |
| P2.2 | ⬜ 未开始 | 依赖 P2.1 |
| P3.1 | ⬜ 未开始 | |
| P3.2 | ⬜ 未开始 | |
| P4.1 | ⬜ 未开始 | 等 P1.1 结果 |
| P4.2 | ⬜ 未开始 | 兜底方案 |

### 6.1 已实施项的落地要点

**P1.1 · OU 探索噪声**
- 配置：`config.ou_enabled=True`（默认开），`config.ou_phi=0.85`（lag-1 目标）。
- 关键文件：`actor_critic.py`（`_ou_conditional` + `get_action_and_value/evaluate_actions` 双向接 `noise_prev, noise_phi`）、`ppo_buffer.py`（新增 `noise_prev` 字段）、`rollout.py`（每 agent 维护 OU 状态 `ou_state[i]`，起手 `~N(0,1)`）、`ppo_trainer.py`（update 时透传 `batch.noise_prev, ou_phi`）。
- 数学：raw_t | z_{t-1} ~ N(mean + std·φ·z_{t-1}, std²·(1-φ²))。new/old policy 对同一 z_{t-1} 计算条件高斯 → ratio = exp(new_lp - old_lp) 无系统偏差。
- 回退：`ou_enabled=False` 或 `ou_phi=0.0` 完全退化为原 iid 行为；`noise_prev` 默认存零向量，evaluate 端在 phi=0 时忽略。

**P1.2 · MAP-Elites SIL 池**
- 配置：`config.sil_map_elites_enabled=True`（默认开），`config.sil_cell_size=5.0` m，`config.sil_cold_start_min=5`。
- 关键文件：`dataset.py` 新增 `add_trajectories_map_elites()` 与内部 `_behavior_cell_id()`；`main_ppo.py` 分支切换（旧 `add_trajectories(keep_ratio)` 仍留在 `main_train` BC 路径不受影响）。
- 逻辑：is_water_trajectory 硬拒 → 计算 progress_metric 的 secured peak (x,y) → 网格化 cell_id → cell 空 or 新分数 > cell 内最优 → 入池；冷启动阶段 (|cells| < cold_start_min) 同 cell 落选轨迹进 extras 直接入池，防止塌缩到 1 条；cell 数达标后清空 extras。
- 兜底：MAP-Elites 之后仍按分数 top-N 裁剪到 `sil_traj_cap`，防止 cell 数长期无界增长。

**首个 sprint 建议命令**（对应第五节；完整 CLI 见 [`main_ppo_usage.md`](main_ppo_usage.md)）：
```bash
# P0.1（无代码）+ P1.1（默认开）+ P1.2（默认开）
python -m src.training.main_ppo \
  --no-random-deploy \
  --steps-per-rollout 400 \
  --max-iterations 40 \
  --checkpoint-dir checkpoints_ou_me \
  --log-dir logs_ou_me \
  --persist-game \
  --resume-bc <某 BC ckpt>
```
观察：`logs_*/ppo_effmap_iter_*.png` 精英三角形 y 分布 + 训练日志 `[SIL/MAP-Elites] cells=` 随迭代的增长。

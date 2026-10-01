# 最后有效入账位置：完整修改计划

日期：2026-09-27。v2 已进入后续训练。文中的实验控制器已删除。

后续采样修订（2026-09-28）：见 [SIL 精英池采样修改计划](sil_td_sampling_plan_2026-09-28.md)，第 0 节记录实际首动作采样缺陷及评估证据。该计划的两个新采样模式替换本文件第 5 节的单一 D 池侧门控；旧采样模式继续保留原规则。t、D、cap 和 v2 奖励/回报定义不变。

目标：用“最后一个有效入账动作后的累计路线进度 D”替代累计 credited，作为轨迹的距离指标；PPO 使用 D 的逐步增量；SIL 筛选、优势回报和 critic 目标同步迁移。

**1. 先固定语义和版本**

- 新模式命名 `confirmed_endpoint_v2`；旧模式 `credited_sum_v1` 保留用于历史恢复和对照。新增 `reference_progress_mode`，新建走廊任务默认 v2；旧 checkpoint 缺字段时明确识别为 v1，禁止因新默认值而静默改变旧任务。
- 版本变更覆盖距离、入账动作定位、奖励、SIL 回报和评估语义。相同路线文件、相同网络形状仍不代表奖励兼容。
- 本次不改游戏协议、状态宽度、锚点、观测维度、路线文件、走廊半宽或网络结构。正常沿走廊推进时继续提供稠密奖励。
- 实施前确认没有运行中的任务依赖将修改的训练源码；保留已有未提交改动。新实验写新目录，冻结新的源码和配置清单，不覆盖旧实验。

**2. 定义 D、有效入账事件和 cap**

令动作 `a[t]` 把 `state[t]` 推进到 `state[t+1]`。

- `legacy_credit[t]`：现有 tracker 计算的置信加权新增进度，保留作资格依据和诊断。
- `credit_event[t]`：本动作符合原入账条件且 `legacy_credit[t] > 0`。保持原有双端在走廊内、超过历史最高进度、非跳变等条件。用 tracker 的 float64 结果生成布尔事件，避免 float32 存储精度影响 cap。
- `s[t+1]`：结果状态在有序参考路线上的进度，扣除本回合初始投影。跨段时累计已知参考段的弧长；未知缺口不补连线距离。等待缺口时使用实际保存的投影，不能自动填到段末。
- `D[0] = 0`；有 `credit_event[t]` 时 `D[t+1] = s[t+1]`，否则 `D[t+1] = D[t]`。
- `delta_D[t] = D[t+1] - D[t]`。正常数据中应非负；明显负值视为语义或数据错误，不能用随意截零隐藏问题。只允许明确的浮点误差容差。
- `cap = max(t where credit_event[t])`；结果状态索引为 `cap + 1`；轨迹距离 `cap_progress_m = D[T]`；有效时间 `t_cap = cap + 1`，单位为物理步。
- 无入账：`D = 0`，cap 明确为无效值（建议 -1），不制造“第 0 个动作有效”的含义。无入账轨迹保留评估记录，但不进入 SIL 前沿和监督窗口。
- v2 的 cap 取严格最后一个事件，不沿用累计和达到总量 `1-1e-6` 的旧近似。v1 保留原算法以复现历史结果。所有消费者通过同一版本化辅助函数取 cap，禁止局部复制。

跨段坐标计算：若起始段为 `j0`、起始投影为 `p0`、当前段为 `j`、段内实际投影为 `p`，则 `s = sum(length[j0:j]) + p - p0`。同段就是 `p-p0`。这个坐标会在下一次入账时包含此前跳过的已知段尾部，但不会计算未知缺口的欧氏距离；它表示路线上的到达位置，不是亲自走过的合格路径长度。

明确接受的变化：走廊外绕行的已知路线距离、正置信权重小于 1 的距离，都可以在后续有效事件时计入 D。置信权重仍影响事件是否成立，正权重不再缩小 D 的幅值。

**3. Tracker 与逐步奖励**

主要文件：`src/training/reference_corridor.py`、`src/training/reward.py`、`src/training/config.py`。

- Tracker 保存起始段/起始投影、当前确认距离 D、最后入账动作索引和轨迹有效性。
- 扩展 `CorridorStep`，分别输出旧 credit、入账事件、确认距离、新距离增量、偏离惩罚；避免一个 `progress` 字段同时表示两种计分。
- 原有高水位机制保持：走廊外推进仍消耗历史最高位置，回到已达位置不会重复触发入账。
- v2 奖励：`r[t] = reference_progress_weight * delta_D[t] - deviation_penalty[t]`；落水继续由 `water_penalty` 覆盖当步奖励。
- 静止、回退、返回旧最高点：delta_D 为零；无入账的缺口帧 D 冻结。奖励在每个物理步计算，不等回合结束，不调用依赖未来轨迹的 cap 计算。
- 跳变检测放在缺口分支提前返回之前，避免 gap 状态漏检。正式采集过程中超过既有位移阈值的异常跳变使该 agent 本条轨迹失效，剔除其 PPO/SIL 样本并记录原因；离线重放同样拒收。下一次正常 reset 建新基线。这样异常位移不会在后续小额入账时补奖。合法部署发生在 tracker 创建之前。
- 正常早停、落水与数据异常分开记录；不把异常跳变伪造为落水，也不对无效样本施加任意任务奖励。

验收恒等式：有效轨迹 `sum(delta_D) = D[T]`。实际奖励总和还包含偏离惩罚和落水覆盖，折扣回报也不等于 D，不能混称。

**4. 轨迹字段、在线采集、保存和重放**

主要文件：`dataset.py`、`rollout.py`、`experimental_rollout.py`、`run_logging.py`、`reference_corridor.py`。

建议字段契约：

- 原 `corridor_progress[T]` 保留旧 credit 的含义，作为兼容诊断；v2 的距离、奖励消费者不再直接对它求和。
- `corridor_credit_events[T]`：布尔入账事件。
- `corridor_confirmed_progress[T+1]`：状态对齐的 D，首项为 0；累计量使用 float64。delta_D 由相邻差分得到，不再保存第二份可不一致的主数据。
- `step_rewards[T]`：Python 实际交给 PPO 的逐物理步奖励，含权重、偏离惩罚和落水覆盖。
- `terminated[T]`：真实终止标志，与 rollout 时长截断、SIL 物理剪枝区分。
- 保留 `guidance_index[T+1]`；增加轨迹有效性、奖励版本及原轨迹/剪枝来源信息。

在线两条采集路径每步只推进 tracker 一次，从同一个 step 结果写入上述字段。普通 PPO buffer 和动作分段 buffer 使用同一个实际奖励值。

动作分段模式继续用 `sum(gamma**j * r[j])` 汇总段内奖励，并沿用按物理时长计算的 GAE；不把段奖励再差分或重复折扣。本次不扩大现有实验模式的 SIL 支持范围。

`replay_corridor_fields` 扩展为结构化、版本化重放结果，在线和离线调用同一核心算法。完整状态、动作、终止信息可重建实际奖励；无法恢复的原生 done 不能猜测。

同步修改所有 Trajectory 构造、剪枝、NPZ 保存/加载、SIL seed 和 resume pool 入口。首版不需要改变网络输入张量。

**5. 轨迹距离、排序、归档与行为门控**

主要文件：`sil_archive.py`、`reward.py`、`main_ppo.py`、`dataset.py`、`ppo_trainer.py`。

- 建统一的轨迹结果辅助函数，返回有效性、cap、t_cap 和 D；有效性和字段一致性只在此处判定。
- `td_point()` 改为 `(cap+1, D)`。Pareto 支配、去重、距离分桶、容量压缩和行为多样性补充都消费同一结果。
- `corridor_score()` 改为 `climb_score(D, cap+1)`，调整其仅接受旧 credit 数组的接口及 `_sil_score()` 调用。
- MAP-Elites 行为位置仍使用 cap 动作后的结果状态；SIL/BC 监督窗口覆盖 `0..cap`；保留现有 cap 后 dwell 观察窗。v2 全零轨迹不生成虚假首动作监督。
- 当前策略侧：从本轮完整、有效采集轨迹计算 D 的平均值；无入账或落水轨迹仍计入其真实 D，不能先按精英条件筛选。部署失败、异常数据单列，分母为实际有合法采集的 agent 数；若没有合法轨迹，本轮不更新。
- td_pareto 池侧：候选 `D > behavior_D_mean * (1 + sil_gate_margin)`。
- MAP-Elites 池侧：池内 D 的中位数与同一个 behavior_D_mean 比较。
- 停滞检测改为池内 max D；新旧值均记录版本与名称，禁止混用旧累计 credit 和新 D。
- 入池前、剪枝后、落盘重载后，cap、D、评分、门控决定必须一致。新旧模式不能混在同一精英池。

**6. SIL 优势回报与 critic 目标一起修改**

主要文件：`ppo_trainer.py`、`dataset.py`；相关测试 `test_sil_advantage_weighting.py`。

现有 `_sil_pool_values()` 用旧 credited 序列计算回报，再用最后一个监督窗口的 V 替换尾部。本次必须替换这条路径，避免末次关键奖励被自举边界吞掉。

- SIL 使用与 PPO 一致的 `step_rewards`，包含新 delta_D、奖励权重、偏离惩罚和落水覆盖；不只替换一个累计距离。
- 对保留到状态 `s[L]` 的轨迹，计算 `R[t] = sum(k=t..L-1, gamma**(k-t) * r[k]) + gamma**(L-t) * V(s[L])`。
- 真实终止时尾部 V 为零；因采集时长或物理剪枝结束时，从保存的末端结果状态自举。遇到 terminated 时停止累计，不跨终止传播。
- cap 动作的奖励必须包括在 R 中；监督窗口的最后一个起始状态与自举终点不是同一个索引。复用现有观测构建原语构建 `s[L]` 的窗口，保持动作历史/patch/锚点/掩码一致。
- cap 后仅“进度增量”为零，偏离惩罚仍可非零；保留观察窗里的实际奖励必须参与回报，不沿用“cap 后奖励全零”的简化。
- `A = R - V`、正优势筛选、归一化和限幅继续使用现有方式。`sil_value_coef` 对应 critic 目标使用同一 R。R 的 bootstrap 部分随当前 critic 重算，不永久缓存旧 V。
- 保留现有 SIL 更新预算、动作目标限幅和其他已启用防线。
- 剪枝保持 D/评分/门控一致；由于自举边界改变，SIL R 与完整轨迹的 R 不保证严格相等，不能写成错误的验收条件。

**7. 评估、可视化和日志**

主要文件：`local_evaluation.py`、`run_logging.py`、`render_episode_motion.py`、`render_policy_window.py`、`plot_sil_td_envelope.py`。

- 新主指标使用 `cap_progress_m`，汇总均值、中位数、Q25 和最大值。旧指标明确标成 legacy credited 诊断，历史文件不改名重写、不伪装成新结果。
- 逐帧 trace 保存当前 D、delta_D、credit_event、实际路线投影、段索引和实际 reward，回合结束给出 cap 和 t_cap。
- 到达判定使用当时的 D 与当时真实投影，不能把全轨迹最终 D 倒填到早期帧，避免未来信息泄漏。
- 保持判据、期末投影、落水、走廊外比例继续独立统计；缺口仍是参考无法确认。新 D 不是“最终守住的位置”，不能因此宣称保持能力提升。
- 速度使用 `D/(cap+1)`，物理时间单位沿用现有采集契约。图表、排行榜、Pareto 图和报告标题同步更新。
- 增加补记事件的次数、单步 delta_D 分布、最大补记量、异常轨迹数，便于发现奖励尖峰和错误投影。
- 到达率的定义已经改变，旧报告中的到达率需重评后才能与 v2 对照。

**8. 历史数据、指纹、checkpoint 和恢复**

主要文件：`reference_fingerprint.py`、`main_ppo.py`、`run_logging.py`、`src/tests/analysis/anchor/migrate_reference_archive.py`。

- 指纹纳入 progress_mode、cap 规则、奖励规则和数据 schema 版本，继续包含路线、锚点、宽度、置信等已有配置。新增语义必须写入 checkpoint，恢复检查不能只比较路线 SHA256。
- 修正种子加载入口对“带 corridor_progress 的校准轨迹”一概拒绝的判断，改成依据指纹和 schema 校验；既能拒绝旧口径，也能接收合法 v2 重放数据。
- 同版本正常 resume 保持原逻辑。v1→v2 首次切换走明确暖启动：复用兼容策略/骨干和探索参数，重新初始化 critic head，重新建立优化器并进行 critic-only 预热，不沿用旧 critic 目标与动量。
- critic 预热沿用冻结骨干和 actor、仅训练 critic_head 的路径。新增模式必须触发预热；检查 checkpoint 保存/恢复预热阶段的行为一致。
- 旧原始轨迹可离线重放生成 v2 D、奖励、cap、评分与归档；输出新文件和迁移报告，不覆盖来源。迁移工具支持单轨迹 NPZ 以及 `t{i}_*` 精英池 NPZ 两种格式。
- 只有旧 credit 没有完整状态/动作的数据，不能转换成 D。已剪枝数据只能重评现存前缀，标记 partial，不能恢复丢失的末次微量入账和尾段。
- 终止信息不足时保留评分/评估用途；在明确恢复终止语义之前，不把该数据作为可信 SIL critic 监督。
- 原有 33D/35D 锚点约束继续有效，不为迁移虚构锅心坐标。
- 旧 PPO buffer、advantages、returns 不能直接拿来训练新目标；新版本训练采集新 on-policy rollout。旧精英轨迹只作为重评分后的 SIL 数据，不当作新 PPO 样本。

**9. 必须完成的验证**

全部 Python 验证使用 `conda run -n getting-over-it-analysis python ...`。

- Tracker：直线推进、静止、回退、再超越、离开/重入、置信 1/0.5/0、缺口与跨段、非零起点、异常跳变和 gap 中跳变。
- 明确数值例：外部到 2m、回廊再到 2.5m，旧 credit 为 0.5m，新 D 为 2.5m；同一轨迹在线和离线结果一致。
- cap：空轨迹、全零、末尾微小正入账、float32 存储、动作/结果状态差一位、多 agent 隔离。无入账不得变成第一步位置。
- 不变量：D 单调，sum(delta_D)=D_end；剪枝/保存/加载不改变 D、cap、score 和门控；旧 v1 回归测试保持可复现。
- 奖励：权重、偏离惩罚、落水覆盖逐步一致；两条 rollout 路径一致；动作分段的物理步折扣无重复和遗漏。
- SIL：池内与行为侧同一 D，零进展计入行为均值；构造排序反转轨迹验证两种池；末次入账奖励完整进入 R；真实终止无自举，截断/剪枝自举位于结果状态；cap 后惩罚计入；价值目标和优势权重引用同一 R。
- 持久化：单轨迹和精英池 round-trip，v1/v2 混用拒绝，显式迁移、partial 标记、缺终止信息、缺锚点数据、旧 checkpoint 缺版本、同版本恢复和跨版本暖启动。
- 评估：trace 不含未来信息、跨段不变负、gap 不被直接计作到达保持、排行榜和训练门控使用相同端点。

优先扩展现有 `test_reference_corridor.py`、`test_prune_trajectory.py`、`test_sil_archive.py`、`test_sil_advantage_weighting.py`、`test_corridor_success.py`、`test_reference_fingerprint.py` 及对应 rollout/decision buffer 测试。

**10. 实施顺序和实验验收**

1. 先实现版本和 tracker/replay 的 D、事件、cap 契约，完成核心离线测试。
2. 接入奖励、两个 rollout 路径与完整轨迹字段，完成实际奖励一致性和持久化验证。
3. 迁移评分、归档、门控、窗口与停滞检测；同步实现 SIL 实际回报和正确自举终点。
4. 更新评估、日志、迁移工具、checkpoint 恢复与文档，验证所有 v2 消费者不再误用 sum(corridor_progress)。
5. 用现有原始轨迹重评分，检查排序反转、补记幅度和 cap 分布。这只验证定义与数据，不证明训练效果。
6. 实施阶段另行运行短实机 smoke：确认 finite 奖励/损失、有效样本数、cap/replay 一致、预热时策略不变、SIL 路径可执行。
7. 对照实验：A 为完整 v1，B 为完整 v2；从同一个策略初始化开始，均使用各自奖励预热 critic，同样的物理样本预算、评估扰动和种子。首轮不同时改变 gamma=.99、lambda=.95、学习率、OU、走廊宽度、网络结构或 SIL 预算。使用多个配对训练 seed；单 seed 只作初步观察。
8. 两组评估都用同一个 v2 评估器重评，同时保留旧 credit 诊断。主要比较 D 中位数/Q25、到达和速度，独立核查落水、期末投影、保持表现、补记奖励和 SIL 正优势比例。D 上升不能单独证明实际控制更好。

代码完成条件：新口径从在线奖励到离线 SIL/评估一致，边界测试通过，旧数据不会静默混用，新 checkpoint 能正确恢复。训练有效性是后续实验结论，不能由工程测试代替。

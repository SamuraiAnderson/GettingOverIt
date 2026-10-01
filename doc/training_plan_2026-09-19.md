# 轨迹走廊训练计划：首轮 6 小时筛选

日期：2026-09-19。用户确认总墙钟预算约 6 小时，先筛选，再决定扩训。本文保留原执行方案；训练已于当日 15:11 完成。五组约 48k 筛选后，B1/R 各两个训练 seed 完成约 72k 复验；因任务派发空档，原 96k 目标按共同预算降级。执行差异和最终结果见[训练筛选结果](../runs/corridor_screen_20260919/RESULT.md)。

启动前离线检查已通过：五组训练参数均能解析，五种模型均能按各自迁移规则加载共同源权重；四段 PowerShell 模板语法检查通过，训练/评估样本预算已复算。以下预算为原计划，实际样本以执行记录为准。

目标是在已有正确性验收基础上，找到值得追加预算的局部控制候选。6 小时包含采集、PPO 更新、评估、存档和故障处理，不承诺学会局部任务。实现依据见 [表示实验](policy_representation_experiments.md)、[走廊奖励](reference_corridor_reward.md)和[优化清单](optimization_priorities_2026-09-16.md)。

## 1. 固定任务和成功口径

- 固定游戏起点，关闭随机投放；参考路线、半宽 1.5 米、奖励系数、碰撞数据及原生姿态标定在本轮冻结。记录代码差异及路线/地图 hash。
- 主要指标：**无噪声策略的局部成功率**。同一有效连续段内，累计有效进度至少 5 米、当前投影也前进至少 5 米，并在走廊内连续保持 10 个物理步；缺口不算成功。
- 每回合最多 **1,200 个物理步，约 10 秒游戏时间**。训练和评估一致。现有 96 步 smoke 仅约 0.8 秒，400 步约 3.3 秒，不用其零成功率推断充分训练后的能力。
- 沿用 10 步保持条件以保留可比性，但它仅约 0.083 秒，不能称为稳定站住。另从评估 trace 统计达标后连续保持 120 步（1 秒）的比例，以及首次成功后是否跌落；不另外启动采样。
- 辅助指标：每回合有效进度的中位数/下四分位数、末帧实际前进距离、跌落率、走廊外时间比例、成功回合到达时间、每秒实际物理样本数。累计奖励和历史最高进度不替代成功率。

先用预检轨迹确认 5 米目标没有触及参考缺口。若目标或 10 秒时限明显不适用，停止本轮比较，统一修订协议后重开；不能只给某个候选放宽条件。参考路线目前仍是含缺口的粗略引导，沿线进步不等于游戏通关。

## 2. 五个候选，按相邻差异解释结果

所有候选都使用逐步二维动作 H=1、单 actor head、固定 PPO+OU（phi=0.85），关闭 SIL、停滞加噪和阶段/优势重加权。

1. **B0：修复后的原表示基线。** Transformer，39D 动力学，原动作历史尺度。
2. **B1：只归一化动作历史。** 在 B0 上将历史动作除以 100；其余不变。B1 对 B0 检验尺度改动。
3. **R：增加关系特征。** B1 + 16D relations。R 对 B1 检验 P1.3。
4. **E：增加直接事件输入。** R + 24D 原生 events，直接投影，不开事件 MLP。E 对 R 检验事件信息。
5. **G：切换 GRU。** 与 E 完全相同的输入、探索和逐步动作，只将主干改为 CNN+GRU。G 对 E 检验当前迁移方案的学习/耗时表现。

首轮不同时加入 geometry、trends、事件 MLP 或 H=4。这些功能已经工程验收，但一次合并会失去明确对照。若某个中间候选较差，仍保留相邻对照数据，不把后续组合提升全部归给一个组件。

统一从 `runs/p2_contact/checkpoints/ppo_iter_0335.pt` 用 `--init-from` 迁移兼容权重，重建优化器，从第 1 轮计数；不从各自 smoke 的已更新权重开始。该源文件已核对为单头 checkpoint，没有保存 config，必须显式指定 `native-pose-v1` 和走廊。

这比较的是“现有模型如何继续训练”。GRU 是新初始化主干，而 Transformer 有旧权重；动作历史归一化也会改变初始行为。因此必须保存每个候选的训练前评估，不能据此宣称 GRU 本身优于或劣于 Transformer。新 seed 是训练随机性的重复，仍共享同一个历史预训练源。

源 checkpoint 的两个 actor log_std 约为 -0.0069、-0.0353，本轮保持其迁移值和统一上限 0；不要误以为 `--init-log-std -1.6` 会覆盖迁移进来的参数。先观察探索与饱和分布；需要调探索幅度时，另立受控实验。

## 3. 共同配置与预算单位

- **2 agents × 1,200 步/轮**，每轮最多 2,400 个 agent 物理样本。保持已验收的 agent 数，本轮不同时搜索并行度。
- 学习率 `1e-4`，entropy coefficient `0.001`，gamma `0.99`，GAE lambda `0.95`，clip `0.2`，PPO epochs `4`，batch size `64`，target KL `0.03`。不为了跑满 epochs 放宽 KL 早停。
- 每次新建训练先做 **2 轮 critic 预热**，最多 4,800 样本，计入预算；之后更新 actor/backbone。续训不重新预热。
- GRU 每块 16 个连续决策，按当前实现重算完整前缀；1,200 步回合下的更新成本必须实测，不能用 96 步 smoke 或纯推理快 1.76 倍外推。
- 本机 `stepFrames=1`，`feature_step_seconds=1/120`。两者本轮不变。gamma=0.99 的折扣尺度约 100 步（0.83 秒），与 10 秒回合不同；先保持可比性，若长程信用成为瓶颈，再单独试折扣参数。

训练 seed 首轮为 **7**；复验为 **17**。同一候选的后续扩训沿用其 checkpoint，不能把新 seed 的训练接在 seed 7 权重上再称独立复验。并行 agent 不是独立训练 seed。

样本计数优先累计 `rollout_environment_steps`；原 B0 没有该字段时，H=1 可使用 `buffer_size`。动作段实验将来仍按物理样本计数。预热计入训练，评估单列。提前落水会让实际样本少于“轮数 × 2,400”；必要时补整轮，目标允许最多一轮超额，并报告真实横轴和终止次数。

## 4. 六小时的执行次序

### 0–20 分钟：较长回合预检

先运行 B0、G 各 3 轮（两轮预热、一轮真实 PPO），最多共 14,400 训练样本，计入后续预算。保存完整 rollout 和采样前 behavior checkpoint；检查冻结概率审计、输入尺度、原生数据对齐、1,200 步回合的更新耗时和存储量。

预检通过的权重继续使用，不丢弃这部分预算。若 G 的完整前缀更新使本轮无法完成，记录为成本受限，将剩余预算转给 Transformer 候选，不直接判定其学习能力差。不能仅对 G 缩短回合或改变 PPO 配方后继续原对照。

### 20–150 分钟：五候选单 seed 筛选

每候选 seed 7 训练到 **48,000 物理样本**，五组合计目标 240,000。完整回合时约 20 轮。采用统一的 3 轮预检 → 20 轮续训分段；预检已完成的 B0/G 不重跑。

每组评估两个时间点：

- 训练前：评估保存的 `behavior_0001.pt`，2 rounds × 2 agents，**每种模式 4 回合**。
- 48,000 样本附近：4 rounds × 2 agents，**每种模式 8 回合**。

评估入口会同时跑 deterministic 和 exploration 两种模式。这里两模式总计最多 144,000 物理样本，不是零成本。训练前必须使用真实采样前 checkpoint；不能让 `local_evaluation --init-from` 用另一个 seed 重新初始化 GRU，当作同一个初始策略。

按确定性成功率优先，随后看进度中位数、跌落率和耗时，选两个候选进入复验。8 回合很少，只作筛选：差异在一两次试验范围内时视为不确定，优先保留直接对照和较简单候选；探索模式只作诊断，不与确定性成功率混合。

### 150–265 分钟：两候选复验

- 两个入选候选的 seed 7 从 48,000 继续到 **96,000**：新增目标 96,000 样本。
- 两个入选候选分别用 seed 17，从共同旧源重新初始化，训练到 **96,000**：新增目标 192,000 样本。
- 对两个 seed 17 的真实训练前 checkpoint 各做一次每模式 4 回合评估：最多 19,200 评估样本。

完整计划训练目标合计 **528,000 物理样本**，包含预检和 critic 预热。96,000 是本轮复验预算，不是已知的收敛门槛。阶段内留存中间 checkpoint，但不反复挑选偶然最好的一次评估结果。

### 265–335 分钟：保留评估集与结论

对 2 个候选 × 2 个训练 seed 的 **96,000 样本端点**进行评估，每组 5 rounds × 2 agents，**每种模式 10 回合**。使用与筛选阶段不同的评估 seed，最多 96,000 评估样本。先固定端点，再看结果，避免用保留集挑 checkpoint。

报告每个 seed 的确定性/探索成功率、进度、跌落率和耗时，保留逐回合结果。若候选在两个 seed 都有方向一致的改善，可建议追加预算；本轮每候选只有两个训练 seed、每 seed 每模式 10 回合，不作“稳定学会”或统计显著性的结论。

### 335–360 分钟：预留与硬截止

预留 25 分钟给慢更新、启动/复位、存档和故障。预测剩余工作超时就停止添加候选或 seed；优先保住已开始对照的成对终点评估。缺失结果标记未完成，不能把被截断的短训练与完整训练直接排名。

当前 CLI 没有按累计物理样本或墙钟自动停止的选项；`--max-iterations` 是迭代上限。执行时需在轮边界监控实际计数和剩余时间，分段设置上限，并为当前轮和收尾留时间。本文没有部署无人值守的六小时定时任务。

## 5. 预算核算与降级规则

完整方案为 528,000 训练 + 259,200 评估 = **787,200 物理样本上限附近**，另有复位/warmup 物理推进与落水后的轮次差异。按已有短跑采集速率 55–59 样本/秒，采集换算约 **3.71–3.98 小时**；暂加 25% 更新/运维余量约 **4.63–4.97 小时**。

这是初始排程估计。55–59 是带原生诊断的短回合采集实测，既不含完整长回合更新成本，也不能保证所有候选同速。预检后使用各候选的“采集 + 更新 + 存档”实测值重估，另加独立评估时间，以 6 小时为上限。

若超预算，按顺序缩减：先推迟昂贵的 G；再只对一个最佳候选及其直接父对照复验；最后降低所有参与复验候选的共同目标样本数。不要删除评估来换取更多训练。未获得重复改善时，结束本轮并检查奖励、探索、初始策略和回合长度，不自动追加几十万样本。

## 6. 评估和停机条件

筛选评估 seed 固定 `20260919`，保留评估 seed 固定 `20261019`；初态位置扰动 `jitter=0.01` 米。所有候选相同 rounds、agent 数、扰动范围和策略噪声设置。保存 manifest 中实际 offsets；当前首个 round 固定无扰动，不能把十个回合都称为独立随机初态。

每个 round 的 agent 共用试验 seed，当前输出的 Wilson 区间只具有描述意义；汇总时保留 round 分组，不把复制体数当作独立 seed 数。后续正式复验再增加独立 round 和训练 seed。当前扰动还会清除刚体速度，仅支持位置扰动鲁棒性的有限结论。

以下属于立即暂停该运行并保留证据的异常：冻结策略审计失败（现有容差最大 `|ratio−1| ≤ 1e-3`）、非有限数、原生包未对齐/步长不符、输入/路线 hash 意外变化、TCP 失败。不得靠放宽 KL 门槛继续。

以下触发诊断和预算复核，不单凭一帧就停训：多个轮次持续接近全动作饱和；新增连续特征某通道裁剪比例持续超过 5%；原生 known 缺失；连续评估只有探索偶然成功而确定性没有进步；G 的更新耗时主导总时间。速度通道短跑已有约 5.15% 裁剪，因此 5% 是检查提示，不是既定失败判据。

若所有候选成功率为零，可按有效进度选后续诊断对象，但必须明确写“尚未出现可重复局部成功”。不能将原地低跌落率、沿线奖励或瞬时 5 米进度视为稳定控制。

## 7. 可执行命令模板

以下在项目根目录 PowerShell 使用；仅展示，本文编写时未执行训练。每个名称对应独立目录，不覆盖已有验收 run。

```powershell
$screenFlags = @{
  B0 = @('--features', 'none', '--policy-backbone', 'transformer', '--no-normalize-action-history')
  B1 = @('--features', 'none', '--policy-backbone', 'transformer', '--normalize-action-history')
  R  = @('--features', 'relations', '--policy-backbone', 'transformer', '--normalize-action-history')
  E  = @('--features', 'relations', 'events', '--policy-backbone', 'transformer', '--normalize-action-history')
  G  = @('--features', 'relations', 'events', '--policy-backbone', 'gru', '--normalize-action-history')
}
$screenArm = 'B0'  # 按阶段选择 B0 / B1 / R / E / G
$screenSeed = 7   # 独立复验改为 17，并从共同源重新初始化
$screenRun = "corridor_screen_20260919_${screenArm}_s${screenSeed}"
$screenArgs = @(
  '--init-from', 'runs/p2_contact/checkpoints/ppo_iter_0335.pt',
  '--reference-trajectory', '--reference-corridor-radius', '1.5',
  '--geometry-profile', 'native-pose-v1', '--no-random-deploy',
  '--recipe', 'ppo-ou', '--seed', "$screenSeed", '--num-agents', '2',
  '--steps-per-rollout', '1200', '--max-iterations', '3',
  '--value-warmup-iters', '2', '--lr', '0.0001', '--ent-coef', '0.001',
  '--event-source', 'native', '--no-event-branch', '--action-segment-steps', '1',
  '--recurrent-seq-len', '16', '--feature-step-seconds', '0.008333333333333333',
  '--run-dir', $screenRun, '--persist-game', '--save-rollout-inputs'
) + $screenFlags[$screenArm]
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo @screenArgs
```

预检通过后，继续到约 48,000 样本（无提前终止时 20 轮）：

```powershell
$screenContinuationSeed = 100000 + $screenSeed
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --resume "runs/$screenRun/checkpoints/ppo_iter_0003.pt" --seed $screenContinuationSeed --max-iterations 20 --run-dir $screenRun --persist-game --save-rollout-inputs
```

入选后从实际端点继续到约 96,000 样本（完整回合时从 iter20 到 iter40）。续训使用 `--resume`，不重传初始化、结构或预热开关。若因落水补了轮次，使用真实 checkpoint 编号，不能照抄 20/40。

```powershell
$screenContinuationSeed = 200000 + $screenSeed
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --resume "runs/$screenRun/checkpoints/ppo_iter_0020.pt" --seed $screenContinuationSeed --max-iterations 40 --run-dir $screenRun --persist-game --save-rollout-inputs
```

当前恢复已验证权重/优化器/迭代，不保证保存和恢复全部 RNG/物理求解器状态。统一采用上述分段和续训 seed 规则并记录实际命令，避免重用同一个随机流；它不是逐位重放承诺。临时故障恢复也要单独记录 seed 和边界。

端点评估示例（48,000 样本筛选；入口自动评估两种模式）：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint "runs/$screenRun/checkpoints/ppo_iter_0020.pt" --output "runs/$screenRun/checkpoints/eval_screen_048k" --steps 1200 --rounds 4 --num-agents 2 --seed 20260919 --jitter 0.01 --goal-progress 5 --dwell 10 --noise-phi 0.85 --native-diagnostics
```

训练前评估将 checkpoint 换成 `checkpoints/rollouts/behavior_0001.pt`，rounds 改为 2，输出目录改为 `eval_initial`。最终保留评估使用真实 96,000 样本端点，rounds=5、seed=20261019、独立输出目录。

主训练与评估串行使用同一个游戏服务；不要并行启动两个入口争用 TCP 或游戏进程。每段调用结束后再启动下一段。保留完整归档前先检查磁盘余量；若需缩减保存频率，统一调整候选的归档策略并留下可重放的预检和评估，不临时删除已有证据。

## 8. 本轮交付和下一轮条件

交付一份结果记录，含每个配置与 seed 的实际训练样本、预热/PPO 更新次数、采集/更新/评估/总耗时、两模式逐回合指标、训练前到端点的变化、异常和缺失项。分别列出 E↔R、R↔B1、B1↔B0、G↔E 的可比数据；没有完整对照就不归因。

值得继续的候选进入 **3 个训练 seed × 每 seed 累计 200,000 物理样本**的对照，并扩大独立评估。该预算另立，不包含在本轮 6 小时内。至少观察各 seed 均有确定性改善；正式局部稳定验收再用更充分的独立试验、1 秒保持条件和失败分析。

后续每次只增加一个主要改动：先对已有直接事件输入测试事件 MLP；随后单独测试 geometry、trends；再比较 H=1、H=2、H=4 及有/无接触中断。动作段会改变 OU 的决策时间尺度，先在固定 PPO 无 OU 下比较段长度，再单独对照 OU，避免把两种改变合并解释。PPO 与 PPO+OU 的收益也在胜出表示上另做配方对照。

若表现持平但 G 达到相同成功水平的总墙钟更低，也可因效率进入下一轮；只报纯推理快不够。只有固定起点局部任务获得可重复成功后，才推进多起点、下一连续段和跌落恢复课程。SIL 与控制内部状态输入仍不纳入本轮。

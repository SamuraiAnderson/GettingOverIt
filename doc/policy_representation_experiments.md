# P1.3 / P2.1–P2.4 表示实验

2026-09-19：按用户要求完成这五项的实现和工程验证。所有候选均可单独启用，原 Transformer、39D 输入和逐步二维动作仍为默认。当前结果证明采集、更新、存档和恢复链路可用；没有完成同预算、多 seed 的学习效果比较。

## 输入及尺度

原始 TCP 状态保持 33D；原有 39D 动力学、局部图和动作历史保留。`--features` 选择追加组，固定顺序为 relations、events、geometry、trends。全部启用时新增 75D，总共 114D。构建入口为 `FeatureBuilder`，在线逐物理步调用和离线序列重建共用实现。

- **P1.3 relations，16D**：cursor−tip 误差、沿杆/垂杆投影、相对速度及其投影、角度误差 sin/cos、由 tip/pole 相对速度得到的角速度、杆长、局部重力方向和有效性标记。位置除以 3.5 米，速度除以 10 米/秒，角速度除以 10 弧度/秒；退化方向为零并置 known=0。
- **事件前置输入，24D**：锤头、身体、锅体各包含 state、enter、exit、距 enter/exit 时间、known、是否见过 enter/exit。时间除以 2 秒并封顶，事件计时跨 Transformer 窗口保持。reset 不生成虚假边缘，未知帧清除事件历史，不跨未知区间推断事件。
- **P2.3 geometry，15D**：锤头原点与角色原点的最近地形轮廓距离、法线、法向/切向速度、known，以及 cursor 和伸展裕量。地形边和包围盒缓存；查询半径默认 3 米。距离是原点到轮廓的近似，负号表示位于某个多边形内，不是角色碰撞体间隙，也不是重叠多边形并集的严格 SDF。裕量使用 cursor 半径 3.5 米、hub→tip 长度尺度 2.955 米；不代表动态可达或抓牢。
- **P2.3 trends，20D**：过去 4/16 个物理步的角色位移、相对 tip 位移、杆转角 sin/cos、速度变化与有效性标记。位移除以 `10 × lag × step_seconds`，保留瞬时输入。历史不足时输出零并置 known=0。

新增连续量裁到 ±4；二值标记为 0/1。`--normalize-action-history` 将历史执行动作除以 `action_scale=100`，不改变传给游戏的动作单位。每轮保存各维 RMS、绝对值 p99、裁剪比例，以及 observation/action embedding RMS。报告键采用 `relations.*`、`events.*`、`geometry.*`、`trends.*`，避免同名 known 标记互相覆盖。

`--event-source native` 默认读取对齐的 `D` v2 数据包中的 `physicsContact/physicsKnown`，使用三个部件的物理接触聚合状态；不使用抓持代理。enter/exit 反映相邻状态边界变化，仍可能漏掉一个物理步内开始又结束的接触。`--event-source geometry` 可切换到已有几何代理，但需要独立验证其误差。原 39D 中的五个几何接触特征仍保留。

本机 `fixedDeltaTime=0.008333334`、`stepFrames=1`，默认 `feature_step_seconds=1/120`。请求原生诊断时会核对该值，不匹配直接报错；`--feature-step-seconds` 用于显式适配不同游戏配置。纯几何模式没有原生包时，应使用已核实的步长。

## 三种结构改动

**P2.1 事件分支。** `--features events` 将事件直接接入观测嵌入：各组线性投影相加，相当于拼接后的分块线性映射。`--event-branch` 改用 24→32 的小 MLP，同时投影到 backbone 和 actor/critic 的共享读出表示；它会自动启用事件特征。两种方式可受控比较，不预设接触后必须反转动作。

**P2.2 CNN+GRU。** `--policy-backbone gru` 使用逐帧 CNN 和单层 GRU，每个 agent 保存独立 hidden，reset 时清零。即使处于动作段中，也消费每个物理帧。训练按连续决策块执行 TBPTT，默认 `--recurrent-seq-len 16` 个决策；从 reset 到块起点的完整前缀用当前权重在 no_grad 下重新计算，块内重新编码并反向传播。没有随机单帧零初始化，也没有复用旧 CNN 编码或旧 hidden。完整前缀重算在长 episode 下可能显著增加更新成本，在线推理加速不能直接外推到训练总耗时。

**P2.4 动作段。** `--action-segment-steps H`，H>1 时 actor 一次采样四个参数，表示两端各一个二维动作，经 `100*tanh` 限幅后，H 个物理步内线性插值。默认在已知接触 enter/exit 后中断，下一步重规划；终止、段长到期和 rollout 结束也关闭当前段。`--no-segment-interrupt-contacts` 可关闭接触中断。段间不强制连续，模型也不必在中断后改变动作。

段参数只记一次 log_prob。实际执行 k 步时，段奖励为 `R = Σ(j=0…k−1) γ^j r_j`，价值差为 `R + γ^k V_next − V`，GAE 延续项为 `(γλ)^k A_next`，终止时截断。γ 和 λ 都按物理步定义；OU 噪声每个决策更新一次，因此 H>1 会改变探索的时间尺度，效果对照必须考虑这一点。每个物理步仍调用一次 TCP STEP，C# `stepFrames` 不变。GRU 中间帧的 CNN/hidden 更新也没有省略。

## 启用、迁移与归档

候选入口为 `main_ppo.py` 和 `local_evaluation.py`。`--init-from` 仅迁移兼容权重，重新创建优化器并从第 1 轮开始；新增特征投影在 warm start 时置零，新增 GRU 使用新参数。二维 actor 迁移到段策略时复制权重到两个端点，四个参数仍分别采样。启用动作历史归一化或 GRU 会改变策略，不能将 warm start 当作无行为变化的恢复。

`--resume` / 评估的 `--checkpoint` 恢复已保存结构与输入语义；显式改动输入组、主干、事件源、动作段或尺度会被拒绝，须改用 `--init-from`。候选当前使用单 actor head，关闭 SIL、停滞加噪和阶段/优势重加权；不支持从旧 BC 直接迁移或使用旧 SIL，因为旧数据缺少对应事件/序列/段参数契约。

以下为一个组合工程 smoke；正式消融应逐项改变开关，固定奖励、预算、seed 和评估集：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --init-from runs/p2_contact/checkpoints/ppo_iter_0335.pt --reference-trajectory --geometry-profile native-pose-v1 --features relations geometry trends --event-branch --policy-backbone gru --normalize-action-history --action-segment-steps 4 --recipe ppo-ou --seed 7 --no-random-deploy --num-agents 2 --steps-per-rollout 96 --max-iterations 2 --value-warmup-iters 1 --run-dir representation_smoke --persist-game --save-rollout-inputs
```

逐步 Transformer 的直接事件输入对照：使用 `--features relations events geometry trends --normalize-action-history`，其余结构开关不传。单项关系特征使用 `--features relations`；单项 GRU 使用 `--policy-backbone gru`。新建实验不要复用上述 smoke 的输出目录。

实验 buffer 为 v3 `physical_sequences`：按 agent 保存物理帧、实际二维动作历史及奖励，并单独保存决策起点、时长、raw 参数、log_prob、value、OU 条件及 GAE。`buffer_size` 为决策数，`rollout_environment_steps` 才是物理样本数。固定评估另存原始状态、原生包、特征尺度和中断原因；正式训练的尺度报告在 `checkpoints/feature_diagnostics/`。原 v2 基线归档仍可重放。

## 已完成的验证与边界

证据汇总见 [verification_summary.json](../runs/p2_representation_acceptance/verification_summary.json)，运行命令见[验收记录](../runs/p2_representation_acceptance/RESULT.md)和[正式入口记录](../runs/p2_representation_training_smoke/RESULT.md)。本机 RTX 4070、conda `getting-over-it-analysis`：

- **35 项 unittest 通过**，覆盖原概率/走廊契约以及单位、缺失与 reset、事件分支梯度、GRU 流式/序列一致性、当前权重前缀重算、段中断、终止、时长 GAE、存档重放和迁移拒绝。
- 两种候选各完成 2 agents × 96 步的确定性、探索评估和真实 PPO 更新。冻结探索概率最大 `|ratio−1|` 分别为 GRU+段 `3.44e−5`、Transformer 直接输入 `2.29e−5`，实际更新参数差 L2 分别为 0.1055、0.1119。正式入口完成 critic 预热、PPO 更新及 optimizer/迭代恢复；第 3 轮全部 57 个决策重放误差 `4.96e−5`，更新前 KL `5.49e−11`，`target_kl=0.03` 未改变。
- 对 776 个含 reset 的实机帧，验证几何数据 hash 后从原始状态和原生接触重建全部 114D 输入，误差为零；段插值动作也逐步一致。checkpoint 差分证实预热仅更新 critic，续训前权重与保存点一致，续训后 CNN、GRU、事件分支和特征投影均有更新。
- GRU+段短跑的 75 个新增通道均无裁剪；Transformer 确定性短跑有 4 个速度/趋势通道触及 ±4，最高比例 5.15%，探索组有 3 个通道为 1.55%。这仍需更长训练的数据分布检查。探索组两种候选的 observation/action embedding RMS 分别约 0.399/0.482 和 0.510/0.559。原生三部件 known 均为 100%；附近表面可能缺失，和接触读取失败是不同含义。早期报告曾发生 `tip_known` 同名覆盖，现已改为分组键，并从完整归档补齐 75 项诊断。
- 同一批归档输入、2 agents × 64 步、32 帧上下文、GPU 预载并同步计时，预热后 3 次推理中位数：Transformer **0.1812 秒**，GRU **0.1030 秒**。对应 CNN 帧编码次数 4096→128；本批纯推理约快 1.76 倍。这个测试排除 TCP、特征构建和训练更新，不是端到端加速或学习收益。带原生诊断的短跑整体仍约 55–59 物理样本/秒。

两种候选的确定性和探索组均为 **0/2 局部成功**。这是旧模型迁移后的短工程验证，不能排序学习能力；身体接触正例、步内短接触、3 seeds × 同预算消融，以及达到相同成功率的总墙钟比较仍未完成。未将组合候选设成默认。

离线复核，不启动游戏：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m unittest src.tests.training.test_ppo_contract src.tests.training.test_reference_corridor src.tests.training.test_policy_experiments -v
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.verify_policy_experiments
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.benchmark_policy_backbones --checkpoint runs/p2_representation_acceptance/checkpoints/gru_segments_v1/behavior.pt --archive runs/p2_representation_acceptance/checkpoints/gru_segments_v1/exploration_000.npz --output runs/p2_representation_acceptance/backbone_benchmark.json
```

特征重建脚本需要原始归档和与 manifest hash 一致的游戏碰撞导出文件；大型权重、NPZ、逐帧 JSON 按仓库惯例保存在忽略的 `checkpoints/` 中。轻量汇总和命令记录可随代码保存。

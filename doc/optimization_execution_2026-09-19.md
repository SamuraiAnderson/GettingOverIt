# 优化优先级执行记录

2026-09-19。对应 [2026-09-16 优先级](optimization_priorities_2026-09-16.md)。本次完成 P0 的工程修复和评估基础设施、P1.4 的独立效率改动，并依据原生观测修正几何标定。学习效果、多 seed 消融和 P2/P3 尚未验收。

## 当前任务和成功判据

根据用户确认，任务已经使用轨迹走廊引导。评估共用 `RolloutWorker.collect_ppo`，分别运行无噪声策略和指定 OU 探索；默认目标是从 reset 基线出发，在同一个连续参考段内获得至少 5 米置信加权有效进度，当前投影位置也前进至少 5 米，并在走廊内连续保持 10 个环境步。`--goal-progress` 与 `--dwell` 可配置。目标跨参考缺口直接报错，不把投影跳变或绕路补分算成功。

记录落水或峰值回落至少 3 米的跌落、离开走廊、到达步数、原始轨迹和动作/接触 enter/exit。每组初态扰动相同，默认 ±0.01 米且第一轮不扰动；传送会清零刚体速度，manifest 显式记录此初始化语义。Unity 物理解算与复位不保证逐位确定。并行 agent 共用 trial seed，小样本 Wilson 区间仅为描述性输出，不能当作独立多 seed 的统计证据。

## 已落地的修改

- **P0.1：概率契约。** 保留采样的 tanh 前 raw action，用稳定 Jacobian 计算密度；采样和训练共用 OU 条件、探索 boost、动作缩放及 log-std 边界。训练前冻结审计，非有限 transition 中止采集，避免跨丢失步骤拼接 GAE。无噪声评估 buffer 不允许进入 PPO 更新。
- **P0.2：时序与重放。** 接触派生状态每个环境步只更新一次，bootstrap 复用末帧；reset 清空历史，终止 agent 不继续计入轨迹。NPZ v2 保存 patch 帧与窗口索引、动力学、mask、动作历史、raw action 和策略上下文，配套采样前权重可离线重放，不依赖后来变化的地图。
- **P1.4：减少重复计算。** 同步批推理活跃 agent，patch 按帧共享，minibatch 再组装窗口；静态地形缓存包围盒。CNN 在 PPO 更新时正常重算并反向传播。分开记录接触、patch、推理、环境、bootstrap、诊断、总采集和更新耗时。
- **原生诊断与标定。** 插件新增可选 `D` 数据包，主线程在状态采样边界同时缓存内部控制字段、引擎接触与部件姿态，带 known mask 和序列号。原 33D 状态包保持原样。Python 在 reset 前启用诊断并校验对齐，正常训练不请求它。
- **可控运行。** `--recipe ppo/ppo-ou/ppo-ou-sil`、`--seed`、`--save-rollout-inputs`、`--geometry-profile` 已接入。走廊模式拒绝按高度筛选的旧 SIL；恢复 checkpoint 的配置后再应用显式覆盖，奖励或几何语义变化时重新预热 critic。

## 实测证据和限制

环境为本机 RTX 4070、conda `getting-over-it-analysis`。主要工程验证从旧单头策略 `runs/p2_contact/checkpoints/ppo_iter_0335.pt` 开始；它未针对新走廊或新几何标定训练。详细数值见 [verification_summary.json](../runs/p0_corridor_acceptance/verification_summary.json)，原始 NPZ、权重和逐帧 JSON 保留在各实验的 `checkpoints/`，按仓库惯例不入库。

**冻结概率与真实更新。** 合成 ±20 raw action（执行值达到 ±100）通过稳定 Jacobian 和概率重算。实机新标定评估共 2048 transitions：最大 `|ratio-1|=0.000689`，更新前平均 KL 最大约 `7.97e-10`；确定性组 ratio 误差为零。新标定下短 PPO 更新执行了 3 个 minibatch，参数差 L2 为 0.1531。旧几何的正式入口两轮执行 3/5 个 minibatch，全部 256 个归档样本重放也通过。

冻结审计的 ratio 容差为 `1e-3`，用于 float32 CUDA 的不同 batch kernel 舍入：早期串行归档按 batch=1 重放精确为零，batch=16/64 约 `1.47e-4`；原 `2e-4` 门槛曾误拦截 `2.13e-4`。这是数值容差，PPO `target_kl=0.03` 与 1.5 倍早停保持原值，实际更新仍能触发早停。常规训练抽查 128 样本，验收/离线重放检查整个 buffer；未证明任意硬件和任意长训练永不触发审计。

**吞吐与内存。** 2 agents × 128 步、32 帧上下文，patch 实际帧存储 4 MiB，原重复窗口为 128 MiB，即该部分减少 32 倍；这不是整个进程内存下降 32 倍。正式入口热轮采集 118.6 samples/s，其他无诊断请求短轮约 119 samples/s。冷启动轮为 55–95 samples/s；首次 CUDA 初始化、物理状态和诊断采样都会影响速度，不据这些短轮宣称严格的端到端加速倍数。开启逐步诊断请求的新标定评估约 47–60 samples/s。

**姿态标定。** 旧标定 2056 个对齐帧显示：锤头 Transform 相对杆向量约为 ±180°，身体相对 player→hub 约为 −84.498°；旧配置两者均为 −90°。同一批状态仅修正角度后，锤头接触不一致从 137/2056（6.66%）降到 52/2056（2.53%），锅体从 165/2056（8.03%）降到 87/2056（4.23%）。这验证的是几何代理，不是学习收益。

独立新标定实机采样的 2056 帧中，锤头接触不一致为 71/2056（3.45%），锅体为 87/2056（4.23%）。抓持代理仍漏掉 190/310 个原生正例，不能当作可信抓牢事件；身体接触原生正例为零，不能据零误差宣布身体接触已验证。六个内部控制字段的 known 比例均为 100%，其中 mouseSnap、inputsToSkip、pauseInputTimer 在本批数据始终为零，尚未覆盖这些控制分支。详细计数见 [原姿态诊断](../runs/p0_corridor_acceptance/native_contact_audit.json)和[新标定实机诊断](../runs/p0_corridor_acceptance/calibrated_native_contact_audit.json)。

已生成并检查[接触与动作时序图](../runs/p0_corridor_acceptance/logs/contact_action_alignment.png)。横轴动作 t 对应执行前状态 t 的原生/代理接触，竖线为原生锤头 enter/exit；reset 不虚构边缘。图保存在忽略的 logs 目录，可用下面命令从归档重建。

`native-pose-v1` 是姿态代理的版本名，不表示接触等于原生物理真值。新模型默认采用它；保存过配置的旧模型恢复原值，缺少配置的旧模型按 legacy。显式 profile 才改变旧策略的输入。BC 历史归档没有几何元数据，BC 入口继续保持 legacy；PPO 从 BC 迁移也遵循这一兼容策略。

**局部成功。** 新标定短评估为 2 rounds × 2 agents × 256 步，两种策略各 0/4 成功、0/4 跌落、4/4 曾离开走廊。有效进度均值分别约 0.480 米和 0.204 米。旧模型加新输入的工程 smoke 不能用来评价走廊学习上限，也不足以宣称 OU 优劣。

## 复核命令

项目根目录 PowerShell，Python 统一通过上述 conda 环境：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m unittest src.tests.training.test_ppo_contract src.tests.training.test_reference_corridor -v
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.training.test_patch_tip_coverage
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.training.test_raster_area_coverage
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint runs/p2_contact/checkpoints/ppo_iter_0335.pt --geometry-profile native-pose-v1 --output runs/p0_corridor_acceptance/checkpoints/calibrated_native --steps 256 --rounds 2 --num-agents 2 --native-diagnostics --train-smoke 1
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.diagnose_native_contacts runs/p0_corridor_acceptance/checkpoints/calibrated_native --output runs/p0_corridor_acceptance/calibrated_native_contact_audit.json
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.plot_contact_alignment runs/p0_corridor_acceptance/checkpoints/calibrated_native --output runs/p0_corridor_acceptance/logs/contact_action_alignment.png
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint runs/p0_corridor_calibrated_smoke/checkpoints/rollouts/behavior_0336.pt --replay runs/p0_corridor_calibrated_smoke/checkpoints/rollouts/iter_0336.npz --output runs/p0_corridor_calibrated_smoke/checkpoints/replay
```

本次 23 项 unittest、锤头窗口与栅格覆盖均通过；OU/MAP-Elites、接触几何、动力学维度、34D checkpoint 扩展和 SIL smoke 也已通过。C# 插件使用本机 VS2019 BuildTools 编译、部署，并完成上述真实 TCP 交互验证。正式入口切换标定时重启 critic 预热的结果见 [calibrated smoke](../runs/p0_corridor_calibrated_smoke/RESULT.md)。

## 后续状态更新

本文保留第一阶段 P0/P1.4 的测量记录。随后按用户要求完成 P1.3 和 P2.1–P2.4 的可选实现与工程验证，详见[表示实验](policy_representation_experiments.md)。原生事件模式现在会在训练时请求诊断包；本文“正常训练不请求”指原基线路径。最新状态如下。

## 仍待完成的阶段

1. P0.2 补充身体触地正例、短暂接触和抓持分支的原生对照；短 smoke 不足以关闭这些验证项。
2. 在固定几何和同一走廊目标下训练 PPO / PPO+OU，按相同样本预算和独立 seeds 评估；原建议的每组每 seed 20 万样本、3 seeds 尚未执行。不能从工程吞吐或旧策略评估推导学习收益。
3. P1.1 的事件/时间/known 和 P1.3 控制关系特征已接入并支持单独启用，效果消融待做。P1.2 的六个字段目前仅诊断读取，进入模型还需输入/归一化/权重迁移。
4. SIL 要先改成与走廊目标一致的筛选与保存契约，才能参与同预算对照；现阶段保留显式拒绝，避免把高度 SIL 混入路线实验。
5. P2.1–P2.4 候选已完成实现、短 PPO 更新、重放和恢复续训；还需逐项比较成功率及达到相同成功率的墙钟时间。局部成功稳定后再推进 P3。

# 战役编排

## 树下攀爬诊断与单变量配置

`prepare_tree_climb.py` 离线读取实际 GRU checkpoint、完整精英池和冻结评估，
重算奖励/确认进度，审计存档行为概率，检查观测重建和隐藏态前缀，追踪历史
SIL 片段到当轮精英池，并生成 `REPORT.md`、JSON 证据及独立实验配置。
不启动游戏、不执行 optimizer 更新。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.campaign.prepare_tree_climb --output runs/tree_climb_preflight_v2
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --experiment-config runs/tree_climb_preflight_v2/recipes/precheck_control_s7.json --dry-run
```

配置完整继承源 checkpoint，而非关闭 SIL 的评估配置。四组分别为原行为、
独立引导扩散宽度、轨迹价值与质心动力学采样的 GRU SIL（sil_mixed）、较长奖励折扣；同一组种子内仅
改变一个因素。共用原 actor/GRU/归一化/std，重新初始化 critic，使用新优化器，
两轮仅预热 critic。预检三轮和筛选十轮是各自从源模型开始的实验，不能重复计数。
共同设置 `sil_start_iter=2`，第三轮首次 PPO 更新后即检查 GRU SIL（原始起始轮次 5
不适合三轮工程预检）；若 PPO 实际更新预算不足以执行 SIL，在线验收必须报未覆盖。
`--dry-run` 是纯检查；去掉它会训练，准备脚本不会自动这样做。

阶段标签使用动作前状态和几何接触代理。高处锤头接触只是树枝接触候选，未标定
为游戏物理真值。历史日志没有逐帧优势，报告明确留空；最终模型复算的优势不是
当时梯度。所有新配置仍需游戏内采样/更新/恢复预检，前缀回放后的策略接管入口
尚未实现。旧 checkpoint 缺少采样字段时继续采用 `trajectory_uniform`；
新建 PPO 的 GRU SIL 使用 `quality_dynamics_v3`，确认进度排名决定轨迹份额，
轨迹内按非重力接触加速/转向及前后短上下文分配。两个层次分别保留 10% 均匀机会。
恢复旧模型时通过 `--gru-sil-sampling-mode quality_dynamics_v3` 显式切换；未指定就保留原行为。
对照组固定为原采样，`reference_guidance_sigma=None` 保持旧观测行为。

源码改变后旧配置的哈希检查会拒绝执行。可用 `--configs-only --output <新目录>`
核对模型/精英池/路线/插件身份并重新生成 24 份配置。这只通过配置预检，不能代替
全量离线审计或游戏内预检；旧报告保留为当时的证据。

## 绘制精英池采样情况

复用现有包络绘图入口，保留历史规则并比较 v2 软阶段事件与 v3 轨迹价值/动力学：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.present.plot_sil_td_envelope --pool runs/gru_sil_8h_20261003/elites/full_episodes/elites_iter_0184.npz --sampling-checkpoint runs/gru_sil_8h_20261003/checkpoints/ppo_iter_0184.pt --output runs/gru_sil_dynamics_sampling_20261004
```

热图使用训练采样器的精确片段概率，阶段图再对照 20,000 次离线抽样频率；
`sampling.json` 保存每个片段的概率分量、事件重叠范围、抽样次数、阶段覆盖及输入身份。按实际片段长度
统计帧占比，不把抽样次数当成正优势帧数或接受更新量。此入口不创建模型、不启动游戏。
`sampling_trajectory_mass.png` 分开显示整条轨迹份额；`sampling_dynamics.png` 显示
前两条轨迹的质心速度变化、核心与上下文重要性、实际片段概率，方便检查价值判断。

要把同一组概率标在锅心的空间轨迹上，使用已有运动轨迹入口的
`--elite-pool <pool.npz> --manifest <manifest.json> --sampling-report <sampling.json>`。
它核对池文件 SHA-256，生成完整池总览和每条精英的 v2 / v3 规则并排图；
也支持历史报告的原规则 / 旧混合规则图。
动作 t 的概率对应状态 t→t+1 线段，封顶后尾段用灰虚线表示。
可加 `--sampling-focus 12 192`，在池索引 12 的并排图上标明第 192 步所在片段。

历史对照的一次性控制器已删除，结论留在对应的 `runs/` 报告里。

- `prepare_v2_formal.py` — 检查稀疏几何权重和 v2 精英池，写出当时那次迁移的第一段参数。不启动游戏。
- `compare_backbones.py` — 当前隔离物理环境的 Transformer / GRU 从零比较。
  使用 `getting-over-it-analysis` 环境，以模块
  `src.tests.analysis.campaign.compare_backbones` 执行。依次运行
  `--stage precheck`、`--stage screen`、`--stage evaluate`；
  `--stage report` 可随时汇总已有结果，不启动游戏。

比较协议保存在 `runs/backbone_compare_20261002/protocol.json`：三个配对种子
7/17/47，共享前端与归一化，PPO+OU，8 agents × 1200 步，32 帧 Transformer
与单层 GRU128（TBPTT 32）。前两轮只预热 critic，第三轮执行 PPO 并做完整
行为概率回放审计。每种模型每个种子的筛选预算为 96,000 个物理世界步；
预检计入预算。部分 agent 提前终止后接收零动作，仍随其余世界推进；这些步不进入
训练 buffer。实际训练样本数另列，生产日志的 `rollout_environment_steps` 恰是
活动 agent 的转移计数，不能冒充全部世界推进量。执行器从动作段记录核对每轮
是否跑满 1,200 个向量步；全体提前终止导致预算不足时明确报错。
两种架构参数量不同，结果只代表当前配置。

根据用户后续明确的任务范围，每个模型的 0/10 轮 checkpoint 只在精确出生状态
评估：2 轮 × 2 个隔离世界，位置扰动为零。四次重复用于检查跨世界、跨 reset 的
复现性，每个策略只计一个确定性结果，训练种子才是独立复验单位。
结果写入 `eval_exact/`。此前已完成或中断的扰动评估保留在 `eval/`，不纳入此次
选型；原协议保留在 `protocol_perturbation_archived.json`。5m 到达及保持判据
沿用现有评估器，其中保持容差尚未独立标定；参考路径除出生点外也不是逐点实机
验证过的路线。不要将短预算的零成功率解读为某种架构不能学习。

训练、完整行为 checkpoint、采样输入、命令及耗时均归档；不加载旧精英池。
恢复使用生产入口已有的 optimizer 恢复与 seed 重置行为，两种架构保持相同的
3+7 轮进程边界。脚本不自动启动 192k 扩展，也不自动选出赢家。

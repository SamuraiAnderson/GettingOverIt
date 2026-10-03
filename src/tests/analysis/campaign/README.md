# 战役编排

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

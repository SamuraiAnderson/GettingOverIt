# 表示候选工程验收（2026-09-19）

完整说明见 [P1.3/P2 表示实验](../../doc/policy_representation_experiments.md)，轻量证据见 [verification_summary.json](verification_summary.json)。权重、逐帧状态、原生包、v3 完整输入和尺度报告在本目录的 `checkpoints/`，按仓库惯例忽略。

## 实际执行

项目根目录 PowerShell：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --init-from runs/p2_contact/checkpoints/ppo_iter_0335.pt --geometry-profile native-pose-v1 --features relations geometry trends --event-branch --policy-backbone gru --normalize-action-history --action-segment-steps 4 --output runs/p2_representation_acceptance/checkpoints/gru_segments_v1 --steps 96 --rounds 1 --num-agents 2 --native-diagnostics --train-smoke 1
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --init-from runs/p2_contact/checkpoints/ppo_iter_0335.pt --geometry-profile native-pose-v1 --features relations events geometry trends --normalize-action-history --output runs/p2_representation_acceptance/checkpoints/transformer_concat --steps 96 --rounds 1 --num-agents 2 --native-diagnostics --train-smoke 1
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint runs/p2_representation_acceptance/checkpoints/gru_segments_v1/behavior.pt --replay runs/p2_representation_acceptance/checkpoints/gru_segments_v1/exploration_000.npz --output runs/p2_representation_acceptance/checkpoints/gru_segments_v1/replay
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.benchmark_policy_backbones --checkpoint runs/p2_representation_acceptance/checkpoints/gru_segments_v1/behavior.pt --archive runs/p2_representation_acceptance/checkpoints/gru_segments_v1/exploration_000.npz --output runs/p2_representation_acceptance/backbone_benchmark.json
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.verify_policy_experiments
```

首次 `gru_segments` 用旧默认步长 0.02 秒，和本机 1/120 秒不符，被对齐检查正确拒绝。失败记录保留；修正默认步长后使用新目录 `gru_segments_v1`。

## 结果

- GRU+段：每模式 192 物理步，确定性 56 决策/18 次接触中断，探索 63 决策/32 次中断；探索冻结 ratio 最大误差 3.44e−5。独立训练 smoke 执行 3 次 minibatch 更新，参数差 L2=0.1055。
- Transformer 直接输入：每模式 192 步/192 决策，探索 ratio 最大误差 2.29e−5；训练 smoke 执行 2 次更新，参数差 L2=0.1119。
- 从真实原始状态和对齐原生接触重建 776 帧全部 114D 输入，与存档逐值相同；实际动作与对应段参数插值一致。所有 75D 的完整尺度报告已补齐，原始早期报告的未分组 `tip_known` 覆盖问题保留在历史文件中，复核脚本会显式识别并从完整输入重建。
- 相同输入的纯 GPU 推理：预热后 3 次中位数 Transformer 0.1812 秒、GRU 0.1030 秒，CNN 编码帧数 4096 与 128。排除 TCP/特征构建/训练，不能当作整体训练加速比。
- 两种候选每个模式均 0/2 成功；样本太短，且结构、动作和探索时间尺度有多项变化，不构成学习效果排名。

单元测试最终为 35 项通过。正式训练入口与恢复的证据另见 [training smoke](../p2_representation_training_smoke/RESULT.md)。

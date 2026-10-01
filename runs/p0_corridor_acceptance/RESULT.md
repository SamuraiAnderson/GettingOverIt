# P0 / P1.4 工程验收

2026-09-19。本目录验证冻结概率、真实更新、同管线局部评估、接触原生对照及采集开销。旧策略为 `runs/p2_contact/checkpoints/ppo_iter_0335.pt`，并非走廊训练后的策略。

轻量结果见 `verification_summary.json`、`native_contact_audit.json`、`calibrated_native_contact_audit.json`；详细命令和解释见 [执行记录](../../doc/optimization_execution_2026-09-19.md)。`checkpoints/` 保留可重放 NPZ、behavior.pt、manifest、逐帧轨迹/事件/原生包和 results.json，不纳入 git。

- `verification`：最初串行概率对照；小于 float32 batch 差异的旧容差导致短更新停止，未保存最终 results。保留输入；不把失败试验包装为成功。
- `optimized`：批推理/共享 patch 的 128 步 × 2 agents × 2 rounds；冻结审计和两轮实际更新通过。此阶段旧版原生包未在物理采样边界对齐，接触对照已被后两组替代，不再用作原生接触结论。
- `aligned_native`：旧几何，v2 对齐诊断，256 步 × 2 agents × 2 rounds，共 2056 个状态帧。用于姿态标定，并在同一批状态上重算新角度。
- `calibrated_native`：相同规模，新几何的独立实机验证。2048 transitions 冻结概率最大 ratio 误差 0.000689；一次 PPO smoke 执行 3 个 minibatch，参数差 L2=0.1531。锤头接触误差 3.45%、锅体 4.23%；抓持漏判和身体正例覆盖尚未解决。

## 结论

raw action 概率契约与按帧存储经过实机和重放验证；32 帧窗口的 patch 存储减少至原来的 1/32。当前短评估的无噪声和探索组均为 0/4 成功，不能得出学习收益或 OU 优劣。三组轻量摘要的字段随诊断版本变化，旧版未对齐接触值不宜横向混合。

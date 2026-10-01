# A：原采样 + 原 D 门控

复现冻结起点上的旧 SIL 路径。`sil_td_sampling_mode=legacy_min_length`，不改 `sil_td_windows_per_traj=0`，单一 D 门控仍按行为水平生效。

其余训练配置沿用 checkpoint：`td_pareto`、v2 进度、batch 64、SIL epoch 1、`sil_max_batches=40`、`sil_budget_ratio=1.0`。新增 192000 个物理样本，种子 47。

```powershell
.\launch.ps1          # 只解析配置
.\launch.ps1 -Train   # 才会启动训练
```

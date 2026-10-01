# 探索 std 上限从 1.0 压到 0.6

在 B 组配置上加 `--log-std-max -0.5`，其他不变。B 组第 143 轮 `actor_log_std ≈ [-0.03, -0.06]`（std≈0.94–0.97），已经贴到默认 clamp 上限 0.0。压到 −0.5 让 std ≤ 0.6，在线动作幅度中位从约 70–80 降到约 55。

- 起点：`runs/confirmed_endpoint_v2_formal_20260928_S_s47/checkpoints/ppo_iter_0123.pt`
- 精英池：同目录 `elites/elites_iter_0123.npz`
- 采样：`balanced_stratified`（复用 B 的模式），H=1200，K=32
- 训练种子：47，新增 192000 个物理样本

判据（对比 B 组的 5 次独立评估）：
- 确认进度均值提升 ≥ 1 米，且保持成功次数不下降：认为压 std 有效，保留新的上限。
- 否则恢复 `ppo_log_std_max=0.0`，问题不在探索强度。

```powershell
.\launch.ps1
.\launch.ps1 -Train
```

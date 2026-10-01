# SIL 采样对照（不自动开训）

三组实验从同一份冻结起点继续，只改变 SIL 采样。`launch.ps1` 默认是 `--dry-run`，真正训练必须显式加 `-Train`。

- 起点：`runs/confirmed_endpoint_v2_formal_20260928_S_s47/checkpoints/ppo_iter_0123.pt`
- 精英池：同目录 `elites/elites_iter_0123.npz`
- 新增物理样本：192000（8 agent × 1200 步 × 20 轮）
- 种子：47。优化器随 `--resume` 恢复。

| 组 | 采样 | 单一 D 门控 |
| --- | --- | --- |
| A | `legacy_min_length` | 保持原门控 |
| B | `balanced_stratified`，K=32 | 新模式关闭 |
| C | `coverage_share`，H=1200，K=32 | 新模式关闭 |

A/B 同时改变了采样覆盖、门控和实际更新量。只有 C 相对 B 的闭环差异能用来判断面积分摊是否有额外价值。

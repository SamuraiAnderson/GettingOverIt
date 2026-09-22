# SIL 优势加权 smoke（走廊模式，--corridor-sil）

验证 P3.2 优势加权 SIL 在真实游戏回路中端到端工作。7 轮、2 agents、600 步/轮，
从 `p2_contact/ppo_iter_0335.pt` 迁移，种子 `runs/best_motion_20260923/sil_seed_590.npz`
（credited 8.099m，入池 score 8.785）。

## 结果

| iter | sil_loss | sil_trajs | sil_windows | sil_updates | sil_adv_pos_frac | sil_weight_mean |
|------|----------|-----------|-------------|-------------|------------------|-----------------|
| 6 | 0.380 | 5 | 1893 | 30 | 0.056 | 0.468 |
| 7 | 0.289 | 5 | 1893 | 30 | 0.119 | 0.469 |

- 精英池 5/5 轨迹携带 corridor_progress（experimental_rollout 修复生效）；
- MAP-Elites 正常竞争（cells 2→6，种子被更高分采集轨迹替换——设计内行为）；
- sil_loss 为加权高斯 NLL 口径（raw 空间，与旧 MSE 的动作单位口径不可比）；
- `sil_adv_pos_frac` 0.056→0.119：不再钉 0，且随 critic 预热上升——健康方向。
  该值是「精英窗口回报超过当前 critic 预期」的比例，critic 充分训练后应稳定在中等区间；
  长期钉 0（价值高估/视界失配）或钉 1（价值低估）才异常。本 smoke 的 critic 只预热 1 轮，
  低 pos_frac 属预期。
- `sil_weight_mean` ≈0.47 < 1：归一化优势右偏分布下，多数窗口权重 <1、少数高优窗口
  吃到至多 5×——加权机制按设计工作。

## 本 smoke 牵出的三个真实缺陷（均已修复）

1. **走廊模式强制停用 SIL**（`configure_reference`，走廊 SIL 机械就绪前的遗留）——
   新增 `--corridor-sil` 显式启用（覆盖 configure_reference 与 experimental 路径的
   两处强制停用；explore_boost 保持停用，其停滞判据是高度口径）。
2. **`experimental_rollout.collect_decisions` 不挂 corridor 字段**——experimental 特性
   开启时（正式训练的 `--normalize-action-history` 等即触发）精英池完全拿不到
   corridor_progress，走廊评分/剪枝/优势加权全部静默失效。已补齐（逐帧 credit/guidance，
   与 collect_ppo 同口径）。
3. **R/V 视界失配**：R（回合内 capped）比 V（无限时域自举口径）短一个视界，A 被系统性
   压负（首跑 pos_frac=0.000）。已在轨迹末窗加 V 自举修正
   （R_full(τ) = R(τ) + γ^Δ·(V(τ_last) − R(τ_last))）。

历史调试运行：b=SIL 未启用，c=回退 MSE（缺陷 2 现场），d=加权无自举（缺陷 3 现场）。

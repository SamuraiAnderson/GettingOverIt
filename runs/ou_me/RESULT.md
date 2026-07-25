# ou_me

**意图**：P1.1 OU 连贯噪声 + P1.2 MAP-Elites 精英池
**状态**：已完成
**时间**：2026-07-23T18:31:52.678Z

首个 sprint 的验证跑；结论是 4.28m 天花板，触发转向 P2.1 接触信号。

## 命令

```
conda run -n getting-over-it-analysis python -m src.training.main_ppo --resume checkpoints_ou_me/ppo_iter_0040.pt --num-agents 16 --steps-per-rollout 400 --max-iterations 290 --no-random-deploy --checkpoint-dir checkpoints_ou_me --log-dir logs_ou_me --log-std-max 0.5 --persist-game
```

## 客观结果

- 恢复到 250 轮指标（iter 41–290）
- mean_reward：末轮 0.1403，峰值 0.8574
- 精英池最佳 secured_dy：末轮 4.28，峰值 4.28
- MAP-Elites 占据 cell 数：末轮 11，峰值 11

## 结论

P1.1（OU 连贯噪声）+ P1.2（MAP-Elites 精英池）**未能突破第一段攀爬**。跑满 290 轮后
精英池最佳 secured_dy 停在 4.28 m 不再增长，MAP-Elites 占据 cell 数停在 11 —— 行为
多样性本身已经饱和，不是"探索还没铺开"，而是铺开了也撞不到有效的攀爬动作。

这一结果排除了"连贯探索不足"这个假设，把瓶颈推向观测侧：策略看不见自己有没有接触
地形，也就无从学会"勾住再拉"。据此转向 P2.1 接触信号（DYNAMICS_DIM 34→39）。

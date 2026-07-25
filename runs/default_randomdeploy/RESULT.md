# default_randomdeploy

**意图**：跑在默认输出目录的随机投放 PPO（16 agents，y-max-cutoff 125）
**状态**：已完成
**时间**：2026-07-19T11:35:19.470Z

早于 --checkpoint-dir 约定，产物混在 checkpoints/ 与 logs/ 里。只摘走 ppo_iter_*.pt 与 ppo_effmap_*.png；BC 权重、warmup_state 与环境资产（environment.json 等）留在原地，它们是共享输入而非本次产物。

## 命令

```
C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --num-agents 16 --random-deploy --y-max-cutoff 125 --steps-per-rollout 500
```

## 客观结果

- 恢复到 100 轮指标（iter 1–100）
- mean_reward：末轮 -0.1673，峰值 -0.1612

## 结论

random-deploy 路线在默认输出目录下的长跑，100 轮 mean_reward 始终为负。归档时保留的
效率图有 34 张（iter 5–170），比 checkpoint 覆盖的轮次更长，说明这个目录被多次运行
覆写过——**这正是后来引入 `--checkpoint-dir` 分目录的直接原因**，也是本次归档
把产物收敛到 `runs/<tag>/` 的动机。

因此这条记录的指标只能对应到命令栏里那一次运行，早于它的若干次运行已无法分离。

# bcseed

**意图**：BC 种子 + 随机投放的首轮 PPO 试跑
**状态**：已完成
**时间**：2026-07-20T09:10:13.000Z

最早一批 random-deploy 实验，后被固定起点最小任务（P0.1）取代。

## 命令

```
& C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --random-deploy --num-agents 10 --checkpoint-dir checkpoints_bcseed --log-dir logs_bcseed
```

## 客观结果

- 恢复到 17 轮指标（iter 1–17）
- mean_reward：末轮 0.1829，峰值 0.3264

## 结论

random-deploy（随机投放到候选点）路线的首轮试跑。投放姿态不稳定与基座控制未通两个
未解问题叠加，难以归因，17 轮后放弃该路线，转向 P0.1 固定起点最小任务。

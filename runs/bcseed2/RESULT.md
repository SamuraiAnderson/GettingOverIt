# bcseed2

**意图**：bcseed 重跑，加 --y-max-cutoff 125 限定低处投放区
**状态**：中止
**时间**：2026-07-20T12:36:17.089Z

跑到 iter5 即中止，被 bcseed3 以同配方接手。

## 命令

```
& C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --random-deploy --num-agents 10 --checkpoint-dir checkpoints_bcseed2 --log-dir logs_bcseed2
```

## 客观结果

- 恢复到 7 轮指标（iter 1–7）
- mean_reward：末轮 0.2851，峰值 0.2884

## 结论

_未记录（实验中止，无有效结论）。_

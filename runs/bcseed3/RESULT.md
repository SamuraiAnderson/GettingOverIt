# bcseed3

**意图**：bcseed2 配方重跑
**状态**：中止
**时间**：2026-07-20T13:28:54.449Z

跑到 iter5 即中止，被 bcseed4 接手。

## 命令

```
Get-Process python,GettingOverIt -ErrorAction SilentlyContinue | Select-Object Id; echo "---launch---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --random-deploy --num-agents 10 --y-max-cutoff 125 --steps-per-rollout 500 --checkpoint-dir checkpoints_bcseed3 --log-dir logs_bcseed3
```

## 客观结果

- 恢复到 5 轮指标（iter 1–5）
- mean_reward：末轮 0.1781，峰值 0.2551

## 结论

_未记录（实验中止，无有效结论）。_

# bcseed4

**意图**：bcseed3 配方 + 显式 --max-iterations 10
**状态**：中止
**时间**：2026-07-20T13:49:20.149Z

跑到 iter5 即中止。此后放弃 random-deploy 路线，转向固定起点最小任务。

## 命令

```
Get-Process python,GettingOverIt -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 2; echo "---launch bcseed4---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --random-deploy --num-agents 10 --y-max-cutoff 125 --steps-per-rollout 500 --max-iterations 10 --checkpoint-dir checkpoints_bcseed4 --log-dir logs_bcseed4
```

## 客观结果

- 恢复到 5 轮指标（iter 1–5）
- mean_reward：末轮 0.2347，峰值 0.2347

## 结论

_未记录（实验中止，无有效结论）。_

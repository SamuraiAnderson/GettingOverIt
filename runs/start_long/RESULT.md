# start_long

**意图**：固定起点长跑 baseline（60 轮）
**状态**：已完成
**时间**：2026-07-20T14:57:28.531Z

SIL 对照组的 baseline，eval 记录见 runs/_eval/。

## 命令

```
Get-Process python,GettingOverIt -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 2; echo "---launch long fixed-start PPO (60 iters)---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --no-random-deploy --num-agents 10 --steps-per-rollout 500 --max-iterations 60 --checkpoint-dir checkpoints_start_long --log-dir logs_start_long
```

## 客观结果

- 恢复到 60 轮指标（iter 1–60）
- mean_reward：末轮 -0.5274，峰值 1.039

## 结论

固定起点最小任务（P0.1）的 baseline，`start_sil` 的对照组。60 轮内 mean_reward 从峰值
1.04 回落到 -0.53，未出现稳定攀爬。价值主要用于给后续实验提供同配方的比较基准。

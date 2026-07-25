# start1

**意图**：P0.1 固定起点最小任务首试
**状态**：中止
**时间**：2026-07-20T14:01:01.522Z

跑到 iter5 即中止；发现 dropout 未关，由 start2 修复后重跑。

## 命令

```
echo "---launch start1 (fixed start)---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --num-agents 10 --steps-per-rollout 500 --max-iterations 10 --checkpoint-dir checkpoints_start1 --log-dir logs_start1
```

## 客观结果

- 恢复到 9 轮指标（iter 1–9）
- mean_reward：末轮 -0.1756，峰值 1.092

## 结论

_未记录（实验中止，无有效结论）。_

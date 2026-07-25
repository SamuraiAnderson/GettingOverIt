# start2

**意图**：start1 + dropout-off 修复
**状态**：中止
**时间**：2026-07-20T14:20:50.353Z

跑到 iter10 即中止，配方由 start_long 接手做长跑。

## 命令

```
echo "---launch start2 (dropout-off fix)---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --no-random-deploy --num-agents 10 --steps-per-rollout 500 --max-iterations 10 --checkpoint-dir checkpoints_start2 --log-dir logs_start2
```

## 客观结果

- 恢复到 10 轮指标（iter 1–10）
- mean_reward：末轮 -0.1948，峰值 1.115

## 结论

_未记录（实验中止，无有效结论）。_

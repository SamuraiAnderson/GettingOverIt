# treefix

**意图**：地形 mask 3x3 子采样修复后重跑（110 轮）
**状态**：已完成
**时间**：2026-07-21T02:10:03.095Z

验证 mask 混叠修复；由 treefix2 叠加 PBRS gamma=1 继续。

## 命令

```
Stop-Process -Name "GettingOverIt" -ErrorAction SilentlyContinue; Start-Sleep -Seconds 3; chcp 65001 > $null; $env:PYTHONIOENCODING="utf-8"; echo "---launch tree-fix run: mask 3x3 subsample, persist-game, 250 iters, target done ~17:10 CST---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --no-random-deploy --persist-game --num-agents 10 --steps-per-rollout 500 --max-iterations 250 --checkpoint-dir checkpoints_treefix --log-dir logs_treefix 2>&1
```

## 客观结果

- 恢复到 112 轮指标（iter 1–112）
- mean_reward：末轮 -0.1747，峰值 1.071

## 结论

验证地形 mask 3x3 子采样修复（消除树干附近的栅格混叠）。112 轮内 mask 修复本身没有
改变训练形态，后续由 treefix2 在此基础上叠加 PBRS gamma=1 继续。

# start_sil

**意图**：固定起点 + SIL 精英池（190 轮）
**状态**：已完成
**时间**：2026-07-20T16:25:03.383Z

路线图瓶颈判断的证据之一：iter190 效率图价值仍只在起点盆地。

## 命令

```
Get-Process python,GettingOverIt -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 3; echo "---launch SIL control run (60 iters, same BC seed)---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --no-random-deploy --num-agents 10 --steps-per-rollout 500 --max-iterations 60 --checkpoint-dir checkpoints_start_sil --log-dir logs_start_sil
```

## 客观结果

- 恢复到 190 轮指标（iter 1–190）
- mean_reward：末轮 -0.5411，峰值 1.051

## 结论

跑满 190 轮，`logs/ppo_effmap_iter_0190.png` 的价值分布仍只覆盖起点盆地，与
`start_long`（同配方无 SIL，60 轮）相比没有质变。SIL 精英池在上游探索从未产出正样本
的前提下，只能反复强化"起点乱晃"——这是把 SIL 归入"探索的下游机制"的直接证据。

# treefix2

**意图**：mask 3x3 + PBRS gamma=1（190 轮）
**状态**：已完成
**时间**：2026-07-21T04:48:51.942Z

路线图瓶颈判断的主要证据：精英轨迹 top-10 全部在起点附近横向游走，无一翻越 Deadtree。

## 命令

```
Stop-Process -Name "GettingOverIt" -ErrorAction SilentlyContinue; Start-Sleep -Seconds 3; chcp 65001 > $null; $env:PYTHONIOENCODING="utf-8"; echo "---launch treefix2: mask 3x3 + PBRS gamma=1, persist-game, 250 iters, target done ~17:05 CST---"; & C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe -m src.training.main_ppo --resume-bc checkpoints/model_iter_0000.pt --no-random-deploy --persist-game --num-agents 10 --steps-per-rollout 500 --max-iterations 250 --checkpoint-dir checkpoints_treefix2 --log-dir logs_treefix2 2>&1
```

## 客观结果

- 恢复到 190 轮指标（iter 1–190）
- mean_reward：末轮 0.3309，峰值 1.081

## 结论

`doc/optimization_roadmap.md` 第一节瓶颈判断的主要证据来源。精英轨迹 top-10（按
base_score）全部在起点附近横向游走，峰值 y ≈ 0~1，**没有任何一条翻越第一个障碍
Deadtree（x ≈ -30）**。

地形 mask 混叠修复 + PBRS gamma=1 都没有改变这个形态，说明问题不在塑形项的数值细节。

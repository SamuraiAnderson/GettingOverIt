# 策略评估

用已有权重再跑一截，看当前策略走到哪。不更新参数。

- `view_elite_trajectories.py` — 用 checkpoint 现场再采一批 rollout，按训练分数排序后可视化。

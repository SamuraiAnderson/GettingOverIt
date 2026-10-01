# 评估脚本

只消费已经训好的权重或归档。固定走廊评估的入口仍是 `src/training/local_evaluation.py`。

- `policy/` — 用权重再采一批，看当前策略走到哪。
- `replay/` — 把记录下来的动作原样送回游戏。
- `present/` — 把归档画成包络、视频或策略所见的 patch。

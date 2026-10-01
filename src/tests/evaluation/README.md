# 评估脚本

只消费已经训好的权重或归档。进游戏、不更新权重的入口是 `src/training/local_evaluation.py`。

- `replay/` — 把记录下来的动作原样送回游戏。
- `present/` — 把归档画成包络、视频或策略所见的 patch。

# 战役控制器

各次战役共用的执行过程。配方、预算阶梯和报告正文不在这里。

- `io.py` — 写 JSON，给出仓库根目录和当前时间。
- `stats.py` — 从 `metrics.csv` 汇总样本，并为已存评估按同一口径重打分。
- `runner.py` — 检查截止和 STOP，拉起子进程，停掉自己的进程树，核对 checkpoint。
- `loop.py` — 拒绝接管已开的游戏，记录源码哈希，结束时恢复 `runtime_config.json`。

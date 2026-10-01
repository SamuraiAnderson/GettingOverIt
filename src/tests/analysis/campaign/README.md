# 战役编排

有界地调度 `main_ppo`：墙钟、多个 seed、到点评估。不实现算法。

- `prepare_v2_formal.py` — 只写出 confirmed-endpoint 正式跑的配置，不启动游戏。
- `run_v2_formal.py` — 跑 12 小时 v2 正式训练：第一段迁移旧权重，之后只续这次自己的 checkpoint。
- `run_corridor_screen.py` — 按六小时预算串行跑走廊筛选并出报告。
- `run_corridor_credit.py` — 在同一预算里对照 C / D / T 三套奖励与时间尺度。
- `run_corridor_formal.py` — 用 8 个玩家把 C / T 复现跑到共同样本终点。
- `run_corridor_sil.py` — 对照纯 PPO 与走廊 SIL，锚点是已保存的正式 96k。
- `extend_corridor_sil.py` — 把 `corridor_sil_20260923` 两个 seed 续到累计 288k 样本。
- `run_corridor_sil_formal.py` — 六小时只跑 SIL 臂，继承原策略的 σ，不强制退火。
- `run_corridor_sil_guard_ab.py` — 96k 对照纯 PPO 与带门控、限幅、预算的 SIL。
- `continue_guard_ab2.py` — 把 `guard_ab2` 的四个 run 各再训两轮并按原口径评估。
- `run_corridor_sil_td_pareto.py` — seed 47、12 小时，从 `p2_contact` 冷启动，精英池按 (t, D) Pareto 保留。

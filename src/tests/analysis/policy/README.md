# 策略与探索

核对网络表征和探索噪声。不定义参考线，不改奖励。

- `verify_policy_experiments.py` — 离线重建归档特征，汇总表征实验是否按约定生效。
- `benchmark_policy_backbones.py` — 在同一批输入上给不同骨干网络的推理计时。
- `viz_exploration_noise.py` — 画出白噪声、OU 和粉噪声的时间相关性。

---
name: visualize-runs
description: >-
  用仓库里已有的评估与出图脚本查看训练结果，不要新写绘图脚本。
  当用户要求可视化、画轨迹、画精英池、包络、动作回放、策略看到的 patch、
  评估 checkpoint、best-motion 或查看 runs 归档时使用。
---

# 用现有脚本看训练结果

Python 用 conda 环境 `getting-over-it-analysis`。命令写成 `conda run --no-capture-output -n getting-over-it-analysis python -m ...`。不要用系统 Python，也不要在对话里另写 matplotlib 脚本。

先分清要看的东西，再选一个入口。

## 进游戏、不更新权重

固定走廊评估，观测管线与 PPO 采集相同。这是当前唯一的在线评估入口。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint runs/<tag>/checkpoints/ppo_iter_XXXX.pt --output runs/<tag>/eval_view --steps 1200 --rounds 2 --num-agents 8
```

已有游戏在跑时加 `--no-launch`。不要调用已删除的 `test_l9_model_eval.py` 或 `view_elite_trajectories.py`。

## 只读归档，不启动游戏

精英池前沿。输入是 `runs/<tag>/elites/`，或单个 `elites_iter_*.npz`。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.present.plot_sil_td_envelope --elites-dir runs/<tag>/elites --output runs/<tag>/td_envelope
```

一条已归档物理轨迹的视频。需要该回合目录里的 `manifest.json`。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.present.render_episode_motion --episode runs/<tag>/checkpoints/eval_xxx/deterministic_000_agent0.json --output runs/<tag>/motion
```

有排名 JSON 时，把 `--episode` 换成 `--ranking <ranking.json> --top 10`。要标出走廊 credited 封顶处的剪枝点，加 `--mark-prune`。

策略在封顶、到达、剪枝帧实际看到的双尺度 patch。`--episode` 与 `--ranking` 二选一。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.present.render_policy_window --episode runs/<tag>/checkpoints/eval_xxx/deterministic_000_agent0.json --output runs/<tag>/patches
```

## 把归档动作送回游戏

选出一条已归档评估 episode，按记录的动作重放。输出目录必须不存在。已有游戏进程时脚本会拒绝接管。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.replay.replay_evaluation_actions --campaign runs/<campaign> --output runs/<new_replay_dir> --fps 30
```

只排名、不启动游戏时加 `--select-only`。`--campaign` 必须含这些脚本能读的 `manifest.json` 和 `checkpoints/eval_*/*_agent*.json`。

## 不要做的事

- 不要为一次查看新写 `runs/**/_plot_*.py`。
- 不要从 checkpoint 现场再采一批来冒充精英池。精英池以 `runs/<tag>/elites/` 为准。
- `local_evaluation --replay` 只离线核对概率，不是游戏里的动作回放。

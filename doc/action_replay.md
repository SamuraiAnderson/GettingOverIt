# 归档动作回放

入口：`src/tests/evaluation/replay/replay_evaluation_actions.py`。

它读取真实评估回合中保存的动作，启动游戏后逐帧发送。不会调用策略生成新动作，也不会重新训练。当前支持正式训练控制器生成的固定起点评估归档。

## 选择“本轮最佳”

默认扫描指定 campaign 下所有已保存评估回合，按**累计有效走廊进度**降序排列；同分时优先末端投影进度，再优先较短回合。不使用 1 秒保持、10 步保持、最高高度等条件。

累计有效进度只计走廊内首次推进，并沿用参考轨迹置信度权重；这是当前引导奖励中的进度口径，不是全部实际移动路程。第一名仅代表此指标下的已记录最佳回合，不保证末端位置最好或整条路线已完成。`ranking.json` 同时保留末端进度和走廊外时间比例，便于检查这个区别。

训练归档没有保存原始 33D 状态，因此本入口的选择范围是评估回合。本轮正式训练共有 512 条完整评估轨迹。

## 回放过程

保留原来的 8 个 agent 编号，按对应 manifest 执行 warmup、reset 和初态扰动。同批其他 agent 的动作也一并恢复；已终止的 agent 后续执行零动作，与采集逻辑一致。

目标回合之前的评估批次也按原顺序重放，尽量重建求解器接触历史。完整历史物理快照未保存，仍可能产生初态误差和接触混沌放大。回放结果保存位置偏差和路线进度，不能把“动作相同”当作“运动逐帧相同”。

第一名若是原始 Player（agent 0），使用原生跟随相机；其他编号保持自由相机。当前目标回合以每秒最多 30 次 STEP 播放，1,200 个物理步约用 40 秒墙钟时间，物理时间仍约 10 秒。回放结束后关闭自身游戏并恢复运行配置。

## 使用

输出目录必须是新的，防止覆盖之前的回放证据。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.replay.replay_evaluation_actions --campaign runs/corridor_formal_20260922 --output runs/best_action_replay_20260923 --fps 30
```

添加 `--select-only` 只生成排名和选中回合信息，不启动游戏。

产物包括 `selected.json`、`ranking.json`、`original_episode.json`、`replayed_batch_actions.npy`、`replay_states.npy`、`result.json`、`comparison.gif`、`comparison.png`。

对比动画根据原始记录和实际回放采集的状态绘制身体/锤子锚点示意，不是游戏屏幕录像。图像中的物理时间来自游戏诊断包；JSON 中保存实际执行的动作数、初始/末端/最大位置误差和首个超过 10 厘米偏差的物理步。

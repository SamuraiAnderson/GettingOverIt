# 多世界运行与实验记录

独立世界已进入正式运行接口，见 [统一接口](../../../../doc/multiplayer_runtime.md)。
这里的在线入口使用隔离副本 `artifacts/reset_game` 中的同一个 `GameRuntime_v2.dll`，
不再构建或注入 `IsolatedWorldExperiment.dll`。运行前须关闭已有游戏。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python src/tests/control_interaction/world_isolation/run_policy_isolated_8players.py --out runs/isolated_policy_new --steps 1200 --fps 30
conda run --no-capture-output -n getting-over-it-analysis python src/tests/control_interaction/world_isolation/check_isolated_policy_run.py runs/isolated_policy_new
conda run --no-capture-output -n getting-over-it-analysis python src/tests/control_interaction/world_isolation/run_isolated_worlds.py --out runs/isolated_actions_new --worlds 2 --steps 1200 --resets 2
```

权重展示默认使用 `ppo_iter_0269.pt`，复用生产评估与观测构建；冷启动后不进行
额外预热、reset 或投放，读取起点时保存一次不推进物理的快照。正常结束保留
窗口，`--hidden` 在结束后关闭。左键拖动、滚轮缩放；同轨迹 player 会重叠。
固定动作入口接受 `--actions`，`--other-zero` 可改变其他世界的动作来检查隔离。
旧 `--startup-sync` / `--scene-sync-only` 实验开关已移除，正式接口使用统一边界。

`build_probes.ps1` 只编译可选碰撞/回调探针。两探针现在以统一批量 STEP 计步，
`replay_clone_divergence.py` 也会使用正式多世界后端；不再运行共享求解器对照。
探针由临时 `clone_probe.json` 激活，在线回放本身不需要额外 DLL。

`summarize_clone_divergence.py`、`summarize_isolated_worlds.py` 和旧版兼容检查
面向原实验归档；复算需要本地完整原始数据。历史证据仍保存在
`runs/clone_divergence_step31_20261002`、`runs/isolated_worlds_20261002`、
`runs/policy_0269_isolated_8players_20261002`。旧共享求解器代码及反射实验驱动
可在提交 `ab387bd` 查阅；它们不是当前入口。

旧单世界与多世界的同步边界差异，现由所有世界（含 1 player）统一执行末尾
SyncTransforms 处理。不要将新接口结果和旧未同步参考混成同一种初始化协议。

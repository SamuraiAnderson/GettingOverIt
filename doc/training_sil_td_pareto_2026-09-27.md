# 长时间训练：seed 47，锅心走廊 SIL，(t, D) Pareto

已于 **2026-09-26 14:55:08（北京时间）**启动，硬截止 **2026-09-27 02:55:08**。96k 训练完成后控制器曾因参考线路径审计停掉；已允许游戏世界参考线并续跑。

- 输出：`runs/corridor_sil_td_pareto_20260927b`
- 训练目录：`runs/corridor_sil_td_pareto_20260927b_S_s47`
- 状态：`runs/corridor_sil_td_pareto_20260927b/status.json`
- 只跑 seed 47，S 臂：C 配方 + 防线化 corridor SIL + `td_pareto`
- 初始化：`runs/p2_contact/checkpoints/ppo_iter_0335.pt`，**不**加载 `sil_seed_590.npz`
- 计分：`pot_visible_center_v1` + `tcp_protocol=anchor35` + `reference_trajectory_game_world.json`
- 不续 20260926 player-origin 轮，不混旧 `corridor_progress`
- 预算 12 小时；先到 96k 再按 96k 台阶续训
- 游戏在隔离桌面 `GoiRuntime` 启动

当次控制器已删除。停止方式不要再按 `STOP` 文件执行。

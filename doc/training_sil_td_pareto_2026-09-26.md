# 长时间训练：seed 47，走廊 SIL，(t,D) Pareto

已于 **2026-09-26 12:57:28（北京时间）**启动，硬截止 **2026-09-27 00:57:28**。控制器独立后台运行，不依赖对话继续派发。首轮已连上隔离桌面游戏，8 agents 开始采集。

- 输出：`runs/corridor_sil_td_pareto_20260926`
- 训练目录：`runs/corridor_sil_td_pareto_20260926_S_s47`
- 状态：`runs/corridor_sil_td_pareto_20260926/status.json`
- 只跑 seed 47，S 臂：C 配方 + 防线化 corridor SIL + `td_pareto`
- 不跑 seed 31/61，不跑纯 C，不跑 T，不从 20260926 MAP-Elites 96k 续训
- 初始化：`runs/p2_contact/checkpoints/ppo_iter_0335.pt`，SIL 种子 `sil_seed_590.npz`
- 路线比较仍是 `legacy33` + `player_origin`。锅锚点测量见 `runs/tas_anchor_calib_20260926/RESULT.md`，profile 未 verified，本轮不启用 35D
- 预算 12 小时；先到 96k 再按 96k 台阶续训。预算决策不读评估成绩
- 游戏在隔离桌面 `GoiRuntime` 启动

当次控制器已删除。停止方式不要再按 `STOP` 文件执行。

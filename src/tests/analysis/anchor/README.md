# 锅心锚点

把已经有的参考轨迹收成身体局部锅心，并写出训练用的游戏世界轨迹。不从视频里找玩家，也不做走廊计分。

拟合实现在 `training/anchor_fit.py`。这里的命令行只调用它，通过也不把 profile 标成已验证。

- `run_tas_anchor_validation.py` — 在实机截图、检测锅心、拟合并留出。端口读 `project.json`。
- `fit_reference_anchor.py` — 对已有样本拟合一份未验证的身体局部锚点。
- `reproject_tas_live_camera.py` — 像素仍用 TAS 检测，反投影改用实机出生相机，写出 `reference_trajectory_game_world.json`。
- `migrate_reference_archive.py` — 把旧走廊归档迁到新参考指纹；缺 35D 锚点时不填补。

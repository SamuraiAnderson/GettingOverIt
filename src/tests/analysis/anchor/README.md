# 锚点与走廊

把已经有的参考轨迹收成锅心锚点和走廊计分。不负责从视频里找出玩家。

- `collect_anchor_calibration.py` — 在给定传送点采集同一帧的原生姿态和相机。
- `fit_reference_anchor.py` — 拟合一份未验证的身体局部锚点。
- `run_tas_anchor_validation.py` — 截图、检测锅心、拟合并留出；通过也不把 profile 标成已验证。
- `reproject_tas_live_camera.py` — 像素仍用 TAS 检测，反投影改用实机出生相机。
- `migrate_reference_archive.py` — 把旧走廊归档迁到新参考指纹；缺 35D 锚点时不填补。

# CV 参考轨迹草稿

请从 `reference_trajectory.png`、`reference_overlay.mp4` 和 `quality_report.json` 查看结果。

**地形拟合稿**仍是未经真实位置验证的草稿：`reference_trajectory.json` / `trajectory.csv`。`accepted_candidate=true` 表示有角色和镜头支持的世界坐标候选；`point_source=interpolated` 是不超过 0.3 秒、通过运动与地形检查的补全。长缺口留空，`training_ready=false`。

**训练默认改读** `reference_trajectory_game_world.json`：TAS 形状先按实机出生相机反投影，再平移到 Unity `TransformPoint(b)` 的游戏世界锅心。计分、参考线和评估轨迹都用这个世界点，不再用相机坐标或 Player 原点。旧地形拟合稿和 live-camera 中间稿保留。复现：`python -m src.tests.analysis.reproject_tas_live_camera`。

`camera_*.csv` / `player_*.csv` 是可能包含误匹配的中间结果。`diagnostics/` 是早期失败方案和旧截图的归档，请勿用来训练。

复现方法与坐标定义见项目 `doc/video_reference_extraction.md`。

Git 保存轨迹 JSON、CSV、质量报告和结果图；原始视频、叠加视频及下载的 ONNX 权重仅保留在本地。
重新提取前需准备原视频和模型文件，位置与来源见上述文档。

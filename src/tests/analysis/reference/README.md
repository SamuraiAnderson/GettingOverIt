# 参考轨迹

得到或阅读一条参考轨迹：视频草稿、人工操作、TAS 影片，以及对已有轨迹的进度和过树判断。不做锚点拟合。

训练默认读的是 `reference_trajectory_game_world.json`（锚点目录里的实机相机反投影）。视频管线产出的是带缺口的草稿，`training_ready` 为 false。

视频只有一个入口：`python -m src.tests.analysis.reference.extract_video_reference --run`。下面的 `video_*.py` 是它按顺序调用的步骤，不单独当工具跑。`test_video_reference.py` 检查配准和缺失值没有被静默填上。

- `extract_video_reference.py` — 对速通视频做整段离线跟踪草稿，对不上的帧保持缺失。
- `video_player_track.py` — 用金锅外观初始化，再做带掩码的外观跟踪。
- `video_player_color.py` — 在 HSV 里追金色团块，丢失时借用运动提议拉回。
- `video_player_rigid.py` — 用 NanoTrack 正向、反向各跟踪一次刚体锅。
- `video_player_fusion.py` — 把刚体跟踪框收成锅心，去掉向上吃进躯干的部分。
- `video_follow_camera.py` — 用光流外推玩家在画面上的位置，不是检测器。
- `video_translation_track.py` — 修补摄像机速度里的短缺口，保留原始光流。
- `video_edge_registration.py` — 把视频边缘配准到碰撞体并集。
- `video_camera_register.py` — 把运动估计重新钉到地形图上，对不上的帧继续留空。
- `video_global_registration.py` — 在整张地图上找候选，把漂掉的相机运动重新定位。
- `video_path_completion.py` — 在世界坐标里补短缺口，并检查是否穿进地形。锚点反投影也调用它。
- `video_reference_export.py` — 导出观测，并为短缺口插值打上明确标签。
- `record_reference_trajectory.py` — 帧级步进加真实鼠标，录一条人工示范轨迹。
- `ltm_inspect.py` — 解开 libTAS 影片，查看逐帧键鼠输入。
- `calibrate_progress_wx.py` — 从已有轨迹反推横向进度权重 `progress_wx`。
- `analyze_reference_fling.py` — 判断已有参考轨迹的过树段是弹道甩飞还是勾枝、撬杆。

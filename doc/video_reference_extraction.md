# 速通视频的 CV 地形配准与角色轨迹草稿

入口：`python -m src.tests.analysis.reference.extract_video_reference`。

这条流程已实现离线视频读取、静态地形边缘匹配、角色候选检测和轨迹绘制。当前结果是**不完整、未经真实坐标标注验证的候选轨迹**，不是全程追踪成功，也不能作为精确的监督真值。显式启用的 [PPO 走廊奖励](reference_corridor_reward.md) 可将其连续段作为粗略路线做实验，保留缺口和来源标记。完整性与筛选覆盖率见输出目录的 `quality_report.json`。

## 输入与输出

- 原视频：`C:/Users/Symbol/Downloads/37258923076-1-192.mp4`。
- 项目内副本：`src/Data/video_reference/37258923076-1-192/source.mp4`，准备阶段验证 SHA-256，一致性不满足时拒绝覆盖。
- 地形：`checkpoints/environment.json`，使用导出的静态 Collider 多边形并集，排除 Snake。这里是物理碰撞地形，不是渲染截图；装饰、阴影、背景视差和物理轮廓偏差会影响配准。
- 本视频专用配置：1280×720、约 30 fps，处理帧 203–1303；已知起始区域、初始角色大致屏幕位置和金缸外观属于初始化先验。更换视频需重新配置，不能声称无需先验的通用识别。

输出均位于 `src/Data/video_reference/37258923076-1-192/`：

- `reference_trajectory.png`：地形上的候选位置与配准质量曲线，青色叉号单独标记短缺口插值；长缺口断开。
- `reference_overlay.mp4`：原视频叠加结果，青色为地图投影，黄色为角色候选，紫色菱形为可投影的插值点。此视频不含音轨。
- `trajectory.csv`：每帧位置、筛选状态和质量信号。`observed_world_x/y` 保存原始候选，`world_x/y` 包含获准的补全，`point_source` 区分 `observed`、`interpolated`、`missing`。
- `reference_trajectory.json`：分段坐标与来源哈希，明确带有 `training_ready: false`。
- `quality_report.json`：处理范围、各阶段候选覆盖率及误差摘要。
- `interpolation_report.json`：每个缺口的补全/拒绝原因，以及隐藏已检测点后的恢复误差。
- `rigid_tracker_report.json`：缸体跟踪配置、模型来源与 SHA-256。
- `camera_*.csv`、`player_*.csv`、`global_candidates.json`：中间候选，**不能直接当作可靠轨迹使用**。

## 方法

1. 将地形多边形并集合成为边缘图。初始地形位置通过已知起始区域内的边缘距离搜索求得。
2. 在相邻帧上做前向/反向光流检查，把不同平移速度的特征点分组，用空间分布较广的前景运动估计镜头位移。由于背景存在视差，仅采用最大光流内点组会产生累计漂移。
3. 使用带方向信息的 Chamfer 距离进行局部地形配准，并以稀疏全图搜索产生重新定位候选。筛选边界点数、边缘内点比例、残差、空间分布和与运动估计的一致性。近似直线轮廓不能单独充分约束二维镜头位置。
4. 角色侧优先跟踪相对刚性的缸体，初始化框不包含手臂和锤子。采用 OpenCV NanoTrack 的本地预训练权重，分别正向、反向跟踪；结合颜色轮廓与两方向位置差筛选候选。通用跟踪框有时向上扩张到躯干，因此用早期可靠颜色分割拟合框尺寸到缸中心的偏移，避免将框中心混同于缸中心。缸仍会旋转、变暗和被遮挡，不能把“刚性”当作像素外观恒定。
5. 镜头运动积分由地形配准锚点校正。最多跨越 3 秒的锚点间隔，且要求光流连续；光流速度仅允许补全不超过 6 帧、方向与幅值一致的短缺口。`camera_source` 区分直接地图锚点与锚点之间的运动估计。这不是把所有中间帧都声称为独立地图匹配。
6. 在世界坐标中对两端有观测的短缺口做分段线性插值，上限 0.3 秒，不外推、不改动观测点。检查速度、缺口前后速度突变，以及每 0.04 世界单位的地形穿越；深入地形超过 0.2 世界单位时拒绝补全。这些阈值是该视频的质量筛选参数，并非游戏动力学定律。避免使用高阶曲线把接触附近的折点自动抹平。

验证插值时，隐藏已有观测中的 2/4/6 帧，再与恢复位置对照。该误差衡量对已有 CV 观测的恢复能力，**不是真实游戏坐标误差**。`interpolation_report.json` 保存完整样本量、误差分位数与拒绝记录。

像素转世界坐标：`world_x = camera_x + (pixel_x - 640) / scale`，`world_y = camera_y - (pixel_y - 360) / scale`。

这里的位置是**缸的可见中心**，尚未标定为游戏内 `Player Transform` 原点。它也不提供鼠标 action、锤尖接触事件或真实物理速度。不能仅凭边缘残差宣称米级或厘米级坐标精度。

## 运行与检查

从项目根目录执行：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.reference.extract_video_reference --run
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.reference.test_video_reference
```

仅重绘已有中间结果：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.reference.extract_video_reference --export
```

依赖为当前分析环境已有的 OpenCV、NumPy、SciPy、pandas、matplotlib，以及放在 `src/Data/video_reference/models/nanotrack_v2/` 的两个本地 ONNX 文件。模型来自 [OpenCV TrackerNano 文档指定的仓库](https://docs.opencv.org/4.x/d8/d69/classcv_1_1TrackerNano.html)，当前机器已下载，运行脚本本身不会联网或下载模型。

Git 不包含原视频、叠加视频和下载的 ONNX 权重。新检出目录若需重新提取，将同一原视频放入上述项目副本位置（或原始下载位置供 `prepare()` 复制），并按模型目录 `README.md` 的来源和 SHA-256 准备权重。已有轨迹 JSON 可直接供走廊模式读取，不依赖视频或 ONNX；结果图、CSV 和质量报告随代码入库。

脚本不会打开图片窗口、启动游戏、修改奖励或开始训练。按用户最新要求，本轮改进与验证只读取 CV 数值，没有人工打开帧检查；更早探索遗留的截图不构成坐标真值。

数值测试覆盖合成地形配准、投影逆变换、镜头速度短缺口处理、插值来源标记，以及长缺口、穿墙、过快跳跃不能被自动补线。实际视频仍缺少坐标真值验证；角色出画面和长时间缺乏可配准地形时，仍须保留缺失或不确定性。

# GameRuntime

功能: 数据采集

功能性运行时代码（BepInEx 插件与子模块），按功能分组：

- Core: 插件入口与编排（GoiHitboxLogger）
- Monitoring: 帧级监控（StateMonitor, ActionMonitor）
- Tracking: 高频采样与数据导出（ContinuousTracker）
- PlayerAnalysis: Player 对象检查与报告（PlayerAnalyzer）
- Colliders: 碰撞体收集与导出（HitboxCollector）

输入/输出：
- 输入：游戏指令序列（鼠标移动为唯一控制信号）
  频率：固定 30Hz（Getting Over It 固有刷新率；采样与指令对齐为 30Hz）
  数据格式（MouseInput）:
  - mouseX: float（屏幕坐标X）
  - mouseY: float（屏幕坐标Y）
  - timestamp: float（游戏时间秒）
  数据传入形式：
  - 运行态：由 Unity 在 Update 中采样 `Input.mousePosition`（30Hz）
  - 测试态：由外部生成命令序列并注入（见“外部测试指令”规范）

外部测试指令（规范约定）：
- 路径：`src/Data/input_commands.csv`
- 频率：固定 30Hz（逐行一帧）
- 列：`timestamp,mouseX,mouseY`
- 时间：`timestamp` 为相对起点秒，可由 Unity 启动测试时重新对齐；若为空则按行序等间隔 1/30s 递增
- 说明：此为测试数据生成与注入的统一规范，代码侧可在后续实现“测试模式”读取并回放（当前实现为实时采样）

- 输出：游戏采集数据（state, action/输入, 环境元数据）
  数据格式（CSV by ContinuousTracker.ExportTrackingData）:
  - timestamp, deltaTime
  - mouseX, mouseY
  - playerX, playerY, velocityX, velocityY, hammerAngle, hammerAngularVel, isGrounded
  - mouseDeltaX, mouseDeltaY, mouseMoveDistance, mouseMoveSpeed
  - positionDelta, velocityDelta, angleDelta, physicsResponseDelay
  数据保存/传递形式：
  - CSV 保存目录：`TrackingDump/ContinuousTracking_yyyyMMdd_HHmmss.csv`
  - 状态/动作快照：`StateDump/`, `ActionDump/`（文本）
  - Player 分析：`PlayerAnalysis/`（文本）
  - 碰撞体导出：`HitboxDump/`（文本）
  - IPC：文件系统 JSON（见下）

IPC（JSON 文件协议，目录 `src/Data`）：
- auto_mode_signal.json → Unity（Core）
  - enable_auto_mode: bool
  - duration: float（秒）
  - frequency: int（Hz）
  - timestamp: float
- auto_mode_response.json ← Unity（Core）
  - response: string（如 "data_collection_started"）
  - scene_name: string
  - timestamp: float
- collection_signal.json → Unity（Tracking）
  - signal: string（"start"|"stop"）
  - session_id: int
  - duration: float（秒）
  - frequency: int（Hz，测试与训练统一为30）
  - timestamp: float
- unity_response.json ← Unity（Tracking）
  - response: string（"started"|"completed"）
  - session_id: int
  - data_points: int
  - timestamp: float

统一数据集（Python 侧保存，目录 `src/Data/AI_Training/`）：
- 文件：`*_ai_training.npz`；合并：`unified_training_dataset.npz`
- 键：
  - observations: float32 [N, ObsDim]（建议: [playerX, playerY, velocityX, velocityY, hammerAngle, hammerAngularVel, isGrounded]）
  - actions: float32 [N, 2]（[mouseX, mouseY] 或其归一化值）
  - timestamps: float32 [N]
  - meta: dict（JSON 序列化后存储，含频率=30Hz、版本、源CSV路径等）
- 归一化：训练前建议将 actions/observations 映射至 [-1, 1]（Python 侧完成，Unity 不做归一化写入）。

目录约定（相对于游戏根目录的 `…/Getting Over It/GettingOverIt_Data`）：
- 导出目录（Unity 写入）：位于 `…/Getting Over It/` 同级下的 `TrackingDump/`, `StateDump/`, `ActionDump/`, `PlayerAnalysis/`, `HitboxDump/`
- IPC 目录（JSON）：`src/Data/`

维护建议：
- 严格“按功能分组”；新增功能先建模块目录与 readme
- 编排逻辑仅在 Core；业务逻辑各自归属模块
- 文件路径拼接统一使用嵌套 `Path.Combine`（仅双参）
- 日志应高信噪比，避免在 Update 中大量循环打印

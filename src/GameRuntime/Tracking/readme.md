# Tracking

- ContinuousTracker：
  - 高频采样（默认50Hz，可配置）
  - 输入：鼠标位置（mouseX, mouseY）
  - 状态：playerX/Y、velocityX/Y、hammerAngle、hammerAngularVel、isGrounded
  - 输出：CSV 导出至 TrackingDump/；自动化采集响应写入 src/Data/unity_response.json
  - 支持自动化采集：读取 src/Data/collection_signal.json

维护建议：
- 采样结构体尽量稳定，便于下游 Python 统一解析
- 只保留必要统计，复杂分析在 Python 侧完成

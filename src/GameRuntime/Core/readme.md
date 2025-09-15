# Core

- GoiHitboxLogger 主插件入口，负责：
  - 插件生命周期、日志
  - 热键/GUI 提示
  - 自动化流程调度（Loader→Mian 场景、信号收发）
  - 统一初始化与更新各功能模块（Monitoring/Tracking/PlayerAnalysis/Colliders）

依赖关系：
- 读：`Monitoring/StateMonitor`, `Monitoring/ActionMonitor`, `Tracking/ContinuousTracker`, `PlayerAnalysis/PlayerAnalyzer`, `Colliders/HitboxCollector`

产出：
- 在 `src/Data` 下读写 JSON 信号/响应
- 触发各模块的导出功能

维护建议：
- 保持入口类仅做编排，不写具体业务逻辑
- 新功能先抽象为模块，再由核心调度



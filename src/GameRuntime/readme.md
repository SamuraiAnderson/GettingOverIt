# GameRuntime

功能性运行时代码（BepInEx 插件与子模块），按功能分组：

- Core: 插件入口与编排（GoiHitboxLogger）
- Monitoring: 帧级监控（StateMonitor, ActionMonitor）
- Tracking: 高频采样与数据导出（ContinuousTracker）
- PlayerAnalysis: Player 对象检查与报告（PlayerAnalyzer）
- Colliders: 碰撞体收集与导出（HitboxCollector）

输入/输出：
- 输入：运行时场景对象（Player/Mountain 等）
- 输出：GUI 提示、日志、文本/CSV 导出、src/Data 下的信号/响应

维护建议：
- 严格“按功能分组”；新增功能先建模块目录与 readme
- 编排逻辑仅在 Core；业务逻辑各自归属模块

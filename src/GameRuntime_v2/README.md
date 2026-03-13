# GameRuntime_v2

Getting Over It 强化学习运行时系统 - 重构版，采用分层架构和接口驱动设计。

## 🚀 快速开始

1. **数据采集**（首次使用）
   ```bash
   cd C:\Users\Symbol\aCodes\goi-rl
   python switch_mode.py  # 选择 [1] 数据采集模式
   # 然后启动游戏，自动采集 Mountain 碰撞箱
   ```

2. **AI 训练**（待实现）
   ```bash
   python switch_mode.py  # 选择 [2] 游戏运行模式
   python main_train.py   # 启动训练
   ```

详见 **[QUICKSTART.md](QUICKSTART.md)**

## 🎯 三种运行模式

| 模式 | 说明 | 状态 |
|------|------|------|
| **DataCollection** | 采集 Mountain 碰撞箱等静态数据 | ✅ 已实现 |
| **GameRuntime** | AI 训练交互（UDP 通信） | 🔧 框架已搭建 |
| **GameTesting** | 游戏交互性测试 | 🔧 框架已搭建 |

详见 **[Core/ModeManager.md](Core/ModeManager.md)**

## 📁 目录结构

| 文件夹 | 说明 |
|--------|------|
| **Core/** | 核心基础设施（接口、事件、工具、配置、模式管理） |
| **PlayerControl/** | Player 控制服务（输入注入、状态采集） |
| **Communication/** | UDP 通信服务（接收动作、发送状态） |
| **ColliderCollection/** | 碰撞箱采集服务（环境和 Player 碰撞箱） |
| **Testing/** | 测试框架（单元测试、集成测试） |
| **Plugins/** | BepInEx 插件入口 |

详细说明见各子文件夹的 readme.md。

## 📚 文档

- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始指南
- **[MODE_SYSTEM_SUMMARY.md](MODE_SYSTEM_SUMMARY.md)** - 模式管理系统总结
- **[FILE_OVERVIEW.md](FILE_OVERVIEW.md)** - 文件结构总览
- **[Core/ModeManager.md](Core/ModeManager.md)** - 模式管理详细文档



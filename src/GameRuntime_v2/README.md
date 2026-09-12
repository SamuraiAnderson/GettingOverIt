# GameRuntime_v2

Getting Over It 强化学习运行时系统 - 重构版，采用分层架构和接口驱动设计。

## 🚀 快速开始

1. **数据采集**（首次使用）
   ```bash
   cd C:\Users\Symbol\aCodes\GettingOverIt\src
   python main.py  # 选择数据采集模式，启动游戏自动采集 Mountain 碰撞箱
   ```

2. **AI 训练**
   ```bash
   # 写游戏运行模式信号 + 启动游戏 + PPO 训练（由 rollout 统一编排）
   python -m src.training.main_ppo --run-dir my_experiment --persist-game
   ```

详见 **[QUICKSTART.md](QUICKSTART.md)** 与仓库根的 [`usage.md`](../../usage.md)。

## 🎯 三种运行模式

| 模式 | 说明 | 状态 |
|------|------|------|
| **DataCollection** | 采集 Mountain 碰撞箱等静态数据 | ✅ 已实现 |
| **GameRuntime** | AI 训练交互（TCP 帧级步进，核心） | ✅ 已实现 |
| **GameTesting** | 游戏交互性 / 物理能力探测（键盘调试） | ✅ 已实现 |

详见 **[Core/ModeManager.md](Core/ModeManager.md)**

## 📁 目录结构

| 文件夹 | 说明 |
|--------|------|
| **Core/** | 核心基础设施（接口、事件、工具、配置、模式管理、帧级步进） |
| **PlayerControl/** | Player 控制服务（反射注入 + Rewired 拦截、状态采集、复制体管理） |
| **Communication/** | TCP 双线程步进服务端（接收动作、发送状态） |
| **ColliderCollection/** | 碰撞几何采集与可视化（环境和 Player 碰撞箱） |
| **Camera/** | 自由相机控制（`FreeCameraController`） |
| **Testing/** | 调试工具（`PlayerDebugTool`、输入特性探测） |
| **Plugins/** | BepInEx 插件依赖入口 |

详细说明见各子文件夹的 README。

## 📚 文档

- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始指南
- **[Core/ModeManager.md](Core/ModeManager.md)** - 模式管理详细文档
- **[../../doc/entrypoints.md](../../doc/entrypoints.md)** - 交互链路、运行模式、协议、状态维度（权威）

> ⚠️ `FILE_OVERVIEW.md` 已过时，以 [`doc/entrypoints.md`](../../doc/entrypoints.md) 为准。



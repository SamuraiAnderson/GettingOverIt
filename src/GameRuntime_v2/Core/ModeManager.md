# 模式管理系统

## 概述

模式管理系统支持三种互斥的游戏运行模式，可通过配置文件或信号文件灵活切换。

## 三种运行模式

### 1. 数据采集模式 (DataCollection)
- **用途**: 前期采集 Mountain 碰撞箱等静态数据
- **启动**: 仅初始化碰撞箱采集服务
- **通信**: 不需要 TCP 通信

### 2. 游戏运行模式 (GameRuntime)
- **用途**: AI 训练交互
- **启动**: 初始化 Player 控制、TCP 步进服务端、状态采集等
- **通信**: TCP 阻塞式请求-响应（接收动作、发送状态，帧级步进，默认 :9000）

### 3. 游戏测试模式 (GameTesting)
- **用途**: 交互性测试和调试
- **启动**: 初始化 Player 控制、测试工具等
- **通信**: 可选

## 使用方法

### 方法 1: 配置文件控制（推荐用于固定场景）

编辑 `GoiData/runtime_config.json`:

```json
{
  "mode": "DataCollection"
}
```

可选值：`DataCollection`、`GameRuntime`、`GameTesting`

### 方法 2: 信号文件控制（推荐用于 Python 动态切换）

使用 Python 脚本：

```python
from src.start import GameModeController

controller = GameModeController()

# 设置数据采集模式
controller.set_data_collection_mode()

# 设置游戏运行模式
controller.set_game_runtime_mode()

# 设置游戏测试模式
controller.set_game_testing_mode()
```

**优先级**: 信号文件 > 配置文件

## 信号文件说明

信号文件位于 `GoiData/ControlSignals/` 目录：

- `mode_data_collection.signal` - 数据采集模式
- `mode_game_runtime.signal` - 游戏运行模式
- `mode_game_testing.signal` - 游戏测试模式

**注意**: 信号文件在读取后会自动删除，需要每次启动前重新创建。

## C# 集成示例

```csharp
using GoiRuntime.Core;
using GoiRuntime.Core.Configuration;

public class GameRuntimeManager : BaseUnityPlugin
{
    private ModeManager modeManager;
    private RuntimeConfig config;

    void Awake()
    {
        // 加载配置
        config = RuntimeConfig.Load();
        
        // 初始化模式管理器
        modeManager = new ModeManager();
        modeManager.Initialize(config);
        
        // 根据模式初始化不同服务
        if (modeManager.IsDataCollectionMode())
        {
            InitializeDataCollectionServices();
        }
        else if (modeManager.IsGameRuntimeMode())
        {
            InitializeGameRuntimeServices();
        }
        else if (modeManager.IsGameTestingMode())
        {
            InitializeGameTestingServices();
        }
    }
}
```

## 工作流示例

### 场景 1: 初次采集数据
```bash
# 1. Python 设置模式
python -c "from src.start import GameModeController; GameModeController().set_data_collection_mode()"

# 2. 启动游戏
# Unity 自动读取信号，进入数据采集模式

# 3. 采集完成后，数据保存到 GoiData/Colliders/
```

### 场景 2: AI 训练
```bash
# 1. Python 设置模式
python -c "from src.start import GameModeController; GameModeController().set_game_runtime_mode()"

# 2. 启动游戏
# Unity 进入游戏运行模式，启动 TCP 步进服务端

# 3. 启动训练脚本
python -m src.training.main_ppo
```

### 场景 3: 调试测试
```bash
# 1. Python 设置模式
python -c "from src.start import GameModeController; GameModeController().set_game_testing_mode()"

# 2. 启动游戏
# Unity 进入测试模式，可以手动测试控制逻辑
```


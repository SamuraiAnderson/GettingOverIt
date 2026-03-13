# GameRuntime_v2 文件总览

## 📁 目录结构

```
GameRuntime_v2/
├── README.md                              # 项目总览
├── Core/                                  # 核心基础设施
│   ├── README.md
│   ├── ModeManager.cs                     # ✅ 模式管理器
│   ├── ModeManager.md                     # ✅ 模式管理文档
│   ├── GameRuntimeManager.cs              # ✅ 运行时管理器（BepInEx 插件入口）
│   ├── Interfaces/                        # 接口定义
│   │   ├── README.md
│   │   ├── IPlayerServices.cs             # ✅ Player 服务接口（输入、状态、复制体）
│   │   ├── ICommunication.cs              # ✅ 通信接口（UDP 收发）
│   │   ├── IColliderService.cs            # ✅ 碰撞箱采集接口
│   │   └── ITesting.cs                    # ✅ 测试接口
│   ├── Events/                            # 事件系统
│   │   ├── README.md
│   │   ├── EventBus.cs                    # ✅ 事件总线
│   │   └── GameEvents.cs                  # ✅ 事件常量定义
│   ├── Utilities/                         # 工具类
│   │   ├── README.md
│   │   ├── ReflectionHelper.cs            # ✅ 反射辅助
│   │   ├── InputValidator.cs              # ✅ 输入验证（[10,100]）
│   │   ├── PathManager.cs                 # ✅ 路径管理
│   │   └── SignalFileHelper.cs            # ✅ 信号文件工具
│   └── Configuration/                     # 配置管理
│       ├── README.md
│       └── RuntimeConfig.cs               # ✅ 运行时配置（含模式枚举）
├── PlayerControl/                         # Player 控制服务
│   ├── README.md
│   ├── PlayerInputService.cs              # (待实现) 输入注入
│   ├── PlayerStateService.cs              # (待实现) 状态采集
│   └── DuplicateManager.cs                # (待实现) 复制体管理
├── Communication/                         # UDP 通信服务
│   ├── README.md
│   ├── UDPReceiveService.cs               # (待实现) UDP 接收
│   ├── UDPSendService.cs                  # (待实现) UDP 发送
│   └── UDPProtocol.cs                     # (待实现) 协议实现
├── ColliderCollection/                    # 碰撞箱采集服务
│   ├── README.md
│   ├── ColliderCollectorBase.cs           # ✅ 碰撞箱采集基类
│   ├── EnvironmentColliderService.cs      # ✅ 环境碰撞箱
│   ├── PlayerColliderService.cs           # ✅ Player 碰撞箱
│   └── ColliderDataFormatter.cs           # ✅ 数据格式化
├── Testing/                               # 测试框架
│   ├── README.md
│   ├── TestRunner.cs                      # (待实现) 测试运行器
│   ├── UnitTests/                         # (待实现) 单元测试
│   ├── IntegrationTests/                  # (待实现) 集成测试
│   └── DebugTools/                        # (待实现) 调试工具
└── Plugins/                               # BepInEx 插件
    └── README.md
```

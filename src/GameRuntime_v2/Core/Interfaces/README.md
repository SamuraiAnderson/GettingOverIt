# Core/Interfaces - 接口定义

> 2026-10-02：批量运行已统一为独立物理世界，当前接口与 reset 契约见[多 player 运行接口](../../../../doc/multiplayer_runtime.md)。下文旧复制体/刚体快照流程仅供历史参考。

所有服务的接口契约定义。

## 📄 文件列表

| 文件 | 说明 |
|------|------|
| **IPlayerServices.cs** | Player 相关服务接口（输入控制、状态采集、复制体管理） |
| **IStepServer.cs** | TCP 帧级步进服务端接口（`StartListening` / `TryGetCommand` / `SendResponse`） |
| **IGameControl.cs** | 游戏控制接口（暂停/恢复、时间缩放、逐帧步进） |
| **ITesting.cs** | 测试相关接口（测试用例、测试运行器、调试工具） |


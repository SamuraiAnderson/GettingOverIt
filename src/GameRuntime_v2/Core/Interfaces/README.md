# Core/Interfaces - 接口定义

所有服务的接口契约定义。

## 📄 文件列表

| 文件 | 说明 |
|------|------|
| **IPlayerServices.cs** | Player 相关服务接口（输入控制、状态采集、复制体管理） |
| **IStepServer.cs** | TCP 帧级步进服务端接口（`StartListening` / `TryGetCommand` / `SendResponse`） |
| **IGameControl.cs** | 游戏控制接口（暂停/恢复、时间缩放、逐帧步进） |
| **ITesting.cs** | 测试相关接口（测试用例、测试运行器、调试工具） |


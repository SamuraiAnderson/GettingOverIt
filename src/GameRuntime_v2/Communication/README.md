# Communication — TCP 阻塞步进协议

## 架构概述

Python 端作为 **TCP 客户端**，C# 端（Unity BepInEx）作为 **TCP 服务端**。
每个 RL step 对应一次完整的请求-响应交互，保证帧级别同步。

## 文件

| 文件 | 说明 |
|------|------|
| **TcpStepServer.cs** | TCP 服务端实现，监听 localhost:9000 |

## 线程模型

```
Unity 主线程                    后台 Socket 线程
     │                                │
     │ ←── ManualResetEventSlim ─────│
     │                                │
  coroutine                    TcpListener.Accept()
  等待命令                       阻塞读取命令
     │                                │
  执行物理步进                   写入 response
     │                                │
  发送响应 ────────────────────→     │
```

## 二进制协议

### Python → C#（命令）

| 字节 | 含义 |
|------|------|
| 0    | 命令类型：`R`=Reset, `S`=Step, `X`=Close |
| 1-N  | `S` 命令时：`numAgents * actionDim * 4` 字节 float32 actions |

### C# → Python（响应）

| 字节 | 含义 |
|------|------|
| 0-3  | `numAgents`（int32，大端） |
| 4-N  | `numAgents * stateDim * 4` 字节 float32 states（每个 agent 拼接） |

## 配置

端口通过 `RuntimeConfig.tcpPort`（默认 9000）配置，仅监听 `127.0.0.1`。

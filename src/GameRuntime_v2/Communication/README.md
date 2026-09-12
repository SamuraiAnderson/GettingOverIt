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

**小端二进制**、请求-响应式。完整命令语义与状态维度定义见 [`doc/entrypoints.md`](../../../doc/entrypoints.md) 第三节，此处为速查。

### Python → C#（命令）

首字节为命令类型，其余为该命令的负载：

| 命令 | 字节格式 | 含义 |
|------|----------|------|
| `R` (RESET) | `['R']` | 重置所有 agent 到初始快照 |
| `S` (STEP) | `['S'][n:1B][actions: n×2×4B]` | 注入动作并推进物理 `stepFrames` 帧 |
| `N` (NEW_SNAPSHOT) | `['N']` | 以当前状态为新基准重拍快照 |
| `C` (CONFIG) | `['C'][active:1B][mouseXId:4B][mouseYId:4B]` | 配置 Rewired 鼠标拦截 |
| `V` (VISUALIZE) | `['V'][enabled:1B]` | 开/关碰撞箱描边可视化 |
| `E` (EXPORT_COLLIDERS) | `['E']` | 导出碰撞体几何到文件 |
| `T` (TELEPORT) | `['T'][agentIndex:1B][x:4B][y:4B]` | 传送指定 agent 到世界坐标 |
| `F` (CAMERA_FREE) | `['F'][enabled:1B]` | 启用/禁用自由相机 |
| `X` (CLOSE) | `['X']` | 关闭训练循环并断开 |

`actions` 布局为 `[a0_x, a0_y, a1_x, a1_y, ...]`，float32 小端。

### C# → Python（响应）

```
[n: 1B]  然后对每个 agent 交错发送:  [state_i: stateDim×4B][done_i: 1B]
```

- `n` 为 agent 数（单字节）。
- state 与 done **逐 agent 交错**（一个 agent 的 state 紧跟其 done），Python 端必须逐 agent 读取，不能先读所有 state 再读所有 done。
- `stateDim = 33`（基础 29 维 + fakeCursor 4 维），是唯一权威维度（`StepController.STATE_DIM`）。
- 每个 float 显式按小端写出（`BitConverter.GetBytes`，非小端平台 `Array.Reverse`）。
- `CONFIG` / `VISUALIZE` / `EXPORT_COLLIDERS` / `CAMERA_FREE` 回空包（`n=0`）。
- `done` 当前恒为 false（C# 侧不判定终止，落水/终止由 Python 侧 `reward.is_water` 处理）。

## 配置

端口通过 `RuntimeConfig.tcpPort`（默认 9000）配置，仅监听 `127.0.0.1`。

# GameRuntime_v2 快速开始

## 一、数据采集模式（首次使用）

### 目的
采集 Mountain 碰撞箱数据，保存到文件供 AI 训练使用。

### 步骤

1. **设置模式**（Python）

```bash
cd c:\Users\Symbol\aCodes\GettingOverIt
python -c "from src.start import GameModeController; GameModeController().set_data_collection_mode()"
```

2. **启动游戏**
   - 运行 Getting Over It
   - 插件会自动检测信号文件，进入数据采集模式
   - 等待场景加载完成

3. **自动采集**
   - 插件会自动查找 "Mountain" GameObject
   - 采集所有 PolygonCollider2D 组件
   - 导出数据到 `GoiData/Colliders/Mountain_YYYYMMDD_HHMMSS.txt`

4. **完成**
   - 游戏会在 2 秒后自动退出
   - 检查 `GoiData/Colliders/` 目录确认文件生成

---

## 二、游戏运行模式（AI 训练）

### 目的
与 Python 训练脚本进行 TCP 帧级步进通信，进行 AI 训练。

### 步骤

1. **设置模式**（Python）

```bash
cd c:\Users\Symbol\aCodes\GettingOverIt
python -c "from src.start import GameModeController; GameModeController().set_game_runtime_mode()"
```

2. **启动游戏**
   - 运行 Getting Over It
   - 插件进入游戏运行模式，启动 `TcpStepServer`

3. **启动训练脚本**

```bash
python -m src.training.main_ppo --run-dir my_experiment --persist-game
```

> 训练侧的 `RolloutWorker` 会自动完成「写模式信号 → 启动游戏 → 连接 TCP」，通常无需手动执行第 1、2 步；上面拆分只为说明底层流程。

4. **训练过程**
   - Python（TCP 客户端）发 `STEP`，注入 2D 鼠标动作
   - Unity（TCP 服务端 `:9000`）帧级推进物理后回传 33 维状态
   - Python 执行 PPO 训练

---

## 三、游戏测试模式（调试）

### 目的
测试 Player 控制、碰撞箱采集等功能。

### 步骤

1. **设置模式**（Python）

```bash
cd c:\Users\Symbol\aCodes\GettingOverIt
python -c "from src.start import GameModeController; GameModeController().set_game_testing_mode()"
```

2. **启动游戏**
   - 运行 Getting Over It
   - 插件进入测试模式（`PlayerDebugTool` 键盘调试 + 物理探测）

3. **手动测试**
   - 使用热键或调试工具测试功能
   - 查看日志输出

---

## 四、调试热键

在游戏运行时：

- **F1**: 打印当前模式信息
- **F2**: 手动采集 Mountain 碰撞箱（仅数据采集模式）

---

## 五、配置文件

默认配置文件：`GoiData/runtime_config.json`

```json
{
  "mode": "DataCollection",
  "tcpPort": 9000,
  "actionDimension": 2,
  "duplicateCount": 1
}
```

> 状态维度不在配置里持久化，唯一权威为 `StepController.STATE_DIM=33`（避免旧配置残留造成协议错位）。端口须与 `src/config/project.json` 的 `tcp_port` 一致。

**优先级**: 信号文件 > 配置文件

---

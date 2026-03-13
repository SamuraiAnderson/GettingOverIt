# GameRuntime_v2 快速开始

## 一、数据采集模式（首次使用）

### 目的
采集 Mountain 碰撞箱数据，保存到文件供 AI 训练使用。

### 步骤

1. **设置模式**（Python）

```bash
cd c:\Users\Symbol\aCodes\goi-rl
python -c "from game_mode_controller import GameModeController; GameModeController().set_data_collection_mode()"
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
与 Python 训练脚本进行 UDP 通信，进行 AI 训练。

### 步骤

1. **设置模式**（Python）

```bash
cd c:\Users\Symbol\aCodes\goi-rl
python -c "from game_mode_controller import GameModeController; GameModeController().set_game_runtime_mode()"
```

2. **启动游戏**
   - 运行 Getting Over It
   - 插件进入游戏运行模式（当前版本提示"待实现"）

3. **启动训练脚本**

```bash
python main_train.py
```

4. **训练过程**
   - Unity 接收动作（UDP 12345）
   - Unity 发送状态（UDP 12346）
   - Python 执行 SAC 训练

**注意**: 游戏运行模式服务尚未实现，当前仅框架。

---

## 三、游戏测试模式（调试）

### 目的
测试 Player 控制、碰撞箱采集等功能。

### 步骤

1. **设置模式**（Python）

```bash
cd c:\Users\Symbol\aCodes\goi-rl
python -c "from game_mode_controller import GameModeController; GameModeController().set_game_testing_mode()"
```

2. **启动游戏**
   - 运行 Getting Over It
   - 插件进入测试模式（当前版本提示"待实现"）

3. **手动测试**
   - 使用热键或调试工具测试功能
   - 查看日志输出

**注意**: 游戏测试模式服务尚未实现，当前仅框架。

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
  "receiveHost": "localhost",
  "receivePort": 12345,
  "sendHost": "localhost",
  "sendPort": 12346,
  "stateDimension": 39,
  "actionDimension": 2,
  "updateFrequency": 40,
  "duplicateCount": 1
}
```

**优先级**: 信号文件 > 配置文件

---

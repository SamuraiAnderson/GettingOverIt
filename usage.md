# Getting Over It — 编译与运行指南

## 前置条件

- **Getting Over It** 游戏已安装（Steam 版）
- 游戏目录中已安装 **BepInEx**（`BepInEx\core` 下包含 `BepInEx.dll`）
- **Visual Studio 2019 Build Tools**（或完整版 VS），需包含 MSBuild
- **Python 3.9+**

## 1. 配置

### 游戏路径

两处需要保持一致：

1. **C# 项目** — `src/GameRuntime_v2/GameRuntime_v2.csproj` 中的 `<GameRoot>` 属性
2. **Python 配置** — `src/config/project.json` 中的 `executable_path`

```json
{
  "game": {
    "executable_path": "C:\\...\\Getting Over It\\GettingOverIt.exe",
    "max_execution_time": 300
  },
  "tcp_port": 9000
}
```

### Python 依赖

```powershell
pip install -r requirements.txt
```

包含：`torch>=2.0.0`、`numpy`、`matplotlib`、`tensorboard`。

数据分析环境（可选）：

```powershell
cd src\Environment
.\setup_conda_env.bat
```

## 2. 编译 C# 插件

```powershell
$msbuild = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"

& $msbuild src\GameRuntime_v2\GameRuntime_v2.csproj /p:Configuration=Release
```

编译产物 `GameRuntime_v2.dll` 输出到 `bin\Release\`，PostBuild 会自动将 DLL 复制到游戏的 `BepInEx\plugins` 目录。

## 3. 数据采集

采集游戏场景碰撞体数据，从 `src` 目录运行：

```powershell
cd src
python main.py
```

流程：设置数据采集模式 → 启动游戏 → 自动导出碰撞体 → 游戏退出 → 输出数据文件路径。

也可通过命令行工具手动控制：

```powershell
cd src

# 查看当前模式
python start/game_mode_controller.py status

# 设置模式
python start/game_mode_controller.py data       # 数据采集
python start/game_mode_controller.py runtime    # 游戏运行（RL 训练）
python start/game_mode_controller.py test       # 游戏测试

# 启动游戏
python start/game_launcher.py --wait
```

## 4. 训练

训练脚本会自动完成游戏启动、TCP 连接、数据收集、游戏关闭和模型更新，从**仓库根目录**运行。

### 迭代行为克隆（BC）

```powershell
python -m src.training.main_train
python -m src.training.main_train --num-agents 5 --max-iterations 50
python -m src.training.main_train --resume              # 跳过冷启动，从已有模型继续
```

流程：冷启动随机采集 → 构建效率图 → 监督训练 → 用模型采集新轨迹 → 迭代优化。

产出：`checkpoints/model_iter_XXXX.pt`

### PPO 强化学习

```powershell
python -m src.training.main_ppo
python -m src.training.main_ppo --num-agents 5 --max-iterations 200
python -m src.training.main_ppo --steps-per-rollout 300
python -m src.training.main_ppo --resume checkpoints/ppo_iter_0050.pt
```

流程：策略网络采样 → 计算奖励 → GAE 优势估计 → PPO 更新。

产出：`checkpoints/ppo_iter_XXXX.pt`

## 5. 架构概览

```
编译 C# 插件 (MSBuild)
       │
       ▼
DLL → BepInEx\plugins
       │
       ▼
启动游戏 (手动 / Python 脚本自动)
       │
       ▼
BepInEx 加载插件 → C# TCP 服务端监听 :9000
       │
       ▼
Python TCP 客户端连接 → 帧级步进通信 (GoiEnv)
```

- **C# 端**（`src/GameRuntime_v2/`）：BepInEx 插件，运行在 Unity 进程中，负责状态读取、输入注入、TCP 服务端
- **Python 端**（`src/env/`、`src/training/`）：RL 环境客户端 + 训练框架
- **通信**：每个 RL step 对应一次 TCP 请求-响应，保证帧级同步（默认端口 9000）

# 操作手册 — 从零到第一次训练

本文是"怎么把项目跑起来"的唯一权威：前置条件 → 配置 → 编译插件 → 采集地图 → 训练 → 排错。

- 编译器/反编译器的细节与本机坑：见 [`doc/toolchain.md`](doc/toolchain.md)。
- 交互架构与协议、训练算法细节：见 [`doc/entrypoints.md`](doc/entrypoints.md)、[`doc/training.md`](doc/training.md)。
- PPO 完整 CLI 与更多配方：见 [`doc/main_ppo_usage.md`](doc/main_ppo_usage.md)。

---

## 1. 前置条件

- **《Getting Over It》**（Steam 版）已安装。
- 游戏目录里已装 **BepInEx**（存在 `BepInEx\core\BepInEx.dll`）。
- **Visual Studio 2019 Build Tools**（含 MSBuild），用来编译 C# 插件。本机只有 .NET 运行时、没有 SDK，所以**不能用 `dotnet build`**（详见 [`doc/toolchain.md`](doc/toolchain.md)）。
- **Python 3.9+**。训练依赖列在 [`requirements.txt`](requirements.txt)，用下面第 1.2 步一次装好。

### 1.1 游戏路径（两处必须一致）

游戏安装路径要在两个文件里保持一致：

1. C# 项目 — [`src/GameRuntime_v2/GameRuntime_v2.csproj`](src/GameRuntime_v2/GameRuntime_v2.csproj) 的 `<GameRoot>` 属性。
2. Python 配置 — [`src/config/project.json`](src/config/project.json) 的 `executable_path`。

```json
{
  "game": {
    "executable_path": "C:\\...\\Getting Over It\\GettingOverIt.exe",
    "max_execution_time": 300
  },
  "tcp_port": 9000
}
```

> 换机器时这两处都要改；忘了同步是最常见的启动失败原因（见第 6 节排错）。TCP 端口只从这里读，不要在代码里硬编码。

### 1.2 Python 依赖

```powershell
pip install -r requirements.txt
```

数据分析用的 conda 环境（可选，部分分析/评估脚本用 `getting-over-it-analysis`）：

```powershell
cd src\Environment
.\setup_conda_env.bat
```

---

## 2. 编译 C# 插件

改完 `src/GameRuntime_v2/` 下任何代码后，都要重编一次插件。**编译前先关掉游戏**（否则 DLL 被占用，自动拷贝会失败）。

在仓库根目录（PowerShell）执行：

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe' 'src\GameRuntime_v2\GameRuntime_v2.csproj' /t:Build /p:Configuration=Release /nologo /v:minimal
```

产物 `GameRuntime_v2.dll` 输出到 `bin\Release\`，编译脚本会自动把它拷进游戏的 `BepInEx\plugins\`。**改动必须重启游戏才生效**（插件只在启动时加载）。

> 完整排错（"No .NET SDKs found"、拷贝失败、MSBuild 路径变更等）见 [`doc/toolchain.md`](doc/toolchain.md)。

---

## 3. 采集地图几何

首次使用需要先采集一次游戏场景的碰撞体数据。从 `src` 目录运行：

```powershell
cd src
python main.py
```

流程：自动设为数据采集模式 → 启动游戏 → 导出碰撞体 → 游戏退出 → 打印数据文件路径。

也可以手动控制模式与启动：

```powershell
cd src

python start/game_mode_controller.py status     # 查看当前模式
python start/game_mode_controller.py data       # 数据采集
python start/game_mode_controller.py runtime    # 游戏运行（RL 训练）
python start/game_mode_controller.py test       # 游戏测试

python start/game_launcher.py --wait            # 启动游戏
```

---

## 4. 训练

训练脚本从**仓库根目录**运行，会自动完成：启动游戏 → TCP 连接 → 采集 → 更新 → 关闭。

### 4.1 迭代行为克隆（BC）

```powershell
python -m src.training.main_train
python -m src.training.main_train --num-agents 10 --max-iterations 50
python -m src.training.main_train --resume         # 跳过冷启动，从已有模型继续
```

流程：冷启动随机采集 → 构建效率图 → 监督训练 → 用模型采集新轨迹 → 迭代优化。产出 `checkpoints/model_iter_XXXX.pt`。

### 4.2 PPO 强化学习

**当前推荐配方（最小任务）**：固定起点、短 episode、只考核能否翻过第一个障碍，先不要开随机投放。

```powershell
python -m src.training.main_ppo `
  --no-random-deploy `
  --steps-per-rollout 400 `
  --max-iterations 40 `
  --run-dir my_experiment `
  --persist-game
```

从已有 checkpoint 继续：

```powershell
python -m src.training.main_ppo --resume checkpoints/ppo_iter_0050.pt
```

产出 `checkpoints/ppo_iter_XXXX.pt`。每组实验的目录布局（哪些入库、哪些忽略）与归档规范见 [`runs/README.md`](runs/README.md)。

> 完整 CLI 参数、更多配方、以及哪些路径已废弃/慎用，见 [`doc/main_ppo_usage.md`](doc/main_ppo_usage.md)。

---

## 5. 验证

改完代码后建议按改动范围跑对应验证。最小验证（只动 Python 训练侧）：

```powershell
python -m src.tests.run_unit_tests
```

四类验证（离线单测 / 离线分析 / 在环 L1–L9 / 历史标定）与"改了哪层重跑哪层"见 [`doc/verification.md`](doc/verification.md)。

---

## 6. 常见问题排错

| 现象 | 可能原因与解决 |
|---|---|
| Python 连不上游戏 / 连接超时 | 插件没加载成功（游戏目录 `BepInEx\plugins\` 里没有 `GameRuntime_v2.dll`，或没重启游戏）；先按第 2 节重编并重启游戏 |
| 端口占用 / 连接被拒 | 上一个游戏进程没退干净；结束残留的 `GettingOverIt.exe` 再重跑 |
| 启动后行为异常 / 找不到游戏 | `.csproj` 的 `<GameRoot>` 和 `project.json` 的 `executable_path` 不一致，或端口两边不一致（见第 1.1 节） |
| 编译报 "No .NET SDKs were found" | 误用了 `dotnet build`，改用第 2 节的 MSBuild 命令 |
| 编译时 DLL 被占用、拷贝失败 | 游戏还开着，关掉游戏进程再重编 |
| 改了插件但行为没变 | 忘了重启游戏（插件只在启动时加载一次） |
| 多 agent 数量过多时物理交互异常 | 游戏本身不是为多实例设计的，agent 过多时部分交互会互相干扰，适当调小 `--num-agents` |

---

## 7. 架构概览

```
编译 C# 插件 (MSBuild)  →  DLL 拷进 BepInEx\plugins
        │
        ▼
启动游戏 (手动 / Python 脚本自动)  →  BepInEx 加载插件  →  C# TCP 服务端监听 :9000
        │
        ▼
Python TCP 客户端 (GoiEnv) 连接  →  每个 RL step 一次请求-响应，帧级同步
```

- **C# 端**（`src/GameRuntime_v2/`）：BepInEx 插件，运行在游戏进程内，负责读状态、注入输入、跑 TCP 服务端。
- **Python 端**（`src/env/`、`src/training/`）：RL 环境客户端 + 训练框架。
- **通信**：每个 RL step 对应一次 TCP 请求-响应，保证帧级同步（默认端口 9000）。

> 完整交互链路、三种运行模式、线路协议与状态维度见 [`doc/entrypoints.md`](doc/entrypoints.md)。

# Getting Over It — 把封闭商业游戏改造成可编程 RL 环境

把一款没有任何官方接口的 Steam 商业游戏，改造成具备**帧级确定性步进、快照重置、多 agent 并行、可观测内部物理状态**的强化学习环境，并在其上搭建训练、验证与实验记录体系。

**技术栈**：C# · BepInEx · Harmony · Unity · Python · PyTorch · TCP 二进制协议

> 本文是项目的**开发者入口**：讲清项目是什么、现在到哪了、代码在哪、怎么上手、去哪看细节。深度内容全部在 [`doc/`](doc/) 与各模块的 README 里，本文只做导航。

---

## 一、这是什么

《Getting Over It with Bennett Foddy》是一款闭源 Steam 商业游戏：玩家用鼠标操纵一把锤子把角色拖上山，没有检查点，摔落直接回到起点。把它作为 RL 环境有三重困难——控制量是连续的鼠标位移（无离散动作）、奖励极稀疏（只有"登顶"）、没有任何现成接口（无官方 API、无 Gym wrapper）。

现成方案（OpenAI Gym、Unity ML-Agents）在这里都用不了，因为它们要么需要环境主动实现接口、要么需要在工程构建期集成。**唯一入口是从 Unity 运行时下手**：注入插件、读取内部物理状态、劫持输入管线、接管物理时钟。本项目做的就是这件事。

## 二、核心价值

- **主体成果是环境与验证基础设施**：环境侧完整可用，并通过一组按能力递进的集成测试逐项验证——每一项检验的能力都建立在前一项成立之上（从"TCP 能连通"到"多 agent 隔离""空投落点""模型评估"，详见 [`doc/verification.md`](doc/verification.md)）。
- **训练侧是一次尚未完成的实验**：目前还没突破第一段攀爬，瓶颈已定位为硬探索死锁（诚实标注，不夸大）。
- 大量设计决策都有可复现的依据——包括从反编译游戏代码得到的接触判断常量，以及一批负面结果（见[已知结论](#四当前状态)）。

## 三、架构

两个进程，一条 TCP 通道：Python 作训练侧（客户端），游戏进程内的 BepInEx 插件作服务端。一次 RL step = 一次阻塞式 TCP 请求-响应，C# 端在回包前完成物理推进，保证时序严格对齐。

```
┌──── Python 进程（训练侧 / TCP 客户端） ────┐
│  启动游戏 · 写模式信号 · GoiEnv 步进        │
└──────────────┬─────────────────────────────┘
               │ TCP（小端二进制，请求-响应，默认 :9000）
               ▼
┌──── 游戏进程内 BepInEx 插件 GameRuntime_v2（服务端） ────┐
│  帧级物理推进 · 快照/重置/传送 · 输入注入 · 状态采集      │
└──────────────────────────────────────────────────────────┘
```

> 完整交互链路、三种运行模式、线路协议与状态维度见 [`doc/entrypoints.md`](doc/entrypoints.md)。

## 四、当前状态

环境侧完整可用；训练侧到目前为止的尝试（PPO iid 高斯 → OU 时序噪声 + MAP-Elites，累计约 260 轮）都卡在同一个位置——没能稳定翻越第一个障碍。**当前最佳解释**是**硬探索死锁**（50Hz 每帧独立采样，凑不出翻越台阶所需的连贯动作；塑形、模仿、评分都在探索的下游）。**这是对已有尝试为什么失败的诊断，不是对 PPO / RL 方案本身可行性的否定**——roadmap 里 P2.2 / P3 / P4 都还没试。

四条已知结论（含"快照确定性 1e-4 地板""动作是低频的"等，多为负面结果、用于排除假设与守住护栏）汇总见 [`doc/findings.md`](doc/findings.md)；下一步优化项与优先级见 [`doc/optimization_roadmap.md`](doc/optimization_roadmap.md)。

## 五、项目地图

按目录职责快速定位代码。每个模块的细节见其自带 README。

| 目录 | 职责 | 详细 |
|---|---|---|
| `src/GameRuntime_v2/` | C# BepInEx 插件（游戏进程内）总入口 | [README](src/GameRuntime_v2/README.md) |
| `src/GameRuntime_v2/Core/` | 插件入口、模式管理、帧级步进、事件总线、反射工具 | [README](src/GameRuntime_v2/Core/README.md) |
| `src/GameRuntime_v2/Communication/` | TCP 双线程步进服务端 | [README](src/GameRuntime_v2/Communication/README.md) |
| `src/GameRuntime_v2/PlayerControl/` | 反射注入 + Rewired 拦截、状态采集、复制体管理 | [README](src/GameRuntime_v2/PlayerControl/README.md) |
| `src/GameRuntime_v2/ColliderCollection/` | 碰撞几何导出与可视化 | [README](src/GameRuntime_v2/ColliderCollection/README.md) |
| `src/GameRuntime_v2/Camera/` | 自由相机控制（`FreeCameraController`） | — |
| `src/GameRuntime_v2/Testing/` | 调试工具（`PlayerDebugTool`、输入特性探测） | [README](src/GameRuntime_v2/Testing/README.md) |
| `src/env/goi_env.py` | Python TCP 客户端（唯一环境封装） | [entrypoints](doc/entrypoints.md) |
| `src/training/` | 训练框架：模型、PPO、buffer、rollout、奖励、数据集 | [training](doc/training.md) |
| `src/start/` | 进程启动与模式信号 | [README](src/start/README.md) |
| `src/config/project.json` | 集中配置：游戏路径、TCP 端口 | — |
| `src/tests/` | 四类验证脚本 | [README](src/tests/README.md) |
| `src/Data/` | 采集数据、参考轨迹、分析图表 | — |

## 六、开发约定

目录职责划分、文件移动时同步 `.csproj`、路径用 `Path.Combine`、TCP 端口从 `project.json` 读取（禁止硬编码）等工程规范，统一以 [`.cursor/rules/project-standards.mdc`](.cursor/rules/project-standards.mdc) 为准，本文不重复。

实验记录规范见下面第十节。

## 七、快速开始

三条最简命令（前置条件、配置、排错等完整步骤见 [`usage.md`](usage.md)）：

```powershell
# 1. 编译 C# 插件（详见 usage.md 第 2 节）
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe' 'src\GameRuntime_v2\GameRuntime_v2.csproj' /p:Configuration=Release

# 2. 采集地图几何
cd src; python main.py

# 3. 训练（PPO 最小任务）
python -m src.training.main_ppo --no-random-deploy --steps-per-rollout 400 --run-dir my_experiment --persist-game
```

## 八、外部工具

| 工具 | 用途 | 何时需要 | 详细 |
|---|---|---|---|
| VS2019 Build Tools / MSBuild | 编译 C# 插件 | 跑训练必需 | [`doc/toolchain.md`](doc/toolchain.md) |
| BepInEx · Harmony | 插件注入与运行时打补丁 | 跑训练必需 | [`doc/toolchain.md`](doc/toolchain.md) |
| dnSpy / dnlib | 反编译游戏程序集查内部逻辑 | 仅理解游戏逻辑时需要 | [`doc/toolchain.md`](doc/toolchain.md) |

> 训练/运行的 Python 依赖不在此列——直接 `pip install -r requirements.txt`（见 [`requirements.txt`](requirements.txt)）。

## 九、文档导航

| 想了解 | 去看 |
|---|---|
| 怎么把项目跑起来 | [`usage.md`](usage.md) |
| 交互架构、运行模式、协议、状态维度 | [`doc/entrypoints.md`](doc/entrypoints.md) |
| 训练框架、模型结构、奖励/评分、数据流 | [`doc/training.md`](doc/training.md) |
| PPO 完整 CLI 与推荐配方 | [`doc/main_ppo_usage.md`](doc/main_ppo_usage.md) |
| 优化项与优先级、设计原则 | [`doc/optimization_roadmap.md`](doc/optimization_roadmap.md) |
| 已知结论（含负面结果） | [`doc/findings.md`](doc/findings.md) |
| 验证体系与"改了哪层重跑哪层" | [`doc/verification.md`](doc/verification.md) |
| 编译器 / 反编译器工具链 | [`doc/toolchain.md`](doc/toolchain.md) |
| 锤子物理的数学模型 | [`doc/hammer_physics.md`](doc/hammer_physics.md) |
| 快照确定性前提验证 | [`doc/planb_premise_verification.md`](doc/planb_premise_verification.md) |
| 各组实验记录 | [`runs/README.md`](runs/README.md) 及各 `runs/<tag>/RESULT.md` |

## 十、目录结构与实验记录

```
src/          # C# 插件 + Python 环境/训练/测试/数据（模块职责见第五节）
doc/          # 深度参考：entrypoints / training / optimization_roadmap /
              #   findings / verification / toolchain / hammer_physics / planb
runs/<tag>/   # 每组实验：run_meta.json + metrics.csv + RESULT.md（轻量入库）
checkpoints/  # 模型、地图几何、候选投放点、人工排除区
```

**实验记录规范**：每组训练实验在 `runs/<tag>/` 下留一份 `RESULT.md`，固定记录**意图 / 命令 / 客观结果 / 结论**。目的是让负面结果也能复用——多数实验的价值在于排除一个假设。用 `--run-dir <tag>` 启动即自动建目录、写元数据、逐轮记录指标。详见 [`runs/README.md`](runs/README.md)。

## 十一、当前局限

- **训练未达成目标**：PPO 跑到 260 轮仍未稳定翻越第一个障碍，根因见 [`doc/findings.md`](doc/findings.md)。
- **游戏路径硬编码两处**（`.csproj` 的 `<GameRoot>` 与 `project.json` 的 `executable_path`），换机需手动同步。
- **游戏进程、训练脚本、TensorBoard 需分别启动**，无统一编排。
- **多 agent 并发依赖游戏内同时存在 N 个角色实例**，游戏本身未设计为多实例场景，agent 数量过多时部分物理交互会互相干扰。
- **`src/GameRuntime_v2/FILE_OVERVIEW.md` 已过时**，以 [`doc/entrypoints.md`](doc/entrypoints.md) 为准。

## 十二、说明

本项目为个人研究性质，用于探索"在没有现成环境接口的商业软件上构建 RL 环境"的工程路径。不包含任何游戏本体资源；运行需自行拥有游戏正版及 BepInEx。

# 环境交互性测试模块

> **目录位置**: `src/tests/environment_interaction/`  
> **上级文档**: [测试系统总览](../README.md)

## 📋 模块简介

本模块专注于测试游戏环境的交互性和可控性，包括：
- ✅ 完全重置功能
- ✅ 状态保存/加载（Checkpoint）
- ✅ 状态一致性验证

这些功能对强化学习训练至关重要，确保环境可重置、可复现、确定性强。

**测试状态**: 🚧 开发中  
**文档版本**: v1.0  
**创建日期**: 2025-10-20

---

## 🎯 测试目标

| 测试类别 | 核心问题 | RL重要性 |
|---------|---------|---------|
| **完全重置** | 能否快速重置到游戏初始状态？ | ⭐⭐⭐ 必需 |
| **状态保存/加载** | 能否保存并恢复到任意中间状态？ | ⭐⭐⭐ 课程学习 |
| **状态一致性** | 多次重置到同一状态是否完全一致？ | ⭐⭐⭐ 可复现性 |

---

## 📁 目录结构

```
environment_interaction/
├── README.md                      # 本文件
├── test_config.json               # 测试配置
├── requirements.txt               # Python依赖
│
├── unity_components/              # Unity端组件（C#）
│   ├── EnvironmentStateManager.cs        # 状态管理核心
│   └── EnvironmentTestController.cs      # 测试控制器
│
├── test_scripts/                  # Python测试脚本
│   ├── test_full_reset.py         # 完全重置测试
│   ├── test_state_checkpoint.py   # 状态保存/加载测试
│   └── test_state_consistency.py  # 状态一致性验证
│
└── analysis/                      # 分析工具（待添加）
    ├── visualize_consistency.py
    └── report_generator.py
```

---

## 🚀 快速开始

### 1. 环境准备

#### Unity端集成

将Unity组件集成到游戏插件中：

```csharp
// 在 GoiHitboxLogger.cs 中添加
using GoiHitboxLogger;

[BepInPlugin("com.symbol.goi.hitbox_logger", "GOI Hitbox Logger", "1.0.0")]
public class GoiHitboxLogger : BaseUnityPlugin
{
    private GameObject controllerObj;
    
    void Awake()
    {
        // 创建测试控制器
        controllerObj = new GameObject("EnvironmentTestController");
        controllerObj.AddComponent<EnvironmentTestController>();
        DontDestroyOnLoad(controllerObj);
    }
}
```

#### Python环境

```bash
# 进入模块目录
cd src/tests/environment_interaction

# 安装依赖
pip install -r requirements.txt

# 配置游戏路径（编辑 test_config.json）
```

### 2. 配置游戏路径

编辑 `test_config.json`：

```json
{
  "game_exe_path": "<你的游戏安装路径>/Getting Over It.exe",
  "game_data_dir": "<你的游戏安装路径>/GoiData"
}
```

### 3. 运行测试

#### 测试1：完全重置测试
```bash
cd test_scripts
python test_full_reset.py
```

**测试内容**：
- 保存初始状态
- 执行随机动作
- 重置到初始状态
- 验证重置后状态与初始状态的一致性
- 重复10次，计算成功率和重置耗时

**预期输出**：
```
✅ 测试通过！所有重置都成功且状态一致
重置耗时统计:
  平均: 0.123秒
状态差异统计:
  平均: 1.23e-06
```

#### 测试2：状态保存/加载测试
```bash
python test_state_checkpoint.py
```

**测试内容**：
- 在不同时间点保存checkpoint
- 验证checkpoint的保存和加载功能
- 测试从checkpoint开始的可复现性

**预期输出**：
```
✅ 测试通过！所有checkpoint保存/加载都成功
保存/加载成功率: 100.0%
```

#### 测试3：状态一致性验证
```bash
python test_state_consistency.py
```

**测试内容**：
- 从同一初始状态开始
- 执行相同动作序列多次
- 比较最终状态的一致性
- 分析各个刚体的一致性

**预期输出**：
```
✅ 基本一致性测试通过
   最大差异: 8.45e-07
各刚体一致性分析:
   PlayerBody:
     位置标准差: [1.2e-07, 3.4e-07]
     旋转标准差: 2.1e-08
```

---

## 🎮 Unity端功能

### EnvironmentStateManager

**核心类**：状态管理器

**主要功能**：

```csharp
// 1. 捕获当前状态
GameState state = EnvironmentStateManager.CaptureCurrentState("checkpoint_name");

// 2. 保存初始状态
EnvironmentStateManager.SaveInitialState();

// 3. 重置到初始状态
EnvironmentStateManager.ResetToInitial();

// 4. 保存/加载 Checkpoint
EnvironmentStateManager.SaveCheckpoint("cp1");
EnvironmentStateManager.LoadCheckpoint("cp1");

// 5. 比较状态
float diff = EnvironmentStateManager.CompareStates(state1, state2);

// 6. 文件IO
EnvironmentStateManager.SaveStateToFile(state, "path/to/file.json");
GameState loaded = EnvironmentStateManager.LoadStateFromFile("path/to/file.json");
```

**状态数据结构**：

```csharp
public class GameState
{
    public float timestamp;
    public int frameCount;
    public List<RigidbodyState> rigidbodyStates;  // 所有刚体的状态
    public float gameProgress;  // 游戏进度（高度）
    public string checkpointName;
}

public class RigidbodyState
{
    public string name;
    public Vector2 position;
    public float rotation;
    public Vector2 velocity;
    public float angularVelocity;
    // ...
}
```

### EnvironmentTestController

**核心类**：测试控制器

**热键**：
- `F5`: 保存初始状态
- `F6`: 重置到初始状态
- `F7`: 保存Checkpoint
- `F8`: 加载最新Checkpoint
- `F9`: 导出当前状态

**文件信号接口**：

| 信号文件 | 功能 | 内容 |
|---------|------|------|
| `save_initial.signal` | 保存初始状态 | 空 |
| `reset.signal` | 重置到初始状态 | 空 |
| `save_checkpoint.signal` | 保存checkpoint | checkpoint名称 |
| `load_checkpoint.signal` | 加载checkpoint | checkpoint名称 |
| `export_state.signal` | 导出当前状态 | 文件名 |
| `compare_state.signal` | 比较状态 | 状态文件名 |

**结果文件**：

操作完成后会在 `GoiData/TestResults/` 生成结果文件：
- `{operation}_success.result` - 成功
- `{operation}_failed.result` - 失败

---

## 📊 测试数据输出

所有测试数据输出到：`GoiData/`

```
GoiData/
├── TestSignals/         # 信号文件目录
│   ├── save_initial.signal
│   ├── reset.signal
│   └── ...
│
├── GameStates/          # 状态文件目录
│   ├── initial_state.json
│   ├── checkpoint_*.json
│   └── state_*.json
│
└── TestResults/         # 测试结果目录
    ├── full_reset_test_report.json
    ├── checkpoint_test_report.json
    ├── consistency_test_report.json
    └── *.result
```

---

## 📈 性能指标

### 成功标准

| 测试项 | 目标 | 可接受 | 需优化 |
|--------|------|--------|--------|
| 完全重置成功率 | 100% | ≥95% | <95% |
| 重置耗时 | <0.5s | <2s | >2s |
| 状态一致性 | <1e-5误差 | <1e-3 | >1e-3 |
| Checkpoint保真度 | 100% | ≥99.9% | <99.9% |

### 当前状态

⚠️ **注意**：本模块仍在开发中，性能指标待实际测试验证。

---

## 🔧 高级用法

### 自定义测试场景

```python
from test_scripts.test_full_reset import FullResetTester

class CustomTester(FullResetTester):
    def simulate_random_actions(self, duration):
        # 自定义动作序列
        # 例如：注入特定的鼠标输入
        pass

tester = CustomTester()
tester.run_test()
```

### 分析状态差异

```python
import json

# 加载状态文件
with open('GoiData/GameStates/state1.json', 'r') as f:
    state1 = json.load(f)

with open('GoiData/GameStates/state2.json', 'r') as f:
    state2 = json.load(f)

# 比较特定刚体
for rb1, rb2 in zip(state1['rigidbodyStates'], state2['rigidbodyStates']):
    pos_diff = ((rb1['position']['x'] - rb2['position']['x'])**2 + 
                (rb1['position']['y'] - rb2['position']['y'])**2)**0.5
    print(f"{rb1['name']}: 位置差异 {pos_diff:.6e}")
```

---

## 🐛 故障排查

### 问题1：游戏无法启动

**解决方案**：
1. 检查 `test_config.json` 中的 `game_exe_path` 是否正确
2. 确保游戏没有在运行
3. 检查是否有权限问题

### 问题2：信号文件不生效

**解决方案**：
1. 确认Unity组件已正确集成
2. 检查 `GoiData/TestSignals/` 目录是否存在
3. 查看Unity日志（BepInEx console）

### 问题3：状态差异过大

**原因**：
- 物理引擎的非确定性
- 浮点数精度问题
- 时间步长不稳定

**解决方案**：
1. 确保开启VSync（Unity插件会自动处理）
2. 调整 `consistency_threshold` 阈值
3. 检查是否有外部干扰（如其他程序占用CPU）

---

## 📚 相关文档

- [测试系统总览](../README.md)
- [输入特性测试](../input_characteristics/README.md)
- [GameRuntime文档](../../GameRuntime/readme.md)

---

## 🔄 更新日志

### v1.0 (2025-10-20) - 初始版本
- ✅ 创建模块结构
- ✅ 实现Unity端状态管理器
- ✅ 实现测试控制器
- ✅ 创建完全重置测试
- ✅ 创建状态保存/加载测试
- ✅ 创建状态一致性验证测试
- ✅ 创建基础文档

---

**创建日期**: 2025-10-20  
**最后更新**: 2025-10-20  
**维护者**: Getting Over It RL Project Team


# 测试系统总览

本目录包含Getting Over It项目的所有测试模块。

## 📁 目录结构

```
tests/
├── README.md                     # 本文件（总索引）
└── input_characteristics/        # 输入特性标定测试
    ├── README.md                 # 详细文档索引
    ├── [测试脚本 × 10]
    └── [文档 × 4]
```

---

## 🧪 测试模块

### 1. 输入特性标定 (Input Characteristics) ✅

**目录**: `input_characteristics/`

**目的**: 标定游戏输入的精度、有效范围、饱和点等特性，为强化学习提供准确的动作空间配置。

**核心发现**:
- 输入精度: 1.0
- 最小有效值: 10
- 最大有效值: 100
- 有效范围: [10, 100]
- 饱和点: 100 (硬限制)

**状态**: 已完成 (27个测试)

**快速开始**:
```bash
cd input_characteristics
pip install -r requirements_test.txt
python dose_response_test.py
```

**详细文档**: 见 [`input_characteristics/README.md`](input_characteristics/README.md)

---

### 2. 环境交互性测试 (Environment Interaction) 🚧

**目录**: `environment_interaction/`

**目的**: 测试游戏环境的交互性和可控性，包括完全重置、状态保存/加载、状态一致性验证。

**测试内容**:
- 完全重置功能 (10个测试)
- 状态保存/加载 (Checkpoint功能)
- 状态一致性验证 (确定性测试)

**状态**: 开发中

**快速开始**:
```bash
cd environment_interaction
pip install -r requirements.txt
python test_scripts/test_full_reset.py
```

**详细文档**: 见 [`environment_interaction/README.md`](environment_interaction/README.md)

---

## 📊 测试数据输出

所有测试数据输出到游戏目录：
```
<游戏安装目录>/GoiData/
├── GameResults/              # 原始CSV数据
│   ├── result_*.csv          # 粗测数据
│   ├── precision_*.csv       # 精测数据
│   ├── decimal_*.csv         # 小数测试
│   └── maxtest_*.csv         # 最大值测试
└── *.json                    # 分析报告
```

---

## 🔧 通用配置

### test_config.json 模板

```json
{
  "game_exe_path": "<游戏安装目录>/Getting Over It.exe",
  "game_data_dir": "<游戏安装目录>/GoiData",
  "collection_output_dir": "GameResults"
}
```

### Python依赖

```bash
pip install pandas numpy matplotlib psutil
```

---

## 🚀 后续测试计划

### 3. 状态空间测试 (待添加)
- [ ] 状态值范围标定
- [ ] 状态归一化测试
- [ ] 状态变化频率分析

### 4. 奖励函数测试 (待添加)
- [ ] 不同奖励设计对比
- [ ] 奖励尺度验证
- [ ] 稀疏vs密集奖励

### 5. 性能基准测试 (待添加)
- [ ] 随机策略baseline
- [ ] 人类玩家benchmark
- [ ] 简单启发式策略

### 6. 集成测试 (待添加)
- [ ] 端到端测试
- [ ] 长期运行稳定性
- [ ] 内存泄漏检测

---

## 📚 相关文档

- [项目主README](../../readme.md)
- [GameRuntime文档](../GameRuntime/readme.md)
- [Documentation目录](../Documentation/)

---

**创建日期**: 2025-10-20  
**最后更新**: 2025-10-20  
**维护者**: Getting Over It RL Project Team


# 输入特性标定测试

> **目录位置**: `src/tests/input_characteristics/`  
> **上级文档**: [测试系统总览](../README.md)

## 📋 模块简介

本模块专注于游戏输入特性的系统化标定，包括：
- ✅ 输入精度测试
- ✅ 最小/最大有效值标定
- ✅ 饱和点验证
- ✅ 量化方式分析

**测试完成度**: 100% (27个测试用例)  
**文档版本**: v1.2  
**最后更新**: 2025-10-20

---

## 📚 文档导航

### 核心文档（必读）

1. **[游戏输入精度测试方法论.md](游戏输入精度测试方法论.md)** ⭐⭐⭐
   - 系统化的测试方法论
   - 适用于任何游戏
   - 三阶段测试方案
   - 完整代码模板
   - **推荐首先阅读**

2. **[输入特性验证总结.md](输入特性验证总结.md)** ⭐⭐
   - Getting Over It实战案例
   - 18个测试的完整分析
   - 最终配置推荐
   - 强化学习应用指南

3. **[输入特性快速参考.md](输入特性快速参考.md)** ⭐
   - 一页纸快速参考
   - 核心结论
   - 推荐配置代码

---

## 🔬 测试脚本（8个核心文件）

### 阶段1：粗测（确定有效范围）
- `dose_response_test.py` - 粗测主脚本（生成+分析）
- `run_dose_response_tests.py` - 自动运行粗测

### 阶段2：整数精测（确定整数精度）
- `precision_test_50.py` - 整数精测（49-52）
- `run_precision_test_50.py` - 自动运行精测
- `analyze_precision_50.py` - 精密分析

### 阶段3：小数精测（确定输入域）
- `precision_test_decimal.py` - 小数精测（50.0-51.0）
- `analyze_decimal_precision.py` - 小数分析

### 阶段4：最大值测试（标定饱和点）
- `test_max_input.py` - 最大值测试（80-500）
- `test_max_refine.py` - 精细化测试（80-100）

### 通用分析工具
- `analyze_first_response.py` - 多阶段通用分析

### 配置和工具
- `quick_dose_test.bat` - Windows快捷启动
- `test_config.json` - 配置文件
- `requirements_test.txt` - Python依赖

---

## 📊 测试数据

### 粗测数据（9个）
```
GoiData/GameResults/
├── result_005.csv
├── result_010.csv
├── result_015.csv
├── result_020.csv
├── result_025.csv
├── result_030.csv
├── result_050.csv
├── result_080.csv
└── result_100.csv
```

### 整数精测数据（4个）
```
├── precision_49.csv
├── precision_50.csv
├── precision_51.csv
└── precision_52.csv
```

### 小数精测数据（5个）
```
├── decimal_500.csv  (50.0)
├── decimal_501.csv  (50.1)
├── decimal_502.csv  (50.2)
├── decimal_505.csv  (50.5)
└── decimal_510.csv  (51.0)
```

### 最大值测试数据（9个）
```
├── maxtest_0080.csv
├── maxtest_0085.csv
├── maxtest_0090.csv
├── maxtest_0095.csv
├── maxtest_0100.csv
├── maxtest_0150.csv
├── maxtest_0200.csv
├── maxtest_0300.csv
└── maxtest_0500.csv
```

### 分析报告
```
├── dose_response_analysis.json
├── precision_test_report.json
├── decimal_precision_report.json
└── max_input_report.json
```

---

## 🚀 快速开始

### 新用户（3步骤）

**1. 快速了解（2分钟）**
```bash
阅读：输入特性快速参考.md
```

**2. 学习方法（10分钟）**
```bash
阅读：游戏输入精度测试方法论.md
```

**3. 查看案例（5分钟）**
```bash
阅读：输入特性验证总结.md
```

### 运行测试（10分钟）

**Windows用户（推荐）**
```bash
# 双击运行
quick_dose_test.bat

# 选择：
# 1 - 生成测试文件
# 2 - 运行所有测试（自动）
# 3 - 分析结果
```

**命令行用户**
```bash
cd src/tests

# 阶段1：粗测（~5分钟）
python dose_response_test.py              # 生成测试文件
python run_dose_response_tests.py         # 自动运行
python dose_response_test.py --analyze    # 分析结果

# 阶段2：整数精测（~2分钟）
python run_precision_test_50.py           # 自动运行
python analyze_precision_50.py            # 分析结果

# 阶段3：小数精测（~2.5分钟）
python precision_test_decimal.py          # 自动运行
python analyze_decimal_precision.py       # 分析结果

# 阶段4：最大值测试（~2.5分钟）
python test_max_input.py                  # 自动运行+分析
python test_max_refine.py                 # 精细化（可选）
```

---

## 📈 测试流程

```
开始
  ↓
阅读方法论
  ↓
运行粗测（5-10分钟）
  ├─ dose_response_test.py
  └─ 确定有效范围
  ↓
运行精测（2-3分钟）
  ├─ precision_test_50.py
  └─ 确定精度
  ↓
可选：小数测试（2-3分钟）
  ├─ precision_test_decimal.py
  └─ 确定输入域类型
  ↓
最大值测试（2-3分钟）
  ├─ test_max_input.py
  └─ 标定饱和点
  ↓
分析和报告
  ├─ analyze_*.py
  └─ 生成JSON报告
  ↓
应用到强化学习
  └─ 配置动作空间
```

---

## 🎮 最终结论（Getting Over It）

| 特性 | 值 |
|------|-----|
| **输入域** | 整数 |
| **精度** | 1.0 |
| **最小有效值** | 10 |
| **最大有效值** | 100 |
| **有效范围** | [10, 100] |
| **饱和点** | 100（硬限制） |
| **量化方式** | 截断 |

### 推荐动作空间

```python
# 连续空间（推荐）
action_space = Box(low=-100.0, high=100.0, dtype=np.float32)
preprocess = lambda x: np.floor(x)  # 或 np.round

# 或离散空间
action_space = Discrete(201)  # -100到+100
```

---

## 🔧 依赖和环境

### Python依赖
```bash
pip install pandas numpy matplotlib psutil
```

### 游戏设置
- Unity插件已部署：`GoiHitboxLogger.dll`
- 自动采集模式：3秒采集+自动关闭
- 数据输出：`GoiData/GameResults/`

---

## 📊 文件统计

### 核心文件（17个）
| 类别 | 数量 | 说明 |
|------|------|------|
| 测试脚本 | 10 | 完整的四阶段测试系统 |
| 文档 | 4 | 方法论+总结+参考+索引 |
| 配置工具 | 3 | bat+json+txt |

### 测试数据
| 项目 | 数量 |
|------|------|
| 总测试数 | 27 |
| 粗测 | 9 |
| 整数精测 | 4 |
| 小数精测 | 5 |
| 最大值测试 | 9 |
| 总耗时 | ~12分钟 |
| 数据文件 | 27个CSV + 4个JSON |

### 整理效果
| 项目 | 整理前 | 整理后 | 减少 |
|------|--------|--------|------|
| 总文件数 | 33 | 15 | 55% ↓ |
| 测试脚本 | 13 | 8 | 38% ↓ |
| 文档 | 11 | 4 | 64% ↓ |
| 工具 | 9 | 3 | 67% ↓ |

---

## 💡 核心洞察

1. **精度 ≠ 最小有效值**
   - 最小有效值是阈值（门槛）
   - 精度是分辨率（楼梯高度）

2. **即时响应 vs 累积效果**
   - 分析第2帧（即时）
   - 不看第180帧（累积）

3. **稳定区测试**
   - 远离边界
   - 选择线性响应区

4. **多维度分析**
   - 不只看单个字段
   - 加权总和更可靠

---

## 🙏 致谢

感谢在测试过程中发现的所有问题和洞察：
- 混淆精度和阈值的教训
- 单调性不重要的发现
- 小数精度验证的必要性
- 即时响应分析的关键性

---

## 📝 更新日志

### v1.2 (2025-10-20) - 最大值标定
- ✅ 添加最大值测试（9个测试）
- ✅ 确定饱和点为100
- ✅ 验证80-100区间增长
- ✅ 更新所有文档和配置

### v1.1 (2025-10-20) - 文件整理
- ✅ 删除17个过时/冗余文件（55%减少）
- ✅ 保留15个核心文件
- ✅ 更新文档索引
- ✅ 优化文件结构

### v1.0 (2025-10-20) - 完整测试
- ✅ 完成18个测试验证
- ✅ 创建方法论文档
- ✅ 提供代码模板
- ✅ 总结实战案例

---

## 🎯 文件清单（17个核心文件）

### 测试脚本（10个）
✅ dose_response_test.py  
✅ run_dose_response_tests.py  
✅ precision_test_50.py  
✅ run_precision_test_50.py  
✅ precision_test_decimal.py  
✅ test_max_input.py  
✅ test_max_refine.py  
✅ analyze_first_response.py  
✅ analyze_precision_50.py  
✅ analyze_decimal_precision.py  

### 文档（4个）
✅ README_测试文档索引.md（本文件）  
✅ 游戏输入精度测试方法论.md  
✅ 输入特性验证总结.md  
✅ 输入特性快速参考.md  

### 配置（3个）
✅ quick_dose_test.bat  
✅ test_config.json  
✅ requirements_test.txt  

---

## 📞 使用指南

**快速上手**：
1. 阅读"输入特性快速参考.md"（2分钟）
2. 阅读"游戏输入精度测试方法论.md"（10分钟）
3. 运行 `quick_dose_test.bat`（10分钟）

**获取帮助**：
- 查看方法论文档
- 检查测试数据
- 运行分析脚本

---

**最后更新**: 2025-10-20  
**文档版本**: v1.2 (最大值标定完成)  
**文件数量**: 17个核心文件  
**测试完成度**: 100% ✅


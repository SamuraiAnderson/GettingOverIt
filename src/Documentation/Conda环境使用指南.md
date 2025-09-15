# 🐍 Getting Over It 数据分析 - Conda环境使用指南

## 📋 **快速开始**

### **步骤1: 创建conda环境**
```bash
# 方法1: 使用提供的脚本 (推荐)
# Windows:
setup_conda_env.bat

# Linux/Mac:
chmod +x setup_conda_env.sh
./setup_conda_env.sh

# 方法2: 手动执行
conda env create -f environment.yml
```

### **步骤2: 激活环境**
```bash
conda activate getting-over-it-analysis
```

### **步骤3: 验证安装**
```bash
python -c "import pandas, numpy, matplotlib, seaborn, scipy; print('✅ 所有依赖已安装')"
```

### **步骤4: 运行分析**
```bash
# 快速分析
python quick_analysis.py

# 完整分析
python analyze_tracking_data.py

# 生成示例数据 (用于测试)
python generate_sample_data.py
```

---

## 📦 **环境配置详情**

### **environment.yml 说明**
```yaml
name: getting-over-it-analysis  # 环境名称
channels:
  - conda-forge    # 主要包源
  - defaults       # 备用包源
dependencies:
  - python=3.9     # Python版本
  - pandas>=1.3.0  # 数据处理
  - numpy>=1.20.0  # 数值计算
  - matplotlib>=3.3.0  # 基础绘图
  - seaborn>=0.11.0    # 高级绘图
  - scipy>=1.7.0       # 科学计算
  - jupyter            # Jupyter支持
  - ipykernel          # 内核支持
  - pip                # pip包管理
```

### **包功能说明**
- **pandas**: CSV数据读取和DataFrame操作
- **numpy**: 数学计算和数组操作
- **matplotlib**: 基础图表绘制
- **seaborn**: 统计图表和热图
- **scipy**: 峰值检测和统计分析
- **jupyter**: 交互式分析(可选)

---

## 🔧 **常用命令**

### **环境管理**
```bash
# 查看所有环境
conda env list

# 激活环境
conda activate getting-over-it-analysis

# 退出环境
conda deactivate

# 删除环境
conda env remove -n getting-over-it-analysis

# 更新环境
conda env update -f environment.yml
```

### **包管理**
```bash
# 查看已安装包
conda list

# 安装新包
conda install package_name

# 使用pip安装
pip install package_name

# 更新包
conda update package_name
```

### **导出环境**
```bash
# 导出环境配置
conda env export > my_environment.yml

# 导出包列表
conda list --export > requirements.txt
```

---

## 🚀 **使用示例**

### **完整工作流程**
```bash
# 1. 创建并激活环境
conda env create -f environment.yml
conda activate getting-over-it-analysis

# 2. 验证环境
python -c "import pandas as pd; print(f'Pandas版本: {pd.__version__}')"

# 3. 生成测试数据
python generate_sample_data.py

# 4. 快速分析
python quick_analysis.py

# 5. 完整分析 (可选)
python analyze_tracking_data.py
```

### **预期输出示例**
```
🎮 生成示例数据: 30秒, 50Hz, 1500个数据点
✅ 示例数据已生成: sample_tracking_data_30s_50hz.csv

🎯 分析文件: sample_tracking_data_30s_50hz.csv
✅ 数据加载成功: 1500 个数据点

📊 === 基本信息 ===
记录时长: 30.00 秒
采样频率: 50.0 Hz

🎮 === 输入分析 (3DOF) ===
鼠标使用范围: 400 × 300 像素
抓握变化: 15 次
抓握时间: 52.3%

🏆 === 快速结论 ===
✅ 控制精度: 良好
✅ 响应性: 良好
🟡 稳定性: 一般
```

---

## 🎯 **针对不同用户的推荐**

### **🔰 初学者**
```bash
# 最简单的开始方式
conda env create -f environment.yml
conda activate getting-over-it-analysis
python generate_sample_data.py
python quick_analysis.py
```

### **🔬 数据分析师**
```bash
# 完整分析流程
conda activate getting-over-it-analysis
python analyze_tracking_data.py  # 生成详细报告
jupyter notebook  # 启动Jupyter进行交互式分析
```

### **🛠️ 开发者**
```bash
# 开发环境设置
conda activate getting-over-it-analysis
pip install -e .  # 如果有setup.py
pytest  # 运行测试 (如果有)
```

---

## 🐛 **故障排除**

### **问题1: conda命令未找到**
```
'conda' is not recognized as an internal or external command
```
**解决方案**:
1. 安装 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. 重启终端或添加conda到PATH

### **问题2: 环境创建失败**
```
CondaEnvException: Pip failed
```
**解决方案**:
```bash
# 方法1: 分步安装
conda create -n getting-over-it-analysis python=3.9
conda activate getting-over-it-analysis
conda install pandas numpy matplotlib seaborn scipy

# 方法2: 使用pip
pip install -r requirements.txt
```

### **问题3: 包版本冲突**
```
UnsatisfiableError: The following specifications were found to be in conflict
```
**解决方案**:
```bash
# 降低版本要求
conda env create -f environment.yml --force
# 或手动指定版本
conda install pandas=1.3.0 numpy=1.20.0
```

### **问题4: 图表无法显示**
```bash
# 安装GUI后端支持
conda install tk
# 或在代码中设置
plt.switch_backend('Agg')  # 无GUI模式
```

---

## 📊 **性能优化**

### **大数据集处理**
```bash
# 安装加速包
conda install numba
conda install dask
```

### **内存使用优化**
```python
# 在分析脚本中
import pandas as pd
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.precision', 3)
```

### **并行计算**
```bash
# 安装多核支持
conda install joblib
conda install multiprocessing
```

---

## 🔄 **环境维护**

### **定期更新**
```bash
# 更新conda
conda update conda

# 更新环境
conda activate getting-over-it-analysis
conda update --all

# 清理缓存
conda clean --all
```

### **备份环境**
```bash
# 导出当前环境
conda env export > backup_environment.yml

# 恢复环境
conda env create -f backup_environment.yml
```

---

## 💡 **最佳实践**

### **1. 环境隔离**
- 为每个项目创建独立环境
- 不要在base环境中安装包
- 定期清理不用的环境

### **2. 版本管理**
- 固定关键包的版本
- 定期更新environment.yml
- 测试新版本兼容性

### **3. 文档记录**
- 记录环境创建过程
- 注释特殊配置原因
- 分享environment.yml文件

---

## 🎉 **开始使用**

现在您可以：

1. **运行设置脚本**: `setup_conda_env.bat` (Windows) 或 `./setup_conda_env.sh` (Linux/Mac)
2. **激活环境**: `conda activate getting-over-it-analysis`
3. **开始分析**: `python quick_analysis.py`

环境设置完成后，您就拥有了一个专业的Getting Over It数据分析工作环境！🎮🔬✨

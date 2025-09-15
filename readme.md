# Getting Over It - AI训练数据采集系统

一个用于《Getting Over It with Bennett Foddy》的AI训练数据采集和强化学习环境系统。

## 🎯 功能特性

### 🤖 AI训练数据采集
- **高频采样**：30Hz连续数据采集，符合游戏原生采样频率
- **观察空间**：7维状态空间（位置、速度、锤子状态、接地状态）
- **动作空间**：2维连续动作空间（鼠标相对移动）
- **奖励信号**：基于物理响应的多维度奖励函数

### 📊 数据记录
- **实时记录**游戏中所有动态物体（刚体）的运动状态
- **导出静态**场景碰撞体的几何信息和位置数据
- 生成**CSV格式**的数据文件，便于后续分析

### 🎮 实时预览
- **游戏内可视化**：实时显示碰撞体边界框
- **颜色区分**：静态碰撞体（绿色）、动态刚体（红色）
- **快捷键控制**：按 F1 切换预览开关

### 📁 输出文件
- `ContinuousTracking_*.csv` - AI训练原始数据
- `unified_training_dataset.npz` - 统一训练数据集
- `static_colliders.csv` - 静态碰撞体信息
- `dynamic_log.csv` - 动态物体日志

## 🛠️ 构建和安装

### 前置要求

1. **安装 BepInEx**
   - 前往 [BepInEx 发布页面](https://github.com/BepInEx/BepInEx/releases)
   - 下载适用于 Unity IL2CPP 的版本（通常是 `BepInEx_x64_xxx.zip`）
   - 解压到游戏根目录：`...\steamapps\common\Getting Over It\`
   - 启动游戏一次让 BepInEx 初始化

2. **验证游戏目录结构**
   ```
   Getting Over It\
   ├── BepInEx\
   │   ├── core\
   │   │   └── BepInEx.dll
   │   └── plugins\
   ├── GettingOverIt_Data\
   │   └── Managed\
   │       ├── UnityEngine.dll
   │       ├── UnityEngine.CoreModule.dll
   │       └── UnityEngine.PhysicsModule.dll
   └── GettingOverIt.exe
   ```

### 配置项目

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd GettingOverIt
   ```

2. **修改游戏路径**
   
   编辑 `src/GoiHitboxLogger.csproj` 文件第 18 行，将路径改为您的游戏安装路径：
   ```xml
   <GameRoot>C:\Path\To\Your\steamapps\common\Getting Over It</GameRoot>
   ```
   
   **重要提示**：
   - 确保路径中没有多余的空格
   - 游戏数据目录名称是 `GettingOverIt_Data`（无空格）

### 编译方法



在项目根目录打开 PowerShell，运行：

```powershell
cd src
& "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe" GoiHitboxLogger.csproj /p:Configuration=Release
```


### 验证安装

编译成功后应该看到：

1. **编译输出**：
   ```
   已成功生成。
       0 个警告
       0 个错误
   ```

2. **文件自动拷贝**：

## 🎮 使用方法

1. **启动游戏**
   - 正常启动《Getting Over It》
   - 插件会在游戏启动时自动加载

2. **数据输出**
   
   插件运行后会在游戏根目录生成：
   ```
   Getting Over It\HitboxDump\
   ├── static_colliders.csv    # 静态碰撞体信息
   └── dynamic_log.csv         # 动态物理对象数据
   ```

3. **预览控制**
   - **F1键** - 切换碰撞体边界显示
   - **绿色线框** - 静态碰撞体
   - **红色线框** - 动态刚体碰撞体

## 📋 数据格式

### static_colliders.csv
```csv
id,type,layer,path,center,extentOrParams,worldMatrix
123456,BoxCollider,0,World/Ground,0.000|0.000|0.000,size=10.000|1.000|10.000,1.000|0.000|...
```

### dynamic_log.csv
```csv
frame,time,rb_id,path,pos,rot,vel,angvel
1,0.02000,789012,Player/Hammer,1.234|2.345|3.456,0.000|0.000|0.000|1.000,0.100|0.200|0.300,0.010|0.020|0.030
```

## 🛠️ 开发信息

### 项目结构
```
GettingOverIt/
├── src/
│   ├── GoiHitboxLogger.cs     # 主插件代码
│   └── GoiHitboxLogger.csproj # 项目配置
├── .vscode/
│   └── tasks.json             # VS Code 编译任务
├── .gitignore                 # Git 忽略文件
└── readme.md                  # 本文件
```

### 技术细节
- **目标框架**：.NET Framework 3.5
- **依赖项**：BepInEx 5.x、Unity 引擎组件
- **编译器**：MSBuild/Roslyn
- **自动部署**：编译后自动拷贝到游戏插件目录

### 核心依赖模块
```xml
UnityEngine.dll              # Unity 主引擎
UnityEngine.CoreModule.dll   # 核心组件（MonoBehaviour, Transform等）
UnityEngine.PhysicsModule.dll # 物理组件（Rigidbody, Collider等）
BepInEx.dll                  # 模组框架
```

---


Layer 0 锤子
Layer 8 player

---

## 🤖 AI训练工作流程

### 1. 数据采集阶段

#### 手动采集
```bash
# 启动游戏并运行Unity插件
# 使用快捷键进行数据采集：
# F6 - 开始/停止连续跟踪
# F7 - 导出跟踪数据
```

#### 自动化采集
```bash
# 运行自动化采集脚本
src\Scripts\auto_data_collection.bat

# 或使用测试脚本
src\Scripts\test_auto_collection.bat
```

### 2. 数据转换阶段
```bash
# 激活conda环境
conda activate getting-over-it-analysis

# 运行数据转换器
python src/Python/data_converter.py
```

### 3. 数据格式说明
- **观察空间 (7维)**：`[playerX, playerY, velocityX, velocityY, hammerAngle, hammerAngularVel, isGrounded]`
- **动作空间 (2维)**：`[mouseDeltaX, mouseDeltaY]`
- **奖励函数**：基于位置进步、速度稳定性、控制精度、响应效率

### 4. 训练数据输出
```
src/Data/AI_Training/
├── unified_training_dataset.npz    # 统一训练数据集
├── ContinuousTracking_*_ai_training.npz  # 单个文件转换结果
└── ContinuousTracking_*_metadata.json    # 元数据文件
```

### 5. 强化学习环境
- **框架**：Stable Baselines3 + Gymnasium
- **算法**：PPO (Proximal Policy Optimization)
- **环境**：自定义Gym环境 (待实现)
- **控制**：PyAutoGUI鼠标控制

### 6. 项目结构
```
GettingOverIt/
├── src/
│   ├── Unity/                    # Unity C#插件代码
│   │   ├── GoiHitboxLogger.cs    # 主插件
│   │   ├── ContinuousTracker.cs  # 连续数据采集
│   │   └── ...
│   ├── Python/                   # Python分析工具
│   │   ├── data_converter.py     # 数据转换器
│   │   └── ...
│   └── Data/                     # 数据存储
│       ├── AI_Training/          # AI训练数据
│       └── ...
└── readme.md
```

---

## 📝 开发记录

### 关键发现
- **控制机制**：Getting Over It使用纯鼠标移动控制，无按钮输入
- **采样频率**：游戏原生30Hz采样频率
- **最强相关性**：鼠标移动距离与位置变化 (0.122)
- **数据质量**：无缺失值，时间间隔稳定

### 技术栈
- **Unity插件**：BepInEx + C#
- **数据分析**：Python + Pandas + NumPy
- **强化学习**：PyTorch + Stable Baselines3
- **环境管理**：Conda

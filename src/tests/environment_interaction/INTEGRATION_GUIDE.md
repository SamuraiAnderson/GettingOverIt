# Unity端集成指南

## ✅ 集成已完成

Unity端的两个C#文件已经集成到游戏插件中了！

## 📦 已完成的步骤

### ✅ 1. 创建Environment目录
```
src/GameRuntime/Environment/
```

### ✅ 2. 复制C#文件
- `EnvironmentStateManager.cs` → `src/GameRuntime/Environment/`
- `EnvironmentTestController.cs` → `src/GameRuntime/Environment/`

### ✅ 3. 更新项目文件
在 `GoiHitboxLogger.csproj` 中添加了：
```xml
<!-- 环境交互性测试：状态管理和测试控制 -->
<Compile Include="Environment\EnvironmentStateManager.cs" />
<Compile Include="Environment\EnvironmentTestController.cs" />
```

### ✅ 4. 修改JSON序列化
将 `Newtonsoft.Json` 改为使用Unity内置的 `JsonUtility`，提升兼容性。

### ✅ 5. 添加初始化代码
在 `GoiHitboxLogger_Autonomous.cs` 的 `Start()` 方法中添加了：
```csharp
private void InitializeEnvironmentTestController()
{
    GameObject controllerObj = new GameObject("EnvironmentTestController");
    controllerObj.AddComponent<EnvironmentTestController>();
    DontDestroyOnLoad(controllerObj);
    Logger.LogInfo("✅ 环境测试控制器已初始化");
}
```

---

## 🔨 编译和部署

### 步骤1：编译插件

在项目根目录运行：

```bash
cd src/GameRuntime
msbuild GoiHitboxLogger.csproj /p:Configuration=Release
```

或者使用Visual Studio：
1. 打开 `GoiHitboxLogger.csproj`
2. 选择 Release 配置
3. 构建 → 生成解决方案

### 步骤2：验证编译

编译成功后，会自动复制到游戏目录：
```
<游戏目录>/BepInEx/plugins/GoiHitboxLogger.dll
```

### 步骤3：检查文件

确认以下文件已更新：
```bash
# 查看DLL文件时间戳
ls -la "<游戏目录>/BepInEx/plugins/GoiHitboxLogger.dll"
```

---

## 🎮 验证集成

### 方法1：启动游戏查看日志

1. 启动游戏
2. 查看BepInEx日志：`<游戏目录>/BepInEx/LogOutput.log`
3. 搜索关键信息：

```
✅ 环境测试控制器已初始化
🎮 环境测试控制器已启动
📂 信号目录: <path>
```

### 方法2：测试热键

进入游戏场景后，按以下热键测试：

1. 按 `F5` - 保存初始状态
2. 移动角色
3. 按 `F6` - 重置到初始状态
4. 观察角色是否回到初始位置

### 方法3：查看目录

检查是否创建了以下目录：
```
<游戏目录>/GoiData/
├── TestSignals/
├── GameStates/
└── TestResults/
```

---

## 🐛 故障排查

### 问题1：编译失败 - "找不到类型或命名空间"

**可能原因**：缺少引用

**解决方案**：
检查 `.csproj` 文件中的引用路径是否正确，特别是：
- `GameRoot` 路径
- Unity DLL路径

### 问题2：游戏启动后没有日志

**可能原因**：插件未加载或初始化失败

**解决方案**：
1. 检查 `BepInEx/plugins/` 下是否有 `GoiHitboxLogger.dll`
2. 查看BepInEx日志，搜索错误信息
3. 确认BepInEx正确安装

### 问题3：热键无响应

**可能原因**：EnvironmentTestController未初始化

**解决方案**：
1. 查看日志是否有 "环境测试控制器已初始化"
2. 确认在Mian场景中（不是Loader场景）
3. 尝试重启游戏

### 问题4：信号文件不生效

**可能原因**：目录权限或路径问题

**解决方案**：
1. 手动创建 `GoiData/TestSignals/` 目录
2. 检查目录权限（可读写）
3. 查看日志中的实际路径

---

## 📚 下一步

### 运行Python测试

集成完成后，可以运行Python测试脚本：

```bash
cd src/tests/environment_interaction/test_scripts

# 测试完全重置
python test_full_reset.py

# 测试状态保存/加载
python test_state_checkpoint.py

# 测试状态一致性
python test_state_consistency.py
```

### 查看测试文档

详细测试说明见：
- [模块README](README.md)
- [快速启动指南](QUICK_START.md)

---

## 📝 文件位置参考

### Unity端（C#）
```
src/GameRuntime/
└── Environment/
    ├── EnvironmentStateManager.cs       # 状态管理核心
    ├── EnvironmentTestController.cs     # 测试控制器
    └── readme.md                         # 模块文档
```

### Python端（测试）
```
src/tests/environment_interaction/
├── test_scripts/
│   ├── test_full_reset.py
│   ├── test_state_checkpoint.py
│   └── test_state_consistency.py
└── unity_components/                     # 原始文件（已复制）
```

### 游戏数据输出
```
<游戏目录>/GoiData/
├── TestSignals/         # 信号文件
├── GameStates/          # 状态保存
└── TestResults/         # 测试结果
```

---

## ✨ 完成确认

- [x] C#文件已复制到 `src/GameRuntime/Environment/`
- [x] 项目文件已更新
- [x] JSON序列化已修改为Unity兼容方式
- [x] 初始化代码已添加到主插件
- [x] 模块文档已创建

**🎉 集成已完成！可以编译和测试了！**

---

**创建日期**: 2025-10-20  
**集成状态**: ✅ 已完成


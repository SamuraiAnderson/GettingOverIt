# 快速启动指南

## 🚀 5分钟快速开始

### 第一步：Unity端集成

将以下代码添加到 `src/GameRuntime/Core/GoiHitboxLogger.cs`：

```csharp
using GoiHitboxLogger;

void Awake()
{
    // ... 现有代码 ...
    
    // 添加环境测试控制器
    GameObject controllerObj = new GameObject("EnvironmentTestController");
    controllerObj.AddComponent<EnvironmentTestController>();
    DontDestroyOnLoad(controllerObj);
}
```

然后将这两个文件复制到Unity项目：
- `unity_components/EnvironmentStateManager.cs` → `src/GameRuntime/Environment/`
- `unity_components/EnvironmentTestController.cs` → `src/GameRuntime/Environment/`

重新编译Unity插件：
```bash
cd src/GameRuntime
msbuild GoiHitboxLogger.csproj /p:Configuration=Release
```

### 第二步：配置测试环境

1. 编辑 `test_config.json`，设置游戏路径：

```json
{
  "game_exe_path": "C:/你的游戏路径/Getting Over It.exe",
  "game_data_dir": "C:/你的游戏路径/GoiData"
}
```

2. 安装Python依赖：

```bash
pip install -r requirements.txt
```

### 第三步：运行测试

#### 测试1：完全重置
```bash
cd test_scripts
python test_full_reset.py
```

预期耗时：~5分钟

#### 测试2：状态保存/加载
```bash
python test_state_checkpoint.py
```

预期耗时：~8分钟

#### 测试3：状态一致性
```bash
python test_state_consistency.py
```

预期耗时：~6分钟

### 第四步：查看结果

所有结果保存在：`<游戏路径>/GoiData/TestResults/`

- `full_reset_test_report.json` - 重置测试报告
- `checkpoint_test_report.json` - checkpoint测试报告
- `consistency_test_report.json` - 一致性测试报告

---

## 🎮 Unity热键测试（可选）

游戏内直接测试：

1. 启动游戏（已安装插件）
2. 进入游戏场景
3. 按热键测试：
   - `F5` - 保存当前状态为初始状态
   - `F6` - 重置到初始状态
   - `F7` - 保存checkpoint
   - `F8` - 加载最新checkpoint
   - `F9` - 导出当前状态

---

## ❓ 常见问题

### Q: 游戏无法启动
A: 检查 `test_config.json` 中的路径是否正确

### Q: 测试卡住不动
A: 检查 `GoiData/TestSignals/` 目录权限，确保可以创建/删除文件

### Q: Unity端没有响应
A: 查看BepInEx日志：`<游戏路径>/BepInEx/LogOutput.log`

---

## 📚 更多信息

详细文档请参考：[README.md](README.md)


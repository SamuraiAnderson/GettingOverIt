# Player对象分析功能说明

## 🎯 新功能概述

新增了专门的Player对象分析功能，可以全面分析Scene:Main中的Player对象包含的所有内容。

## 📁 文件结构

### `PlayerAnalyzer.cs` - Player对象专用分析器
**专门负责Player对象的深度分析**

#### 核心数据结构：
```csharp
public struct PlayerInfo
{
    public string name;                    // Player名称
    public string fullPath;               // 完整路径
    public Vector3 position;              // 世界坐标
    public Vector3 rotation;              // 旋转角度
    public Vector3 scale;                 // 缩放比例
    public bool isActive;                 // 激活状态
    public int layer;                     // 图层
    public string tag;                    // 标签
    public List<ComponentInfo> components; // 组件列表
    public List<ChildInfo> children;      // 子对象列表
    public PhysicsInfo physics;           // 物理信息
}
```

## 🔍 分析功能

### 1. **智能Player查找**
使用多种策略查找Player对象：
- ✅ 直接查找"Player"名称
- ✅ 通过"Player"标签查找
- ✅ 搜索常见Player命名变体
- ✅ 模糊匹配包含"player"的对象
- ✅ 查找具有玩家控制器组件的对象

### 2. **组件深度分析**
详细分析每个组件的属性：
- **Transform** - 位置、旋转、缩放、子对象数量
- **Rigidbody** - 质量、速度、重力、运动学状态
- **Collider** - 触发器、边界、物理材质
- **Renderer** - 可见性、材质信息
- **AudioSource** - 音频剪辑、音量、播放状态
- **Animation/Animator** - 动画控制器、参数、图层

### 3. **子对象分析**
分析Player的所有子对象：
- 子对象名称和位置
- 激活状态
- 组件数量和类型列表

### 4. **物理系统分析**
专门分析Player的物理属性：
- 刚体信息（质量、速度、重力等）
- 碰撞体类型和数量
- 物理材质属性

## 🎮 使用方法

### 自动模式：
1. **启动游戏** → 进入Scene:Main
2. **自动分析** → 插件自动分析Player对象
3. **查看结果** → 在控制台查看基本信息

### 手动模式：
- **F10** - 手动重新分析Player对象
- **F11** - 导出完整的Player分析报告

### GUI状态显示：
```
F8: 收集碰撞体 | F9: 导出碰撞体 | F10: 分析Player | F11: 导出Player
✅ 已收集 45 个碰撞体
✅ Player已分析: PlayerController (12个组件)
```

## 📊 输出报告示例

```
=== Player对象详细分析报告 ===
分析时间: 2025-09-14 18:40:00
场景名称: Main

📋 基本信息:
  名称: PlayerController
  完整路径: PlayerController
  激活状态: True
  图层: 0 (Default)
  标签: Player

🌍 Transform信息:
  位置: (0.0, 1.5, 0.0)
  旋转: (0.0, 0.0, 0.0)
  缩放: (1.0, 1.0, 1.0)

⚡ 物理信息:
  有刚体: True
    质量: 70.0
    速度: (0.0, -1.2, 0.0)
    角速度: (0.0, 0.0, 0.0)
    使用重力: True
    运动学: False
  有碰撞体: True
    碰撞体类型: CapsuleCollider

🧩 组件列表:
  - Transform (启用: True)
      Position: (0.0, 1.5, 0.0)
      Rotation: (0.0, 0.0, 0.0)
      Scale: (1.0, 1.0, 1.0)
      ChildCount: 3
  - Rigidbody (启用: True)
      Mass: 70
      Velocity: (0.0, -1.2, 0.0)
      UseGravity: True
      IsKinematic: False
  - CapsuleCollider (启用: True)
      IsTrigger: False
      Bounds: ...
  - PlayerMovement (启用: True)
  - PlayerInput (启用: True)
  - AudioSource (启用: True)
      Clip: footsteps
      Volume: 0.8
      IsPlaying: False

👶 子对象:
  - PlayerModel (激活: True)
      本地位置: (0.0, 0.0, 0.0)
      组件数量: 3
      组件类型: Transform, MeshRenderer, MeshFilter
  - PlayerCamera (激活: True)
      本地位置: (0.0, 1.8, 0.0)
      组件数量: 2
      组件类型: Transform, Camera
  - PlayerHammer (激活: True)
      本地位置: (1.0, 0.0, 0.0)
      组件数量: 4
      组件类型: Transform, MeshRenderer, MeshFilter, HammerController
```

## 📁 输出文件

### 位置：
- 文件夹：`PlayerAnalysis/`
- 文件名：`PlayerAnalysis_yyyyMMdd_HHmmss.txt`
- 完整路径示例：`PlayerAnalysis/PlayerAnalysis_20250914_184000.txt`

### 内容包含：
- Player对象的完整层级结构
- 所有组件的详细属性
- 子对象的完整信息
- 物理系统的详细参数
- Transform变换信息

## 🚀 技术特性

### 智能搜索算法：
- 多策略Player对象查找
- 容错性强，适应不同命名方式
- 支持复杂的层级结构

### 全面的组件支持：
- 自动识别组件类型
- 提取关键属性信息
- 支持自定义组件

### 结构化数据：
- 清晰的数据结构设计
- 易于扩展和维护
- 支持程序化访问

## 💡 应用场景

1. **游戏开发调试** - 了解Player对象的完整结构
2. **性能分析** - 检查组件数量和复杂度
3. **物理分析** - 研究Player的物理行为
4. **MOD开发** - 为MOD制作提供详细的Player信息
5. **游戏逆向** - 理解游戏的Player实现机制

这个Player分析器为《Getting Over It》的游戏研究提供了强大的工具，帮助您全面了解Player对象的内部结构和组成！

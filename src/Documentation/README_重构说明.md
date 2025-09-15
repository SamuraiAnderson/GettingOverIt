# GOI Hitbox Logger 重构说明

## 🔄 重构概述

本次重构将原来包含大量调试代码的单一文件分解为清晰的模块化结构，专注于核心功能。

## 📁 文件结构

### `HitboxCollector.cs` - 核心碰撞体收集器
**专门负责PolygonCollider2D的获取和管理**

#### 主要功能：
- ✅ `GetAllMountainColliders()` - 获取Mountain下所有PolygonCollider2D
- ✅ `GetCollidersInGameObject(GameObject)` - 获取指定对象下的碰撞体
- ✅ `ExportCollidersToFile()` - 导出碰撞体数据到文件
- ✅ `FindCollidersByName()` - 按名称查找碰撞体
- ✅ `SortCollidersByVertexCount()` - 按顶点数量排序
- ✅ `PrintColliderStatistics()` - 打印统计信息

#### 核心数据结构：
```csharp
public struct ColliderInfo
{
    public string name;           // 对象名称
    public string fullPath;       // 完整路径
    public Vector3 position;      // 世界坐标
    public PolygonCollider2D collider; // 碰撞体引用
    public int vertexCount;       // 顶点数量
    public bool isTrigger;        // 是否为触发器
    public Bounds bounds;         // 边界信息
}
```

### `GoiHitboxLogger.cs` - 简化的主插件
**专注于核心业务逻辑，去除所有调试功能**

#### 功能特性：
- ✅ 自动检测游戏场景
- ✅ 自动收集碰撞体数据（只收集一次）
- ✅ 简洁的状态显示
- ✅ 基本的按键控制

#### 按键操作：
- **F8** - 手动重新收集碰撞体数据
- **F9** - 导出数据到文件

## 🗑️ 删除的调试功能

### 已移除的功能：
- ❌ 复杂的可视化绘制系统
- ❌ 多种可视化模式（闪烁、纯色、仅顶点）
- ❌ 屏幕绘制和GUI边界显示
- ❌ 对象对比功能
- ❌ 鼠标点击检查功能
- ❌ 详细的GameObject检查器
- ❌ F2-F7按键的调试操作

### 保留的核心功能：
- ✅ PolygonCollider2D数据收集
- ✅ 数据导出功能
- ✅ 基本的状态显示
- ✅ 场景检测逻辑

## 📊 使用示例

### 基本使用
```csharp
// 获取所有Mountain下的碰撞体
List<HitboxCollector.ColliderInfo> colliders = HitboxCollector.GetAllMountainColliders();

// 打印统计信息
HitboxCollector.PrintColliderStatistics(colliders);

// 导出到文件
HitboxCollector.ExportCollidersToFile(colliders);
```

### 高级查询
```csharp
// 查找包含"rock"的碰撞体
var rockColliders = HitboxCollector.FindCollidersByName(colliders, "rock");

// 按顶点数量排序（从多到少）
var sortedColliders = HitboxCollector.SortCollidersByVertexCount(colliders, false);

// 获取特定对象下的碰撞体
GameObject mountain = GameObject.Find("Mountain");
var mountainColliders = HitboxCollector.GetCollidersInGameObject(mountain);
```

## 🎯 代码质量改进

### 模块化
- **单一职责** - 每个类专注一个功能
- **高内聚** - 相关功能聚集在一起
- **低耦合** - 模块间依赖最小化

### 可维护性
- **清晰的命名** - 函数和变量名直观易懂
- **完整的文档** - 每个公共方法都有详细注释
- **结构化数据** - 使用struct组织碰撞体信息

### 性能优化
- **一次性收集** - 避免重复扫描
- **按需操作** - 只在必要时执行expensive操作
- **内存效率** - 移除了大量调试对象

## 🚀 使用建议

1. **游戏启动后** - 插件会自动收集数据
2. **数据更新** - 按F8重新收集最新数据
3. **数据导出** - 按F9导出完整的碰撞体信息
4. **程序化访问** - 通过HitboxCollector类的静态方法访问功能

## 📝 输出文件

导出的文件包含：
- 碰撞体总数统计
- 每个碰撞体的详细信息
- 完整的顶点坐标数据（本地坐标和世界坐标）
- 边界和属性信息

文件位置：`HitboxDump/MountainColliders_[时间戳].txt`

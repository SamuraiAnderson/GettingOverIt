# Testing - 测试模块

## 📁 文件结构

- `PlayerDebugTool.cs` - Player 调试工具（热键调试 + 复制体测试）
- `InputCharacteristicsTest.cs` - 输入特性测试（已移至归档）

---

## PlayerDebugTool - 调试热键

### 基础调试

| 热键 | 功能 |
|------|------|
| **F3** | 打印 Player 状态（位置、速度、角度） |
| **F4** | 打印 Player 碰撞箱信息 |
| **F5** | 暂停/恢复游戏 |
| **F6** | 单帧步进 |
| **F7** | 打印暂停状态 |

### 复制体测试

| 热键 | 功能 |
|------|------|
| **F8** | 创建 3 个 Player 复制体 |
| **F9** | 为复制体设置不同输入 |
| **F10** | 打印复制体状态 |
| **F11** | 销毁所有复制体 |

---

## 复制体功能

### 功能特性

- ✅ 创建多个 Player 复制体（最多 8 个）
- ✅ 每个复制体有独立的 `PlayerControl` 组件
- ✅ 使用 Layer 系统隔离碰撞，复制体之间不会互相碰撞
- ✅ 每个复制体可以接收不同的输入

### 碰撞隔离原理

使用 `Physics2D.IgnoreCollision` 直接忽略特定碰撞体对，**不改变 Layer**：

```
碰撞关系:
├── 原始 Player ←→ Mountain  ✓ 碰撞
├── 复制体 #0   ←→ Mountain  ✓ 碰撞
├── 复制体 #1   ←→ Mountain  ✓ 碰撞
├── 复制体 #2   ←→ Mountain  ✓ 碰撞
│
├── 原始 Player ←→ 复制体    ✗ 忽略
└── 复制体之间                ✗ 忽略
```

**优势**：
- 保持原始 Layer，与环境正常交互
- 只忽略 Player 相关碰撞体之间的碰撞
- 不影响游戏原有物理设置

### 使用方法

1. 进入游戏测试模式
2. 按 **F8** 创建复制体
3. 按 **F9** 为每个复制体设置不同输入
4. 按 **F5** 暂停，**F6** 单帧步进观察
5. 按 **F10** 查看复制体状态
6. 按 **F11** 销毁所有复制体

---

## 依赖服务

- `IGameControlService` - 游戏控制（暂停/步进）
- `PlayerInputService` - 输入注入
- `PlayerStateService` - 状态采集
- `PlayerColliderService` - 碰撞箱采集
- `PlayerDuplicateManager` - 复制体管理

---

## 日志输出示例

```
[Info] === Player 调试工具 ===
[Info] 快捷键:
[Info]   F3: 打印 Player 状态
[Info]   F4: 打印 Player 碰撞箱
[Info]   F5: 暂停/恢复游戏
[Info]   F6: 单帧步进
[Info]   F7: 打印暂停状态
[Info]   --- 复制体测试 ---
[Info]   F8: 创建 3 个复制体
[Info]   F9: 为复制体设置不同输入
[Info]   F10: 打印复制体状态
[Info]   F11: 销毁所有复制体

[Info] ✅ 找到原始 Player: Player
[Info] 📦 原始 Player 碰撞体数量: 3
[Info] ✅ 创建复制体 #0: Player_Duplicate_0, 碰撞体数量=3
[Info] ✅ 创建复制体 #1: Player_Duplicate_1, 碰撞体数量=3
[Info] ✅ 创建复制体 #2: Player_Duplicate_2, 碰撞体数量=3
[Info] ✅ 碰撞隔离已设置：共 36 对碰撞体互相忽略
[Info] 🎮 共创建 3 个复制体，已设置碰撞隔离

[Info] === 复制体状态 (3个) ===
[Info]   #0: pos=(-6.50,-2.50), input=(50.0,0.0), colliders=3
[Info]   #1: pos=(-6.50,-2.50), input=(-50.0,0.0), colliders=3
[Info]   #2: pos=(-6.50,-2.50), input=(0.0,50.0), colliders=3
```

---

**最后更新**: 2025-11-30

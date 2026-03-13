# PlayerControl - Player 控制模块

## 文件说明

- `PlayerInputService.cs` - Player 输入控制服务（通过反射注入 mouseInput）
- `PlayerStateService.cs` - Player 状态采集服务（位置、速度、各部件状态等）

## 已实现功能

### 1. PlayerInputService - 输入控制

**核心功能**：
- ✅ 自动查找 Player 对象
- ✅ 反射获取 PlayerControl 组件
- ✅ 注入 mouseInput 字段（Vector2）
- ✅ 控制 input_enabled 字段
- ✅ 输入范围验证（[-100, 100]）

**API**：
```csharp
bool Initialize()                        // 初始化服务
void SetMouseInput(Vector2 input)        // 设置鼠标输入
Vector2 GetMouseInput()                  // 获取当前输入
void SetInputEnabled(bool enabled)       // 启用/禁用输入
bool GetInputEnabled()                   // 获取输入状态
```

### 2. PlayerStateService - 状态采集

**采集内容**：
- ✅ Player 主体（位置、速度、角速度）
- ✅ Hub（连接器）位置和速度
- ✅ Slider（滑动件）位置和速度
- ✅ Handle（手柄）位置和速度
- ✅ PoleMiddle（锤子中段）位置和速度
- ✅ Tip（锤子尖端）位置和速度
- ✅ 锤子角度计算
- ✅ 时间戳

**状态数据**（39 个浮点数）：
```
[0-4]   Player 主体: x, y, vx, vy, 角速度
[5-9]   Hub: x, y, vx, vy, 角度
[10-14] Slider: x, y, vx, vy, 角度
[15-18] Handle: x, y, vx, vy
[19-22] PoleMiddle: x, y, vx, vy
[23-26] Tip: x, y, vx, vy
[27-38] 锤子角度, 时间戳, 预留位
```

**API**：
```csharp
bool Initialize()                        // 初始化服务
PlayerState GetCurrentState()            // 获取当前状态
float[] GetStateArray()                  // 转换为数组（UDP发送）
Vector2 GetPlayerPosition()              // 获取位置
Vector2 GetPlayerVelocity()              // 获取速度
float GetHammerAngle()                   // 获取锤子角度
```

### 3. PlayerColliderService - 碰撞箱采集

**采集内容**：
- ✅ Player 所有子对象的碰撞箱
- ✅ PotCollider（锅底）顶点
- ✅ TipCollider（锤子尖端）顶点
- ✅ BodyCollider（主体）顶点
- ✅ 实时更新（随 Player 运动）

**API**：
```csharp
bool Initialize(GameObject player)      // 初始化
ColliderData[] CollectPlayerColliders() // 采集所有碰撞箱
Vector2[] GetPotVertices()              // 获取 Pot 顶点
Vector2[] GetTipVertices()              // 获取 Tip 顶点
void UpdateRealtime()                   // 实时更新
```

---

## 游戏运行模式

### 初始化流程

1. 场景加载到 Mian
2. PlayerStateService 初始化（查找所有部件）
3. PlayerInputService 初始化（反射获取控制字段）
4. PlayerColliderService 初始化（采集碰撞箱）

### 调试热键

在游戏运行模式下：

- **F3**: 打印 Player 状态（位置、速度、角度等）
- **F4**: 打印 Player 碰撞箱信息
- **F5**: 测试输入注入（向右移动）
- **F6**: 停止输入

---

## 使用示例

### 设置游戏运行模式

```bash
cd C:\Users\Symbol\aCodes\goi-rl
python -c "from game_mode_controller import GameModeController; GameModeController().set_game_runtime_mode()"
```

### 启动游戏测试

1. 启动 Getting Over It
2. 等待进入 Mian 场景
3. 按 F3 查看 Player 状态
4. 按 F5 测试输入控制
5. 按 F6 停止输入

### 预期日志输出

```
[Info] 🎮 初始化游戏运行模式
[Info] 🔍 查找 Player 组件...
[Info] ✅ 找到 Player 对象
[Info] ✅ PlayerStateService 初始化成功
[Info] ✅ 找到 Player: Player
[Info] ✅ 找到 PlayerControl 组件
[Info] ✅ 成功获取 mouseInput 字段
[Info] ✅ PlayerInputService 初始化成功
[Info] ✅ PlayerColliderService 初始化成功
[Info] 🎮 游戏运行模式初始化完成
```

---

## 技术细节

### 反射访问

```csharp
// 获取 PlayerControl 组件
Component playerControl = player.GetComponent("PlayerControl");

// 获取私有字段
FieldInfo mouseInputField = playerControl.GetType().GetField(
    "mouseInput", 
    BindingFlags.NonPublic | BindingFlags.Instance
);

// 设置值
mouseInputField.SetValue(playerControl, new Vector2(50f, 0f));
```

### 深度查找子对象

```csharp
// 使用队列进行广度优先搜索
Transform FindDeepChild(Transform parent, string name)
{
    Queue<Transform> queue = new Queue<Transform>();
    queue.Enqueue(parent);
    
    while (queue.Count > 0)
    {
        Transform current = queue.Dequeue();
        if (current.name == name) return current;
        
        for (int i = 0; i < current.childCount; i++)
            queue.Enqueue(current.GetChild(i));
    }
    return null;
}
```

---

## 下一步开发

- [ ] UDP 通信服务（接收动作、发送状态）
- [ ] 复制体管理（多 Player 实例）
- [ ] 游戏测试模式（自动化测试）

---

## 编译状态

✅ **已编译部署**
- DLL: `GameRuntime_v2.dll`
- 位置: `BepInEx/plugins/GameRuntime_v2.dll`
- 状态: 就绪

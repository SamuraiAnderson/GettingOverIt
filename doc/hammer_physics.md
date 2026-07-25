# 锤子物理模型与力的传递（IL 反编译实证）

本文记录游戏原生的锤子控制物理：`mouseInput → fakeCursor → 关节电机 → player 受力` 的完整链路及其数学表示。
全部结论来自 `Assembly-CSharp.dll` 的 IL 反汇编（`PlayerControl.FixedUpdate` / `Saviour`，
dump 方法见 `.cursor/skills/decompile-dotnet/SKILL.md`；IL 产物在游戏 `Managed\_il_dump\`，不入仓库）。

## 一、结论概览

- **不是弹簧**。fakeCursorRB 与锤子之间没有任何 Joint；锤子由两个**带力矩/力上限的速度伺服电机关节**驱动：
  - `HingeJoint2D hj` —— 锤子绕枢轴的**角度自由度**（电机转速追 cursor 的方位角）；
  - `SliderJoint2D sj` —— 锤杆的**径向伸缩自由度**（电机线速度追 cursor 沿杆的投影）。
- fakeCursorRB 本身是**运动学驱动**（`MovePosition` + 手工 `set_velocity`），每帧按注入的 mouseInput 积分，
  并被**硬钳制在 player 周围半径 3.5 m 的圆盘内**。
- "手感像弹簧"的来源：速度伺服 = 目标速度 ∝ 误差、输出力 ≤ 上限。小误差区近似弹簧-阻尼；
  大误差区饱和成**恒力/恒扭矩**（这是计算速度包络的关键，见第六节）。

## 二、控制链（每个 FixedUpdate，Δt = fixedDeltaTime）

```
注入 (dx,dy)（Rewired GetAxis 拦截）
  → mouseInput = Lerp(oldMouse, (dx,dy)·sensitivity·mouseCurve(|·|), 0.5)   # 输入平滑
  → 自适应增益 g 更新 → cursor 目标位置 = cursor + g·mouseInput
  → (+ 静默回吸向 tip) → 钳制到 |cursor − player| ≤ 3.5
  → fakeCursorRB.MovePosition(目标)
  → hinge 电机: motorSpeed = 角度伺服律(θ_cursor − θ_joint)
  → slider 电机: motorSpeed = 径向伺服律(cursor 沿杆投影)
  → Physics2D 求解器施加 ≤ maxMotorTorque/Force 的力 → 经关节反作用传给 player
```

注意：RL 注入的动作 `(dx,dy) ∈ [-100,100]` **不是米**——要依次乘 `mouseSensitivity`、
`mouseCurve(|·|)`、0.5 平滑、自适应增益 `g` 才变成 cursor 的世界位移。

## 三、fakeCursor 运动学（数学表示）

记 player 位置 `x_p`、cursor 位置 `x_c`、锤头（tip）位置 `x_t`、注入动作 `a`。每个 FixedUpdate：

```
m   ← Lerp(m_old, mouseSensitivity · mouseCurve(|a|) · a, 0.5)        # 输入平滑（EMA）
g   ← Lerp(g, max(0.1·|m|, 0.001), 0.05) + 0.005                      # 自适应增益（mouseVelocityAverage）
x_c' = x_c + g·m                                                       # 积分
x_c' += 0.5·(x_t − x_c) · clamp(0.2 − min(g, 0.2), 0, 0.2)            # 静默回吸：无输入时 cursor 缓慢滑向 tip
if |x_c' − x_p| > 3.5:  x_c' = x_p + 3.5 · normalize(x_c' − x_p)      # 硬半径钳制（R = 3.5 m）
MovePosition(x_c');  velocity = x_c' − x_c                             # 运动学驱动，非弹簧
```

对 RL 的两个直接推论：

- **零动作 ≠ 保持**：`g` 衰减后回吸项生效，cursor 会自己漂向 tip（光标"吸附锤头"，与 `Saviour.Load`
  收尾的 `cursor = hammer.position` 同语义）。
- 观测里 `cursor_rel = x_c − x_p` 的定义域是半径 3.5 的圆盘，这是动作空间的**真实可达边界**。

## 四、Hinge 角度伺服（切向通道）

记 hinge 枢轴位置 `x_h`（`hj.transform.position`）、关节角 `θ_j`（度）、参考角 `θ_ref`。

```
θ*  = atan2(y_c − y_h, x_c − x_h)·180/π − θ_ref          # cursor 相对枢轴的方位角（IL 里以 −180 恒等式实现）
e   = −DeltaAngle(θ*, θ_j)                                # 最短路角度误差（度）
ẽ   = sign(e) · max(|e|/2, e²) · clamp01(|e|/angleEpsilon) # 非线性整形 + 死区平滑
ω   = clamp( 3·ẽ · clamp01(|x_c − x_h| / deadzone)², −800, +800 )   # 目标转速（度/秒）
```

- 纯 P 控制：IL 中微分项系数为字面量 `0`（`(ẽ − oldAngle)·0`），已被作者禁用。
- 整形项 `max(|e|/2, e²)`：|e| > 0.5° 后二次项主导；**|e| ≳ 16° 时 3·e² ≥ 800，目标转速饱和**。
- `clamp01(|x_c − x_h|/deadzone)²`：cursor 贴近枢轴时电机趋零——"光标收回身边 = 锤子瘫软"。
- 求解器以 `|τ| ≤ hj.motor.maxMotorTorque` 驱动关节转速逼近 ω。

## 五、Slider 径向伺服（径向通道）

记杆方向单位向量 `r̂ = normalize(x_t − x_h)`、cursor 到 tip 向量 `w = x_t − x_c`：

```
κ = r̂ · ŵ                       # 对齐度（cursor 是否在杆轴线上）
ρ = |w| · κ                      # cursor 沿杆方向到 tip 的带符号投影距离
λ = 16 − max(0.001·τ_reaction, 5)  # 载荷反馈：slider 关节反扭矩越大，驱动越弱（默认 λ = 11）
v = clamp( −ρ·|ρ| · λ · κ⁴, −50, +50 )   # 目标线速度
```

- 二次误差 `ρ|ρ|` + 对齐度四次方 `κ⁴`：只有 cursor 大致落在杆的延长线上，伸缩才有力；
  横向拖动几乎不驱动 slider（由 hinge 承担）。
- `λ` 是**力反馈**：撑地载荷大时自动减速（τ_reaction ≥ 16000 时 λ ≤ 0，驱动反转/归零）。
- 求解器以 `|F| ≤ sj.motor.maxMotorForce` 驱动关节线速度逼近 v。

## 六、力如何传到 player（两种机制）

### 6.1 自由空间（锤未接触地形）

电机力矩是 player–hammer 系统的**内力偶**：hinge 给锤 `+τ`、给 player `−τ`。系统总动量只受重力，
但甩锤可重分配角动量、移动合成质心 → 荡摆/翻滚（起跳前的蓄力摆）。

### 6.2 接触锚定（tip 顶/勾在地形点 p）——攀爬的主通道

tip 被接触约束固定后，player–锤杆–地形构成闭链，player 近似绕 p 做受控杠杆运动。
准静态近似（杆质量忽略、player 质量 m 集中于锅），记杆臂 `L = |p − x_h|`、
切向单位向量 `t̂ ⊥ (p − x_h)`、径向 `r̂`：

```
切向力（hinge）:  F_t = τ / L,        |F_t| ≤ τ_max / L      # 绕锚点甩摆
径向力（slider）: F_r 沿 r̂,          |F_r| ≤ F_max          # 撑离/拉向锚点
瞬时可达力集合:  F ∈ { α·t̂ + β·r̂ : |α| ≤ τ_max/L, |β| ≤ F_max } ∩ C(p, n̂)
```

`C(p, n̂)` 是接触可行锥：**推**要求 −F 压紧接触面（法向分量 + 摩擦锥）；**勾**靠几何互锁可承受拉力。
这就是"不同接触角度给出不同速度方向"的数学形式——接触法向 `n̂` 决定 `C`，
从而裁剪可达力集合的方向。

**速度增量（冲量）**：接触时长 `T_c` 内 `Δv = (1/m)∫F dt`，上界

```
|Δv| ≤ √((τ_max/L)² + F_max²) · T_c / m
```

**饱和恒力区**：锚定时关节几乎不转/不滑，而误差整形后目标速度极大（ω 动辄饱和 ±800°/s），
电机长期顶在 `τ_max`/`F_max` 上 → 接触期间近似**恒力**，冲量 ≈ 力上限 × 接触时间，
与 cursor 拉多远无关（拉更远只改变方向与维持时间）。这使 `V(p)`（出射速度包络）可解析估计，
不必完全依赖经验探测。

## 六点五、接触如何产生速度（冲量模型，含 pot 接触）

物理求解器是 Box2D（Physics2D）顺序冲量法。**系统里唯一的能量来源是 hinge/slider 电机做的功**
（功率 `P = τ·ω + F·v`）+ 重力势能；所有接触（锤头/锅体）都是约束，只做**重定向或耗散**，
不注入能量。"接触产生速度"的准确说法是：接触约束把电机功/重力势转换成特定方向的动量。

### 接触冲量的数学（单点接触，法向 n̂、切向 t̂）

```
法向: j_n ≥ 0 使得 v_rel·n̂ ≥ −e·v_n⁻        # e=restitution，游戏无弹跳 → e≈0，落地完全非弹性
切向: |j_t| ≤ μ·j_n                            # 库仑摩擦锥；锥内→粘着(stick)，锥外→滑动
速度变化: Δv = (j_n·n̂ + j_t·t̂)/m,  Δω = (r × j)/I    # r = 接触点相对质心
有效质量: m_eff = (1/m + (r⊥·n̂)²/I)⁻¹                 # 旋转自由度会"稀释"冲量效果
```

### 锤头（tip）接触：主动通道 + 双稳态摩擦（IL 实证，`HammerCollisions`）

tip 碰撞体的 `PhysicsMaterial2D` 在**每个 FixedUpdate 动态切换**（静/滑两套摩擦系数）：

```
slide 判定（OnCollisionStay2D）：
  任一接触点两帧间移动 < moveThreshold(0.03 m) → slide = false（换 staticFriction 材质）
  强制滑动：|v_tip| > 0.3 且 max((1+|j_t|)/(1+|j_n|)) > 5 → slide = true
  离开接触（OnCollisionExit）→ slide = true（默认滑动材质）
```

这是**抓地迟滞（grip hysteresis）**：锤头一旦"站稳"（接触点不动）就换高摩擦材质，锚点更牢；
一旦开始刮擦就换低摩擦材质，更易滑。对控制的含义：勾/撑的成败对"接触初期是否立即稳住"高度敏感
——这正是该游戏操作不连续性的物理根源之一。

另一个接触触发的**控制链不连续**（`OnCollisionEnter2D` → `HammerReturn`）：
碰撞相对速度 `|v_rel| > 14` 且 tip 低于 player（`tip.y − 0.4 < player.y`）时，
`mouseSnap = true` → 下一帧**跳过输入积分、cursor 直接吸附到 tip**。
即高速砸地会把玩家的"手"拽回锤头——RL 高速挥击后注入的动作会被丢弃一帧。

### 锅体 / 身体（pot）接触：纯被动通道

pot 碰撞没有任何专属代码（`GroundCol` 只管音效颜色；`PoseControl` 是纯视觉 IK），
就是标准 Box2D 刚体接触。它**不能产生速度**，只有三种作用：

1. **终止/吸收**：落地时法向冲量消掉法向速度分量（e≈0 → 动能直接耗散，不反弹）。
   高度差 h 的下落损失动能 ≈ m·g·h·cos²(坡角)。
2. **被动滑动**：斜坡上 `a = g(sinθ − μcosθ)` 沿坡滑，μ 来自 pot 材质（prefab 序列化，
   运行时可读 `collider.friction`）。这是"滑回起点"的机制。
3. **支点（fulcrum）**：pot 触地粘着时，接触点成为免费转动支点——hinge 电机的力偶可绕它
   把身体撬起/翻转，而不需要锤头另找锚点；粘着的摩擦力同时把旋转转换成净水平推进。
   数学上：绕接触点 p 的角动量方程 `I_p·ω̇ = τ_motor + m·g·(x_com − x_p)`，
   电机扭矩与重力矩围绕 p 竞争。

### 三条速度来源路径汇总

| 路径 | 机制 | 能量 | 数学核心 |
|------|------|------|----------|
| 锤头锚定发力 | 闭链约束把电机扭矩/力转成 player 加速度 | 电机注入 | 第六节 `F ∈ {α t̂ + β r̂} ∩ C(p,n̂)` |
| 自由空间甩摆 | 内力偶重分配动量/角动量 | 电机注入 | 总动量只受重力，质心弹道不变 |
| pot 接触 | 法向终止 + 摩擦耗散/粘着支点 | 只耗散 | 冲量模型 + 绕支点杠杆方程 |

## 七、常数表

IL 中的明文常数（`IL_PlayerControl.txt` 可复核）：

| 常数 | 值 | 语义 |
|------|-----|------|
| R | 3.5 m | cursor 相对 player 的最大半径（硬钳制） |
| 输入平滑 | Lerp 0.5 | mouseInput EMA |
| 自适应增益 | 0.1 / 0.001 / 0.05 / +0.005 | mouseVelocityAverage 更新式 |
| 静默回吸 | 0.5 × clamp(0.2 − min(g,0.2), 0, 0.2) | 无输入时 cursor 向 tip 滑动 |
| hinge P 增益 | 3（D 项 ×0 禁用） | 角度伺服 |
| hinge 整形 | sign(e)·max(|e|/2, e²) | ~16° 后目标转速饱和 |
| hinge 转速上限 | ±800 °/s | motorSpeed clamp |
| slider 误差 | −ρ·\|ρ\| | 二次 |
| slider 载荷反馈 | λ = 16 − max(0.001·τ_react, 5) | 默认 11，重载减速 |
| slider 对齐度 | κ⁴ | 偏轴不驱动 |
| slider 速度上限 | ±50 | motorSpeed clamp |
| tip 静/滑判定 | moveThreshold 0.03 m | 接触点两帧位移 < 此值 → 静摩擦材质 |
| tip 强制滑动 | \|v_tip\|>0.3 且 j_t/j_n 比>5 | 切向冲量占优 → 滑动材质 |
| HammerReturn | \|v_rel\|>14 且 tip 低于 player 0.4 m | cursor 强制吸附 tip（丢一帧输入） |

prefab 序列化、IL 不可见但**运行时可反射读取**的参数（需要时经 GameTesting/`PlayerDebugTool` 探测）：
`mouseSensitivity`、`mouseCurve`/`trackpadCurve`、`deadzone`、`angleEpsilon`、
`hj.motor.maxMotorTorque`、`sj.motor.maxMotorForce`、各刚体质量/惯量。
其中 `τ_max`、`F_max`、`m` 是第六节速度包络公式仅缺的三个数。

## 八、对训练侧的推论

1. **极坐标是原生驱动空间**：物理自由度就是（hinge 角 θ，slider 半径 ρ），动作若重参数化为
   cursor 相对 player 的 `(θ, r)` 目标，与执行机构一一对应（见 `doc/training.md` 搜索空间讨论）。
2. **动作单位换算**：注入 `(dx,dy)` 要过 sensitivity × curve × 0.5 平滑 × 自适应增益 g 才是米；
   增益随输入幅度自适应，同样的动作序列在不同"手速历史"下 cursor 位移不同（观测已含 cursor 状态，
   马尔可夫性由 33D 状态保证）。
3. **零动作漂移**：静默回吸使 cursor 主动滑向 tip，BC/PPO 里"输出 0"不是中性动作。
4. **速度包络可解析**：拿到 `τ_max/F_max/m` 后，任一接触点 p 的出射速度上界 =
   `√((τ_max/L)² + F_max²)·T_c/m`，方向锥由接触几何 `C(p, n̂)` 裁剪——可供性探测只需标定
   接触时间 `T_c` 与摩擦/互锁形态，不必盲扫全部动作参数。

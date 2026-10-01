---
name: decompile-dotnet
description: 反编译/反汇编本项目游戏的 .NET 程序集（Getting Over It 的 Assembly-CSharp.dll 等）以查清游戏内部逻辑。当需要弄清存档/载入(Saviour/SaveState)、PlayerControl/fakeCursor/关节、场景与菜单(Loader/SceneMenu/NewGame/Continue)、Rewired 输入、或任何"游戏原生行为"时使用；也适用于任意 Unity/Mono .NET DLL 的 IL 查看。触发词：反编译、decompile、dnSpy、ILSpy、dnlib、Assembly-CSharp、Saviour、存档、IL、游戏内部逻辑。
---

# Decompile .NET (本项目游戏程序集)

用来查清游戏内部实现（Python/C# 插件读不到的部分）。**本机环境有坑，直接用下面验证过的路径，别走弯路。**

## 环境事实（已验证，勿重复踩坑）

- 目标程序集：`<GameRoot>\GettingOverIt_Data\Managed\Assembly-CSharp.dll`（+ `Assembly-CSharp-firstpass.dll`）。GameRoot 见 `src/config/project.json` 或 `src/GameRuntime_v2/GameRuntime_v2.csproj` 的 `<GameRoot>`。
- 反编译工具：dnSpy 在 `C:\Users\Symbol\code_tool\dnSpy\`，引擎 DLL 在 `...\dnSpy\bin\`（`dnlib.dll`、`ICSharpCode.Decompiler.dll`）。
- **`dnSpy.Console.exe` 不可用**：在非交互控制台会崩（`Console.OutputEncoding` 抛 `IOException 句柄无效`，反编译前就退出）。换新窗口也没用。别再尝试它。
- **没有 .NET SDK**（只有 runtime + `dotnet.exe`），所以 `dotnet tool install -g ilspycmd` 用不了。
- `ICSharpCode.Decompiler.dll` 是 **.NET 5** 程序集，且其 C# AST 反编译 API（`DecompilerContext`/`AstBuilder`）依赖 `dnSpy.Contracts` 一堆内部类型（如 `MetadataTextColorProvider`），脚本化成本高 → **不要走完整 C# 反编译**。
- **可用路径 = 纯 dnlib 的 IL dump**（`dnlib.dll` 在 Windows PowerShell 与 pwsh 7 下都能加载；用 `LoadFrom` + `Unblock-File` 规避 `0x80131515` 已阻止文件错误）。

## 用法（首选）

用脚本 dump 指定类型的 IL：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .cursor/skills/decompile-dotnet/scripts/dump_il.ps1 `
  -Assembly "C:\Users\Symbol\software\game_store\steam\steamapps\common\Getting Over It\GettingOverIt_Data\Managed\Assembly-CSharp.dll" `
  -Types Saviour,SaveState,PlayerControl,SceneMenu,Loader
```

输出到 `<Managed>\_il_dump\`：每个类型一个 `IL_<Type>.txt`，外加 `_all_types.txt`（全类型清单，用来先确认类名/找候选）。`-Types` 支持通配，如 `'*Save*'`、`'*Menu*'`。

工作流：
1. 先不确定类名时，跑一次任意 `-Types '*'` 或读 `_all_types.txt` 定位目标类。
2. 再对目标类 dump IL，用 Read 工具读 `IL_*.txt`。
3. 需要引用解析（跨程序集）时脚本已自动把 Managed 目录加入 resolver；跨 DLL（如 firstpass）就把对应 dll 传给 `-Assembly` 再跑一次。

## 读 IL 速查

- `ldstr "..."` → 字符串常量（PlayerPrefs 键、日志、版本判断）。
- `call/callvirt <Method>` → 调用链（如 `PlayerPrefs::GetString`、`XmlSerializer::Deserialize`、`Rigidbody2D::set_position`）。
- `stfld/ldfld <Field>`、`call set_X/get_X` → 字段/属性读写（看"存了什么/写回了什么"）。
- `newarr` + 循环里的 `stelem` → 数组构造（如逐刚体位姿数组）。
- `ldc.r4 <num>` → 硬编码浮点常量（如原生姿态坐标）。
- 分支 `br*/b**` 的目标偏移可对照左侧 `XXXX:` 地址。

## 已知结论（可直接复用，省得重查）

- 存档 = `SaveState`（`playerPos/playerRot`、`hinge/slider` 关节位速、**每刚体 `rbPositions/rbAngles/rbLinearVelocities/rbAngularVelocities`**），XML 序列化进 `PlayerPrefs` 键 `SaveGame0/1`，计数键 `NumSaves`。
- Continue 载入 = `Saviour.LoadNewestSave` → `Saviour.Load(SaveState)`：临时切 `Physics2D.simulationMode=Script` → 逐刚体写回 position/rotation/velocity → 收尾 **`cursor.position = hammer.position`（光标吸附锤头，零牵引）** → 切回 `FixedUpdate` + 关节电机复位。
- 掉水/按 r 复位 = `Saviour.ResetPlayerButNotDialogue`，内含**硬编码锅底原生姿态**（6 刚体坐标、`playerRot=Euler(0,0,359.91)`、`hingePos=-1109.84`、`sliderPos=-0.7707`）。
- 自动化开局点的是 `Canvas/Column/NewGame`（新游戏，非 Continue）。
- 锤子控制 = `PlayerControl.FixedUpdate`：**无弹簧关节**，fakeCursorRB 运动学驱动（MovePosition，硬钳制 |cursor−player|≤3.5m），锤子由 `HingeJoint2D hj`（角度伺服，P 增益 3、误差整形 `sign(e)·max(|e|/2,e²)`、转速 clamp ±800°/s、D 项 ×0 禁用）+ `SliderJoint2D sj`（径向伺服，`−ρ|ρ|·λ·κ⁴`、λ=16−max(0.001·τ_react,5)、clamp ±50）的限力电机驱动。完整数学表示见 `doc/hammer_physics.md`。

## 注意

- 反编译只读，不改游戏文件。IL dump 产物默认写到游戏 `Managed\_il_dump\`（临时分析用），不要提交进仓库。
- 路径含空格，命令里务必给 `-Assembly` 加引号。

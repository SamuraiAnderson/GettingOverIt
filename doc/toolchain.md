# 外部工具链

本文介绍项目依赖的两类外部工具：**编译工具链**（把 C# 插件编译成游戏能加载的 DLL，跑训练必需）和**反编译工具链**（查看游戏原生代码，只有在需要弄清游戏内部逻辑时才用）。

训练/运行需要的 Python 依赖不在这里叙述——它们已经完整列在仓库根目录的 [`requirements.txt`](../requirements.txt) 里，直接 `pip install -r requirements.txt` 即可，无需重复描述。

> 面向 AI agent 的详细操作步骤另有两份 skill（`.cursor/skills/build-plugin`、`.cursor/skills/decompile-dotnet`）。本文是给人看的版本，讲清"用什么、为什么、本机有哪些坑"，具体命令与本文一致。

---

## 一、编译工具链（跑训练必需）

C# 插件 `GameRuntime_v2` 必须先编译成 `GameRuntime_v2.dll`，游戏启动时由 BepInEx 加载进游戏进程，才能和 Python 通信。

### 用到的工具

| 工具 | 作用 | 本机情况 |
|---|---|---|
| MSBuild（VS2019 Build Tools 自带） | 把 `.csproj` 编译成 DLL | 已装在 `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\` |
| .NET 运行时 | 插件运行依赖 | **只有运行时，没有 SDK** |
| BepInEx | Unity 插件框架，负责在游戏启动时把我们的 DLL 注入进游戏 | 需事先装进游戏目录（见 [usage.md](../usage.md) 前置条件） |
| Harmony | 运行时给游戏方法打补丁（拦截 Rewired 鼠标输入、跳过原生 `PlayerControl.Update`），随 BepInEx 提供 | 编译时由 `.csproj` 引用 |

### 本机的两个关键事实（避免踩坑）

1. **不能用 `dotnet build`**。本机只装了 .NET 运行时、没装 SDK，`dotnet build` 会报 "No .NET SDKs were found"。编译一律走下面的 MSBuild 命令。
2. **编译前先关游戏**。`.csproj` 里有一步 PostBuild 会自动把编译好的 DLL 拷进游戏的 `BepInEx\plugins\`；如果游戏还开着，DLL 被占用，拷贝会失败。

### 编译命令

在仓库根目录（PowerShell）执行：

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe' 'src\GameRuntime_v2\GameRuntime_v2.csproj' /t:Build /p:Configuration=Release /nologo /v:minimal
```

成功时输出里会有这两行：

```
GameRuntime_v2 -> C:\Users\Symbol\aCodes\GettingOverIt\bin\Release\GameRuntime_v2.dll
已拷贝: ...\Getting Over It\BepInEx\plugins\GameRuntime_v2.dll
```

改完代码后**必须重启游戏进程**新插件才生效（插件只在启动时加载一次）。

### 排错

| 现象 | 原因与解决 |
|---|---|
| 报 "No .NET SDKs were found" | 误用了 `dotnet build`，改用上面的 MSBuild 命令 |
| PostBuild 拷贝失败 / DLL 被占用 | 游戏还开着，关掉游戏进程再重编 |
| MSBuild 路径不存在 | VS 版本可能变了，用 `Get-ChildItem -Path 'C:\Program Files (x86)' -Recurse -Filter MSBuild.exe` 重新定位 |
| 改动没生效 | 忘了重启游戏；或编译的是别的 Configuration（本项目用 `Release`） |

> 详细步骤见 `.cursor/skills/build-plugin/SKILL.md`。

---

## 二、反编译工具链（只有查游戏内部逻辑时才用）

这条工具链**不在训练主流程里**。只有当你需要弄清游戏自己是怎么实现某件事的——比如存档/读档（`Saviour`/`SaveState`）、锤子控制（`PlayerControl`/`fakeCursor`/关节）、场景与菜单（`Loader`/`NewGame`/`Continue`）、Rewired 输入判定——才需要它。

项目里不少设计依据都来自这条工具链。例如观测里那 5 维接触信号的阈值常量（`CONTACT_EPSILON=0.03` 等），就是反编译游戏代码后从游戏原生的接触判断里直接抄出来的，不是拍脑袋的启发式（见 [findings.md](findings.md) 与 [optimization_roadmap.md](optimization_roadmap.md) P2.1）。

### 反编译什么

游戏的 .NET 程序集：

- `<GameRoot>\GettingOverIt_Data\Managed\Assembly-CSharp.dll`（游戏主逻辑）
- 同目录的 `Assembly-CSharp-firstpass.dll`（部分基础代码）

`<GameRoot>` 就是游戏安装目录，取值见 [`src/config/project.json`](../src/config/project.json) 或 `.csproj` 里的 `<GameRoot>`。

### 用到的工具与本机坑

| 工具 | 本机情况 |
|---|---|
| dnSpy | 装在 `C:\Users\Symbol\code_tool\dnSpy\`；引擎 DLL（`dnlib.dll`、`ICSharpCode.Decompiler.dll`）在其 `bin\` 下 |
| dnlib（dnSpy 自带） | **本机唯一可靠的反编译路径**，用它导出 IL |

本机验证过的三个坑，直接绕开别再试：

1. **`dnSpy.Console.exe` 不能用**。在非交互控制台里它会因为设置输出编码抛异常，在反编译开始前就崩掉，换新窗口也没用。
2. **装不了 `ilspycmd`**。它需要 .NET SDK，本机只有运行时，`dotnet tool install -g ilspycmd` 用不了。
3. **别走完整 C# 反编译**。`ICSharpCode.Decompiler` 那套 C# 源码还原 API 依赖 dnSpy 内部一堆类型，脚本化成本很高。

所以本机可行的办法就一个：**用 dnlib 把指定类型的 IL（中间语言）导出成文本文件，然后直接读**。IL 不是 C# 源码，但足够看清"调用了谁、读写了哪些字段、硬编码了哪些常量"。

### 用法

用现成脚本导出指定类型的 IL：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .cursor/skills/decompile-dotnet/scripts/dump_il.ps1 `
  -Assembly "C:\...\Getting Over It\GettingOverIt_Data\Managed\Assembly-CSharp.dll" `
  -Types Saviour,SaveState,PlayerControl,SceneMenu,Loader
```

- 输出到 `<Managed>\_il_dump\`：每个类型一个 `IL_<Type>.txt`，外加一个 `_all_types.txt`（全类型清单，先用它确认类名）。
- `-Types` 支持通配，如 `'*Save*'`、`'*Menu*'`。
- 不确定类名时，先跑一次读 `_all_types.txt` 定位目标类，再对目标类导出 IL。
- 路径含空格，`-Assembly` 一定要加引号。

### 读 IL 速查

| IL 指令 | 含义 |
|---|---|
| `ldstr "..."` | 字符串常量（PlayerPrefs 键、日志、版本判断） |
| `call` / `callvirt <Method>` | 方法调用链（如 `PlayerPrefs::GetString`、`Rigidbody2D::set_position`） |
| `stfld` / `ldfld <Field>`、`call set_X/get_X` | 字段/属性读写（看"存了什么、写回了什么"） |
| `ldc.r4 <num>` | 硬编码浮点常量（如原生姿态坐标、阈值） |

### 注意

- 反编译**只读**，不改游戏文件。
- IL 导出产物默认写到游戏目录的 `_il_dump\`（临时分析用），**不要提交进仓库**。

> 详细步骤、已查清的游戏内部结论（存档格式、锤子控制数学）见 `.cursor/skills/decompile-dotnet/SKILL.md` 与 [hammer_physics.md](hammer_physics.md)。

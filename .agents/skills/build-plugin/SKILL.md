---
name: build-plugin
description: 编译本项目的 C# BepInEx 插件 GameRuntime_v2（用本机 VS2019 BuildTools 的 MSBuild）。当修改了 src/GameRuntime_v2 下任意 .cs/.csproj 后需要让改动在游戏里生效、用户要求「编译/重新编译/build 插件」、或需产出并部署 GameRuntime_v2.dll 时应用。
disable-model-invocation: true
---

# 编译 GameRuntime_v2 插件

## 关键事实

- 本机**只有 .NET 运行时，没有 .NET SDK** → `dotnet build` 会报 "No .NET SDKs were found"，不要用。
- 编译走本机已装的 **VS2019 Build Tools 的 MSBuild**（下方绝对路径）。
- 工程文件：`src/GameRuntime_v2/GameRuntime_v2.csproj`
- 产物：`bin/Release/GameRuntime_v2.dll`
- `.csproj` 的 **PostBuild 会自动**把 dll 拷到游戏目录 `BepInEx\plugins\GameRuntime_v2.dll`，无需手动复制。
- 编译改动**必须重启游戏进程**才生效（插件在启动时加载）。

## 编译命令

在仓库根目录（PowerShell）执行：

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe' 'src\GameRuntime_v2\GameRuntime_v2.csproj' /t:Build /p:Configuration=Release /nologo /v:minimal
```

成功输出形如：

```
GameRuntime_v2 -> C:\Users\Symbol\aCodes\GettingOverIt\bin\Release\GameRuntime_v2.dll
已拷贝: ...\Getting Over It\BepInEx\plugins\GameRuntime_v2.dll
```

## 流程

1. 若游戏正在运行，**先关闭游戏进程**（否则 PostBuild 拷贝会因 dll 被占用而失败）。
2. 运行上面的编译命令。
3. 确认输出里有 `-> ...\bin\Release\GameRuntime_v2.dll` 且「已拷贝」到 `BepInEx\plugins`。
4. 重启游戏 / 重跑对应脚本让新插件生效。

## 排错

- **报 "No .NET SDKs were found"**：说明误用了 `dotnet build`，改用上面的 MSBuild 命令。
- **PostBuild 拷贝失败 / dll 被占用**：游戏还开着，关掉游戏进程后重编。
- **MSBuild 路径不存在**：用 `Get-ChildItem -Path 'C:\Program Files (x86)' -Recurse -Filter MSBuild.exe` 重新定位（可能升级到别的 VS 版本）。
- **改动没生效**：忘了重启游戏；或改的是别的 Configuration（本项目用 `Release`）。

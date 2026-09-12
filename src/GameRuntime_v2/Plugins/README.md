# Plugins - BepInEx 插件入口（占位）

> **现状**：本目录当前不含代码。插件的 BepInEx 主入口（`BepInPlugin` / `BaseUnityPlugin`）
> 实际是 [`Core/GameRuntimeManager.cs`](../Core/GameRuntimeManager.cs)，负责初始化与组装所有服务。

BepInEx 依赖（`BepInEx.dll` / `0Harmony.dll`）通过 `GameRuntime_v2.csproj` 从游戏目录
`$(GameRoot)\BepInEx\core` 引用；编译产物 `GameRuntime_v2.dll` 由 csproj 的构建后步骤
自动拷贝到 `$(GameRoot)\BepInEx\plugins`。

如需扩展插件装配逻辑，请编辑 `Core/GameRuntimeManager.cs`，而非在此目录新建入口。

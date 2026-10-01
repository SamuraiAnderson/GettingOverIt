<#
.SYNOPSIS
  用 dnlib 反汇编 .NET 程序集里指定类型的 IL（含字符串/方法/字段操作数），导出到文本文件。

.DESCRIPTION
  本机环境下 dnSpy.Console.exe 在非交互控制台会崩溃（Console.OutputEncoding 句柄无效），
  且 dnSpy 的 C# AST 反编译 API 依赖 dnSpy.Contracts 一堆内部类型，难以脚本化。
  纯 dnlib 的 IL dump 是最稳可用的路径：能看清 PlayerPrefs 键、方法调用链、字段读写，
  足以还原保存/载入、初始化等逻辑。IL 可读性够用（ldstr=字符串, call/callvirt=调用,
  stfld/ldfld=字段写/读, newarr/stelem=数组）。

.PARAMETER Assembly
  目标程序集路径（如 Assembly-CSharp.dll）。其所在目录会自动加入引用解析搜索路径。

.PARAMETER Types
  类型名通配模式数组（匹配 Name 或 FullName），如 'Saviour','*Save*','PlayerControl'。

.PARAMETER OutDir
  输出目录，默认 <assembly 目录>\_il_dump。每个类型写一个 IL_<Type>.txt，并生成 _all_types.txt。

.PARAMETER DnlibPath
  dnlib.dll 路径。默认取 dnSpy 自带的 bin\dnlib.dll。

.EXAMPLE
  pwsh -File dump_il.ps1 -Assembly "C:\...\Assembly-CSharp.dll" -Types Saviour,SaveState,PlayerControl
#>
param(
    [Parameter(Mandatory = $true)] [string]$Assembly,
    [Parameter(Mandatory = $true)] [string[]]$Types,
    [string]$OutDir,
    [string]$DnlibPath = 'C:\Users\Symbol\code_tool\dnSpy\bin\dnlib.dll'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Assembly))  { throw "找不到程序集: $Assembly" }
if (-not (Test-Path $DnlibPath)) { throw "找不到 dnlib.dll: $DnlibPath（改用 -DnlibPath 指定）" }

$managed = Split-Path -Parent $Assembly
if (-not $OutDir) { $OutDir = Join-Path $managed '_il_dump' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# 下载来的 DLL 常被 Windows 标记为已阻止（0x80131515），先解锁
Unblock-File -Path $DnlibPath -ErrorAction SilentlyContinue
[void][System.Reflection.Assembly]::LoadFrom($DnlibPath)

# 载入模块 + 程序集解析器（指向 assembly 所在目录，解析 UnityEngine 等引用）
$ctx = New-Object dnlib.DotNet.ModuleContext
$resolver = New-Object dnlib.DotNet.AssemblyResolver($ctx)
$ctx.AssemblyResolver = $resolver
$resolver.DefaultModuleContext = $ctx
[void]$resolver.PostSearchPaths.Add($managed)
$module = [dnlib.DotNet.ModuleDefMD]::Load($Assembly, $ctx)

$allTypes = @($module.GetTypes())
$allTypes | ForEach-Object { $_.FullName } | Sort-Object |
    Set-Content -Encoding UTF8 (Join-Path $OutDir '_all_types.txt')
Write-Output ("TYPE_COUNT=" + $allTypes.Count)

function Dump-One($td) {
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("// ===== TYPE: " + $td.FullName + " =====")
    [void]$sb.AppendLine("// BaseType: " + $(if ($td.BaseType) { $td.BaseType.FullName } else { '<none>' }))
    [void]$sb.AppendLine("// ---- Fields ----")
    foreach ($f in $td.Fields) { [void]$sb.AppendLine("  " + $f.FieldType.FullName + " " + $f.Name) }
    foreach ($m in $td.Methods) {
        [void]$sb.AppendLine("")
        $ps = ($m.Parameters | Where-Object { -not $_.IsHiddenThisParameter } |
               ForEach-Object { $_.Type.FullName + ' ' + $_.Name }) -join ', '
        [void]$sb.AppendLine("// ---- METHOD: " + $m.ReturnType.FullName + " " + $m.Name + "(" + $ps + ") ----")
        if ($m.HasBody) {
            foreach ($ins in $m.Body.Instructions) {
                $op = $ins.Operand
                $ostr = ''
                if ($null -ne $op) { $ostr = if ($op -is [string]) { '"' + $op + '"' } else { $op.ToString() } }
                [void]$sb.AppendLine(('    {0}: {1} {2}' -f $ins.Offset.ToString('X4'), $ins.OpCode.Name, $ostr))
            }
        } else { [void]$sb.AppendLine("    <no body>") }
    }
    $safe = ($td.FullName -replace '[^\w\.]', '_')
    $file = Join-Path $OutDir ("IL_" + $safe + '.txt')
    $sb.ToString() | Set-Content -Encoding UTF8 $file
    Write-Output ("OK  " + $td.FullName + " -> " + $file)
}

# 兜底：某些调用方式会把 "A,B,C" 作为单个元素传入，这里统一按逗号再拆一次
$patterns = @($Types | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim().Trim("'`"") } | Where-Object { $_ })

foreach ($pat in $patterns) {
    $matched = $allTypes | Where-Object { $_.Name -like $pat -or $_.FullName -like $pat }
    if (-not $matched) { Write-Output ("MISS  no type matches '" + $pat + "'") ; continue }
    foreach ($td in $matched) { Dump-One $td }
}
Write-Output 'DONE'

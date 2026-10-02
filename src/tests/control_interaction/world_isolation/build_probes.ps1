param([string]$GameDirectory = 'artifacts/reset_game')
$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$fixture = (Resolve-Path -LiteralPath $GameDirectory).Path
$outputDirectory = Join-Path $repository 'bin/WorldIsolationExperiment'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$references = @('UnityEngine.dll', 'UnityEngine.CoreModule.dll', 'UnityEngine.Physics2DModule.dll', 'UnityEngine.JSONSerializeModule.dll', 'UnityEngine.AudioModule.dll') |
    ForEach-Object { '/r:' + (Join-Path $fixture ('GettingOverIt_Data/Managed/' + $_)) }
foreach ($probe in @('CloneDivergenceProbe', 'CallbackIsolationProbe')) {
    $source = Join-Path $PSScriptRoot ($probe + '.cs')
    $outputFile = Join-Path $outputDirectory ($probe + '.dll')
    & C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe /nologo /target:library "/out:$outputFile" @references "/r:$fixture/BepInEx/core/BepInEx.dll" "/r:$fixture/BepInEx/core/0Harmony.dll" "/r:$fixture/BepInEx/plugins/GameRuntime_v2.dll" $source
    if ($LASTEXITCODE -ne 0) { throw "Failed to compile $probe" }
    Write-Output $outputFile
}

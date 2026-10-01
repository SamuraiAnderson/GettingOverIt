param([switch]$Train)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ckpt = Join-Path $root 'runs\confirmed_endpoint_v2_formal_20260928_S_s47\checkpoints\ppo_iter_0123.pt'
$pool = Join-Path $root 'runs\confirmed_endpoint_v2_formal_20260928_S_s47\elites\elites_iter_0123.npz'
$runDir = Join-Path $root 'runs\log_std_clamp_20260929'
$pyArgs = @(
    '-m', 'src.training.main_ppo',
    '--resume', $ckpt,
    '--sil-resume-pool', $pool,
    '--corridor-sil',
    '--sil-archive-mode', 'td_pareto',
    '--sil-td-sampling-mode', 'balanced_stratified',
    '--sil-td-windows-per-traj', '32',
    '--log-std-max', '-0.5',
    '--seed', '47',
    '--max-iterations', '143',
    '--additional-physical-samples', '192000',
    '--torch-threads', '4',
    '--save-interval', '1',
    '--run-dir', $runDir,
    '--persist-game',
    '--save-rollout-inputs'
)
if (-not $Train) { $pyArgs += '--dry-run' }
$python = 'C:\Users\Symbol\code_tool\miniconda\envs\getting-over-it-analysis\python.exe'
Push-Location $root
try {
    & $python @pyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

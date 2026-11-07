param([Parameter(Mandatory=$true)][string]$PythonPath)
$ErrorActionPreference = 'Stop'
$root = 'results/supervised/runs/expanded_128'
$dense = 'data/supervised/trajectories_t100_dense_128'
$long = 'data/supervised/trajectories_t1000_128'
$warm = 'results/supervised/models/rollout_warm.pt'
$jobs = @(
    @{ name = 't1000_nn_direct'; model = 'nn_direct'; data = $long; extra = @('--save-predictions') },
    @{ name = 't100_nn_direct'; model = 'nn_direct'; data = $dense; extra = @('--save-predictions') },
    @{ name = 't100_nn_block'; model = 'nn_block'; data = $dense; extra = @('--warm', $warm) },
    @{ name = 't100_nn_geometry'; model = 'nn_geometry'; data = $dense; extra = @('--warm', $warm) }
)
$failed = $false
foreach ($job in $jobs) {
    $name = $job.name
    $out = Join-Path $root $name
    $extra = $job.extra
    $log = Join-Path $root "$name.log"
    Add-Content -Path "$root/queue.log" -Value "$(Get-Date -Format s) start $name"
    & $PythonPath -u -m "src.$($job.model).train" --data $job.data --out $out --device cuda --steps 6000 --batch 8 --seed 20261106 --report-every 250 @extra *> $log
    $code = $LASTEXITCODE
    Add-Content -Path "$root/queue.log" -Value "$(Get-Date -Format s) exit $name code=$code"
    if ($code -ne 0) { $failed = $true }
}
if ($failed) { exit 1 }

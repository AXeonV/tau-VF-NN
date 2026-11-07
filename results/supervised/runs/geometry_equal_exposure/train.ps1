param([Parameter(Mandatory=$true)][string]$PythonPath)
$ErrorActionPreference = 'Stop'
$root = 'results/supervised/runs/geometry_equal_exposure'
$warm = 'results/supervised/models/rollout_warm.pt'
$arms = @(
    @{ name = '128x8400'; data = 'data/supervised/trajectories_t100_dense_128'; steps = 8400 },
    @{ name = '96x6000'; data = 'data/supervised/trajectories_t100_dense'; steps = 6000 }
)
$failed = $false
foreach ($seed in 20261106,20261107) {
    foreach ($arm in $arms) {
        $name = "$($arm.name)_$seed"
        $out = Join-Path $root $name
        $log = Join-Path $root "$name.log"
        Add-Content "$root/queue.log" "$(Get-Date -Format s) start $name"
        & $PythonPath -u -m src.nn_geometry.train --data $arm.data --out $out --device cuda --warm $warm --steps $arm.steps --batch 8 --seed $seed --report-every 300 *> $log
        $code = $LASTEXITCODE
        Add-Content "$root/queue.log" "$(Get-Date -Format s) exit $name code=$code"
        if ($code -ne 0) { $failed = $true }
    }
}
if ($failed) { exit 1 }

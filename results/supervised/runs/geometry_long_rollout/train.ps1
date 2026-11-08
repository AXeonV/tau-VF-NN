param([Parameter(Mandatory=$true)][string]$PythonPath)
$ErrorActionPreference = 'Stop'
$root = 'results/supervised/runs/geometry_long_rollout'
$data = 'data/supervised/trajectories_t100_dense_128'
$warm = 'results/supervised/models/rollout_warm.pt'
foreach ($seed in 20261106,20261107) {
    $out = "$root/rollout16_$seed"
    Add-Content "$root/queue.log" "$(Get-Date -Format s) start $seed"
    & $PythonPath -u -m src.nn_geometry.train --data $data --out $out --device cuda --warm $warm --steps 8400 --batch 8 --seed $seed --report-every 300 --long-rollout *> "$root/train_$seed.log"
    $code = $LASTEXITCODE
    Add-Content "$root/queue.log" "$(Get-Date -Format s) train $seed exit=$code"
    if ($code -ne 0) { exit $code }
    & $PythonPath -u -m utils.diagnose_rollout $out --output "$out/reset_diagnostics.json" *> "$out/diagnostics.log"
    $code = $LASTEXITCODE
    Add-Content "$root/queue.log" "$(Get-Date -Format s) diagnostics $seed exit=$code"
    if ($code -ne 0) { exit $code }
}

param([Parameter(Mandatory=$true)][string]$PythonPath)
$ErrorActionPreference = 'Stop'
$root = 'results/supervised/runs/geometry_initial_starts'
$data = 'data/supervised/trajectories_t100_dense_128'
$warm = 'results/supervised/models/rollout_warm.pt'
$deadline = (Get-Date).AddHours(4)
Add-Content "$root/queue.log" "$(Get-Date -Format s) waiting for CUDA"
do {
    & $PythonPath -c "import sys,torch; sys.exit(not torch.cuda.is_available())"
    if ($LASTEXITCODE -eq 0) { break }
    if ((Get-Date) -gt $deadline) {
        Add-Content "$root/queue.log" "$(Get-Date -Format s) CUDA still unavailable"
        exit 2
    }
    Start-Sleep -Seconds 60
} while ($true)
$runs = @()
foreach ($seed in 20261106,20261107) {
    $out = "$root/initial16_$seed"
    if (Test-Path "$out/history.npy") { throw "Existing training output in $out" }
    $runs += $out
    Add-Content "$root/queue.log" "$(Get-Date -Format s) start $seed"
    & $PythonPath -u -m src.nn_geometry.train --data $data --out $out --device cuda --warm $warm --steps 8400 --batch 8 --seed $seed --report-every 300 --long-rollout --initial-starts *> "$root/train_$seed.log"
    $code = $LASTEXITCODE
    Add-Content "$root/queue.log" "$(Get-Date -Format s) train $seed exit=$code"
    if ($code -ne 0) { exit $code }
    & $PythonPath -u -m utils.diagnose_rollout $out --output "$out/reset_diagnostics.json" *> "$out/diagnostics.log"
    $code = $LASTEXITCODE
    Add-Content "$root/queue.log" "$(Get-Date -Format s) reset diagnostics $seed exit=$code"
    if ($code -ne 0) { exit $code }
}
& $PythonPath -u -m utils.diagnose_tip_births @runs --device cuda --output "$root/tip_diagnostics.json" *> "$root/tip_diagnostics.log"
$code = $LASTEXITCODE
Add-Content "$root/queue.log" "$(Get-Date -Format s) birth diagnostics exit=$code"
exit $code

param([Parameter(Mandatory=$true)][string]$PythonPath)
$r='results/supervised/runs/tip_correction'
$deadline=(Get-Date).AddMinutes(30)
while(!(Test-Path "$r/drift_20261107/metrics.json")) {
    if ((Select-String -Path "$r/queue.log" -Pattern 'exit .*code=[1-9]' -Quiet) -or (Get-Date) -gt $deadline) { exit 2 }
    Start-Sleep -Seconds 15
}
foreach($seed in 20261106,20261107) {
    $name="weighted_$seed"
    $warm="results/supervised/runs/tip_density/paired_dense_$seed/model.pt"
    Add-Content "$r/queue.log" "$(Get-Date -Format s) start $name"
    & $PythonPath -u -m src.nn_geometry.tip_correction --warm $warm --out "$r/$name" --arm drift --steps 1200 --seed $seed --recovery-weight 16 *> "$r/$name.log"
    $code=$LASTEXITCODE
    Add-Content "$r/queue.log" "$(Get-Date -Format s) exit $name code=$code"
    if ($code -ne 0) { Get-Content "$r/$name.log" -Tail 12; exit $code }
}

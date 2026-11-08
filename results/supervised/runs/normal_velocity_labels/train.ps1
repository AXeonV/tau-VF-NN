param([Parameter(Mandatory=$true)][string]$PythonPath)
$r='results/supervised/runs/normal_velocity_labels'
foreach($seed in 20261106,20261107) {
    foreach($arm in 'control','velocity') {
        $name="${arm}_$seed"
        Add-Content "$r/queue.log" "$(Get-Date -Format s) start $name"
        & $PythonPath -u -m src.nn_geometry.train_velocity_labels --arm $arm --seed $seed --steps 1200 --out "$r/$name" *> "$r/$name.log"
        $code=$LASTEXITCODE
        Add-Content "$r/queue.log" "$(Get-Date -Format s) exit $name code=$code"
        if ($code -ne 0) { Get-Content "$r/$name.log" -Tail 16; exit $code }
    }
}
$checkpoints=@()
foreach($seed in 20261106,20261107) {
    $checkpoints += "results/supervised/runs/tip_density/paired_dense_$seed/model.pt"
    $checkpoints += "$r/control_$seed/model.pt"
    $checkpoints += "$r/velocity_$seed/model.pt"
}
Add-Content "$r/queue.log" "$(Get-Date -Format s) start evaluation"
& $PythonPath -u -m utils.evaluate_velocity_labels @checkpoints --out "$r/evaluation.json" *> "$r/evaluation.log"
$code=$LASTEXITCODE
Add-Content "$r/queue.log" "$(Get-Date -Format s) exit evaluation code=$code"
if ($code -ne 0) { Get-Content "$r/evaluation.log" -Tail 16; exit $code }

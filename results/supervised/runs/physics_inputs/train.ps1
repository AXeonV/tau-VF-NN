param([Parameter(Mandatory=$true)][string]$PythonPath)
$r='results/supervised/runs/physics_inputs'
$arms='geometry_control','neighbors','history','combined'
$checkpoints=@()
foreach($seed in 20261106,20261107) {
    $checkpoints += "results/supervised/runs/normal_velocity_labels/velocity_$seed/model.pt"
    foreach($arm in $arms) {
        $name="${arm}_$seed"
        Add-Content "$r/queue.log" "$(Get-Date -Format s) start $name"
        & $PythonPath -u -m src.nn_geometry.train_physics_inputs --variant $arm --seed $seed --steps 1200 --out "$r/$name" *> "$r/$name.log"
        $code=$LASTEXITCODE
        Add-Content "$r/queue.log" "$(Get-Date -Format s) exit $name code=$code"
        if ($code -ne 0) { Get-Content "$r/$name.log" -Tail 16; exit $code }
        $checkpoints += "$r/$name/model.pt"
    }
}
Add-Content "$r/queue.log" "$(Get-Date -Format s) start evaluation"
& $PythonPath -u -m utils.evaluate_velocity_labels @checkpoints --out "$r/evaluation.json" *> "$r/evaluation.log"
$code=$LASTEXITCODE
Add-Content "$r/queue.log" "$(Get-Date -Format s) exit evaluation code=$code"
if ($code -ne 0) { Get-Content "$r/evaluation.log" -Tail 16; exit $code }
& $PythonPath -u -m utils.compare_physics_inputs "$r/evaluation.json" --out "$r/comparison.json" *> "$r/comparison.log"
$code=$LASTEXITCODE
Add-Content "$r/queue.log" "$(Get-Date -Format s) exit comparison code=$code"
Get-Content "$r/comparison.log"
exit $code

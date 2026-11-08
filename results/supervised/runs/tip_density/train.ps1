param([Parameter(Mandatory=$true)][string]$PythonPath)
$r='results/supervised/runs/tip_density'
foreach($seed in 20261106,20261107) {
    $warm="results/supervised/runs/geometry_initial_starts/initial16_$seed/model.pt"
    foreach($arm in 'sparse','dense') {
        $name="${arm}_$seed"
        Add-Content "$r/queue.log" "$(Get-Date -Format s) start $name"
        & $PythonPath -u -m src.nn_geometry.tip_density --warm $warm --out "$r/$name" --labels $arm --steps 1600 --seed $seed *> "$r/$name.log"
        $code=$LASTEXITCODE
        Add-Content "$r/queue.log" "$(Get-Date -Format s) exit $name code=$code"
        if ($code -ne 0) { Get-Content "$r/$name.log" -Tail 12; exit $code }
    }
}

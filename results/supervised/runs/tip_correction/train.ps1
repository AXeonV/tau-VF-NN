param([Parameter(Mandatory=$true)][string]$PythonPath)
$r='results/supervised/runs/tip_correction'
foreach($seed in 20261106,20261107) {
    $warm="results/supervised/runs/tip_density/paired_dense_$seed/model.pt"
    foreach($arm in 'clean','drift') {
        $name="${arm}_$seed"
        Add-Content "$r/queue.log" "$(Get-Date -Format s) start $name"
        & $PythonPath -u -m src.nn_geometry.tip_correction --warm $warm --out "$r/$name" --arm $arm --steps 1200 --seed $seed *> "$r/$name.log"
        $code=$LASTEXITCODE
        Add-Content "$r/queue.log" "$(Get-Date -Format s) exit $name code=$code"
        if ($code -ne 0) { Get-Content "$r/$name.log" -Tail 12; exit $code }
    }
}

param([Parameter(Mandatory=$true)][string]$PythonPath)
$r='results/supervised/runs/drift_diagnostic'
foreach($seed in 20261106,20261107) {
    foreach($architecture in 'baseline','normal_speed') {
        $name="probe_${architecture}_$seed"
        Add-Content "$r/probe_queue.log" "$(Get-Date -Format s) start $name"
        & $PythonPath -u -m src.nn_geometry.normal_speed_probe --architecture $architecture --seed $seed --steps 800 --out "$r/$name" *> "$r/$name.log"
        $code=$LASTEXITCODE
        Add-Content "$r/probe_queue.log" "$(Get-Date -Format s) exit $name code=$code"
        if ($code -ne 0) { Get-Content "$r/$name.log" -Tail 15; exit $code }
    }
}

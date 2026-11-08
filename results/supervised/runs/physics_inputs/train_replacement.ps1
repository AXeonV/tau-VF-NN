param([Parameter(Mandatory=$true)][string]$PythonPath)
$r='results/supervised/runs/physics_inputs'
$limit=(Get-Date).AddHours(2)
while (!(Select-String -Path "$r/queue.log" -Pattern 'exit comparison code=0' -Quiet)) {
    if ((Select-String -Path "$r/queue.log" -Pattern 'exit .*code=[1-9]' -Quiet) -or (Get-Date) -gt $limit) { exit 2 }
    Start-Sleep -Seconds 15
}
$checkpoints=@()
foreach($seed in 20261106,20261107) {
    $name="history_replace_$seed"
    Add-Content "$r/replacement_queue.log" "$(Get-Date -Format s) start $name"
    & $PythonPath -u -m src.nn_geometry.train_physics_inputs --variant history_replace --seed $seed --steps 1200 --out "$r/$name" *> "$r/$name.log"
    $code=$LASTEXITCODE
    Add-Content "$r/replacement_queue.log" "$(Get-Date -Format s) exit $name code=$code"
    if ($code -ne 0) { Get-Content "$r/$name.log" -Tail 16; exit $code }
    $checkpoints += "$r/$name/model.pt"
}
& $PythonPath -u -m utils.evaluate_velocity_labels @checkpoints --out "$r/replacement_evaluation.json" *> "$r/replacement_evaluation.log"
$code=$LASTEXITCODE
Add-Content "$r/replacement_queue.log" "$(Get-Date -Format s) exit evaluation code=$code"
if ($code -ne 0) { Get-Content "$r/replacement_evaluation.log" -Tail 16; exit $code }
@"
from pathlib import Path
import json
from src.trajectory_metrics import atomic_json
folder=Path('results/supervised/runs/physics_inputs')
original=json.loads((folder/'evaluation.json').read_text())
replacement=json.loads((folder/'replacement_evaluation.json').read_text())
assert original['cases']==replacement['cases'] and original['labels']==replacement['labels']
original['runs']+=replacement['runs']
atomic_json(folder/'all_evaluation.json',original)
"@ | & $PythonPath -u -
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
& $PythonPath -u -m utils.compare_physics_inputs "$r/all_evaluation.json" --out "$r/comparison.json" *> "$r/comparison.log"
$code=$LASTEXITCODE
Add-Content "$r/replacement_queue.log" "$(Get-Date -Format s) exit comparison code=$code"
Get-Content "$r/comparison.log"
exit $code

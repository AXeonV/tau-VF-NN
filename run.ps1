param([string]$PythonPath = 'python')
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $PythonPath -m src.traditional.evolution --T 100 --n 512 --dt .04 --save-every .25 --out results/_initialization/t100
    if ($LASTEXITCODE -ne 0) { throw 'Initialization failed.' }
    & $PythonPath -m src.traditional.evolution --T 500 --n 768 --dt .1 --save-every 1 --restart results/_initialization/t100/initialization.npz --out results/_initialization/t500
    if ($LASTEXITCODE -ne 0) { throw 'Continuation failed.' }
    & $PythonPath -m src.traditional.evolution --T 1000 --n 1024 --dt .15 --save-every 2 --restart results/_initialization/t500/initialization.npz --out results/ellipse_long
    if ($LASTEXITCODE -ne 0) { throw 'Long initialization failed.' }
    & $PythonPath -m src.traditional.optimize --case results/ellipse_long
    if ($LASTEXITCODE -ne 0) { throw 'Optimization failed.' }
    & $PythonPath -m utils.render --case results/ellipse_long
    if ($LASTEXITCODE -ne 0) { throw 'Rendering failed.' }
    $temporaryPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'results/_initialization'))
    $resultsPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'results'))
    if (-not $temporaryPath.StartsWith($resultsPath + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected temporary directory.' }
    if (Test-Path -LiteralPath $temporaryPath) { Remove-Item -LiteralPath $temporaryPath -Recurse -Force }
} finally { Pop-Location }

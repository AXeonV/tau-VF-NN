param([string]$PythonPath = 'python')
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $modules = @('src.traditional.generate', 'src.nn.train', 'src.nn.predict', 'utils.render')
    foreach ($module in $modules) {
        & $PythonPath -m $module
        if ($LASTEXITCODE -ne 0) { throw "Module failed: $module" }
    }
} finally {
    Pop-Location
}

$ErrorActionPreference = "Stop"

$tpDir = Join-Path $PSScriptRoot "Trabajos Practicos\TP Integrador"
$venvDir = Join-Path $tpDir ".venv"
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"

if (-not (Test-Path $tpDir)) {
    Write-Error "No se encontro la carpeta del TP Integrador: $tpDir"
    exit 1
}

Set-Location $tpDir

if (-not (Test-Path $activateScript)) {
    Write-Host "Creando entorno virtual en $venvDir..."
    try {
        py -3.12 -m venv $venvDir
    }
    catch {
        Write-Host "No se pudo crear con py -3.12, probando con python..."
        python -m venv $venvDir
    }
}

if (-not (Test-Path $activateScript)) {
    Write-Error "No se pudo crear el entorno virtual. Instala Python 3.10, 3.11 o 3.12."
    exit 1
}

. $activateScript

python -m pip install --upgrade pip
pip install -r requirements.txt
python tp_integrador_gui.py

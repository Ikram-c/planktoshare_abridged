# =============================================================================
#  install.ps1 — one-click setup for plankton-preprocess (Windows)
#  Run from the repo root in PowerShell:
#      Set-ExecutionPolicy -Scope Process Bypass; .\install.ps1
#  Optional flag:
#      .\install.ps1 -ForceCuda 12.1   # pin a CUDA version instead of auto
# =============================================================================
[CmdletBinding()]
param(
    [string]$ForceCuda = ""   # e.g. "11.8", "12.1", "12.4"
)

$ErrorActionPreference = "Stop"

# ── colours ──────────────────────────────────────────────────────────────────
function Info    { param($m) Write-Host "[INFO]  $m"  -ForegroundColor Cyan    }
function Success { param($m) Write-Host "[OK]    $m"  -ForegroundColor Green   }
function Warn    { param($m) Write-Host "[WARN]  $m"  -ForegroundColor Yellow  }
function Fail    { param($m) Write-Host "[ERROR] $m"  -ForegroundColor Red; exit 1 }

# ── require PowerShell 5.1+ ──────────────────────────────────────────────────
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Fail "PowerShell 5.1 or later is required. Current: $($PSVersionTable.PSVersion)"
}

Info "Platform: Windows ($env:PROCESSOR_ARCHITECTURE)"

# ── Python version check ─────────────────────────────────────────────────────
$pythonBin = $null
foreach ($candidate in @("python3.13","python3.12","python3.11","python3","python")) {
    try {
        $ver = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -match '^(\d+)\.(\d+)$') {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            if ($maj -eq 3 -and $min -ge 11) { $pythonBin = $candidate; break }
        }
    } catch {}
}

if (-not $pythonBin) {
    Warn "Python 3.11+ not found on PATH."
    Info "Attempting install via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        # Refresh PATH so the new python is visible
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path","User")
        $pythonBin = "python"
    } else {
        Fail "winget not available. Install Python 3.11+ from https://python.org and re-run."
    }
}
$pyVer = & $pythonBin --version
Success "Using Python: $pyVer"

# ── uv installation ──────────────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Info "Installing uv..."
    $uvInstallScript = (Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -UseBasicParsing).Content
    Invoke-Expression $uvInstallScript

    # Refresh PATH so uv is found immediately
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User") + ";" +
                "$env:USERPROFILE\.local\bin" + ";" +
                "$env:USERPROFILE\.cargo\bin"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail "uv installed but not found in PATH.`nAdd %USERPROFILE%\.local\bin to your PATH and re-run."
    }
}
Success "uv $(uv --version)"

# ── virtual environment ──────────────────────────────────────────────────────
$venvDir = ".venv"
if (Test-Path $venvDir) {
    Warn "Existing .venv found — reusing it. Delete it first to start fresh."
} else {
    Info "Creating virtual environment..."
    uv venv $venvDir --python $pythonBin
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Fail ".venv created but python.exe not found at expected path: $venvPython"
}

# ── main dependencies ─────────────────────────────────────────────────────────
Info "Installing project dependencies from pyproject.toml..."
uv pip install --python $venvPython -e ".[ome-models]"
Success "Core dependencies installed."

# ── PyTorch ──────────────────────────────────────────────────────────────────
Info "Determining PyTorch variant..."

$torchIndex = "https://download.pytorch.org/whl/cpu"
$torchLabel = "cpu (no GPU detected)"

if ($ForceCuda -ne "") {
    $cudaTag    = "cu" + $ForceCuda.Replace(".", "")
    $torchIndex = "https://download.pytorch.org/whl/$cudaTag"
    $torchLabel = "CUDA $ForceCuda (forced)"

} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $smiOut = & nvidia-smi 2>&1
        if ($smiOut -match "CUDA Version:\s*([\d.]+)") {
            $cudaRaw   = $Matches[1]
            $parts     = $cudaRaw.Split(".")
            $cudaMajor = [int]$parts[0]
            $cudaMinor = [int]$parts[1]

            if     ($cudaMajor -ge 12 -and $cudaMinor -ge 4) { $cudaTag = "cu124" }
            elseif ($cudaMajor -ge 12 -and $cudaMinor -ge 1) { $cudaTag = "cu121" }
            elseif ($cudaMajor -ge 11 -and $cudaMinor -ge 8) { $cudaTag = "cu118" }
            else {
                Warn "CUDA $cudaRaw is older than 11.8 — installing CPU-only torch."
                $cudaTag = "cpu"
            }

            $torchIndex = "https://download.pytorch.org/whl/$cudaTag"
            $torchLabel = "CUDA $cudaRaw → wheel $cudaTag"
        } else {
            Warn "nvidia-smi found but CUDA version unreadable — falling back to CPU."
        }
    } catch {
        Warn "nvidia-smi check failed — falling back to CPU torch."
    }
}

Info "PyTorch variant: $torchLabel"
uv pip install --python $venvPython torch torchvision --index-url $torchIndex
Success "PyTorch installed."

# ── smoke test ────────────────────────────────────────────────────────────────
Info "Running smoke test..."
& $venvPython -c @"
import numpy, zarr, numcodecs, skimage, yaml, torch, ome_zarr, pandas, opencv-contrib-python, pyclesperanto-prototype
print(f'  numpy        {numpy.__version__}')
print(f'  zarr         {zarr.__version__}')
print(f'  scikit-image {skimage.__version__}')
print(f'  torch        {torch.__version__}')
cuda = 'OK' if torch.cuda.is_available() else '-'
print(f'  CUDA={cuda}')
"@
if ($LASTEXITCODE -ne 0) { Fail "Smoke test failed." }
Success "Smoke test passed."

# ── activation hint ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Setup complete. Activate the environment with:" -ForegroundColor White
Write-Host "  .venv\Scripts\activate" -ForegroundColor Cyan
Write-Host ""
Write-Host "Then run the pipeline:"
Write-Host "  python run_preproccess_pipeline.py config.yaml" -ForegroundColor Cyan
Write-Host "  python run_preproccess_pipeline.py config.yaml --profile" -ForegroundColor Cyan
#!/usr/bin/env bash
# =============================================================================
#  install.sh — one-click setup for plankton-preprocess
#  Supports: Ubuntu / WSL2, macOS (Intel + Apple Silicon)
#  Usage:  bash install.sh [--cuda 12.1]
#           --cuda <ver>   Force a specific CUDA version (e.g. 11.8, 12.1, 12.4)
#                          Default: auto-detect via nvidia-smi; CPU fallback.
# =============================================================================
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

# ── args ─────────────────────────────────────────────────────────────────────
FORCE_CUDA=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda) FORCE_CUDA="$2"; shift 2 ;;
        *)      error "Unknown argument: $1" ;;
    esac
done

# ── platform detection ───────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        if grep -qEi "(microsoft|wsl)" /proc/version 2>/dev/null; then
            PLATFORM="wsl"
        else
            PLATFORM="linux"
        fi
        ;;
    Darwin)
        PLATFORM="macos"
        ;;
    *)
        error "Unsupported OS: $OS"
        ;;
esac

info "Platform: ${BOLD}${PLATFORM}${RESET} (${ARCH})"

# ── Python version check ─────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR="${VER%%.*}"; MINOR="${VER##*.}"
        if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 11 ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    warn "Python 3.11+ not found."
    if [[ "$PLATFORM" == "macos" ]]; then
        info "Installing Python via Homebrew..."
        if ! command -v brew &>/dev/null; then
            error "Homebrew not found. Install it first: https://brew.sh"
        fi
        brew install python@3.12
        PYTHON_BIN="python3.12"
    else
        info "Installing Python 3.12 via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
        PYTHON_BIN="python3.12"
    fi
fi
success "Using Python: $($PYTHON_BIN --version)"

# ── uv installation ──────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for the remainder of this script
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        error "uv installation succeeded but binary not found in PATH.\n  Add ~/.local/bin or ~/.cargo/bin to your shell profile and re-run."
    fi
fi
success "uv $(uv --version)"

# ── virtual environment ──────────────────────────────────────────────────────
VENV_DIR=".venv"
if [[ -d "$VENV_DIR" ]]; then
    warn "Existing .venv found — reusing it. Delete it first to start fresh."
else
    info "Creating virtual environment with $PYTHON_BIN..."
    uv venv "$VENV_DIR" --python "$PYTHON_BIN"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_UV="uv pip install --python $VENV_PYTHON"

# ── main dependencies ─────────────────────────────────────────────────────────
info "Installing project dependencies from pyproject.toml..."
uv pip install --python "$VENV_PYTHON" -e ".[ome-models]"
success "Core dependencies installed."

# ── PyTorch ──────────────────────────────────────────────────────────────────
info "Determining PyTorch variant..."

if [[ "$PLATFORM" == "macos" ]]; then
    # macOS: always CPU+MPS (torch picks MPS automatically on Apple Silicon)
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_LABEL="cpu (MPS available on Apple Silicon at runtime)"

elif [[ -n "$FORCE_CUDA" ]]; then
    # User explicitly requested a CUDA version
    CUDA_TAG="cu$(echo "$FORCE_CUDA" | tr -d '.')"
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
    TORCH_LABEL="CUDA ${FORCE_CUDA} (forced)"

elif command -v nvidia-smi &>/dev/null; then
    # Auto-detect CUDA version from nvidia-smi
    RAW=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)
    if [[ -z "$RAW" ]]; then
        warn "nvidia-smi found but could not read CUDA version — falling back to CPU."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_LABEL="cpu (fallback)"
    else
        CUDA_MAJOR="${RAW%%.*}"
        CUDA_MINOR="${RAW#*.}"; CUDA_MINOR="${CUDA_MINOR%%.*}"
        # Map driver CUDA version to the nearest supported PyTorch wheel
        if   [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 4 ]]; then CUDA_TAG="cu124"
        elif [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 1 ]]; then CUDA_TAG="cu121"
        elif [[ "$CUDA_MAJOR" -ge 11 && "$CUDA_MINOR" -ge 8 ]]; then CUDA_TAG="cu118"
        else
            warn "CUDA ${RAW} is older than 11.8 — installing CPU-only torch."
            CUDA_TAG="cpu"
        fi
        TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
        TORCH_LABEL="CUDA ${RAW} → wheel ${CUDA_TAG}"
    fi
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_LABEL="cpu (no GPU detected)"
fi

info "PyTorch variant: ${BOLD}${TORCH_LABEL}${RESET}"
$VENV_UV torch torchvision --index-url "$TORCH_INDEX"
success "PyTorch installed."

# ── smoke test ────────────────────────────────────────────────────────────────
info "Running smoke test..."
"$VENV_PYTHON" - <<'EOF'
import numpy, zarr, numcodecs, skimage, yaml, torch, ome_zarr
print(f"  numpy       {numpy.__version__}")
print(f"  zarr        {zarr.__version__}")
print(f"  scikit-image {skimage.__version__}")
print(f"  torch       {torch.__version__}")
cuda = "✅" if torch.cuda.is_available() else "—"
mps  = "✅" if torch.backends.mps.is_available() else "—"
print(f"  CUDA={cuda}  MPS={mps}")
EOF
success "Smoke test passed."

# ── activation hint ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Setup complete.${RESET} Activate the environment with:"
echo -e "  ${CYAN}source .venv/bin/activate${RESET}"
echo ""
echo "Then run the pipeline:"
echo -e "  ${CYAN}python run_preproccess_pipeline.py config.yaml${RESET}"
echo -e "  ${CYAN}python run_preproccess_pipeline.py config.yaml --profile${RESET}"
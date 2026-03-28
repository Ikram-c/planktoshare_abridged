# Installation Guide

---

## Quick Install (Recommended)

The easiest way to get started is to use the one-click installer included with the software.

### Windows

Double-click the file named `install.bat`.

### Mac and Linux

Before you can run the installer, you need to give your computer permission to open it. Open your Terminal app and run this command:

```bash
chmod +x ./install.sh
```

Then double-click the file named `install.sh`.

---

## Manual Installation

If the quick installer did not work, you can set everything up yourself by following the steps below. You will be copying and pasting a few commands into your terminal.

**What is a terminal?**
It is a text-based window where you type instructions directly to your computer.
- **Mac / Linux:** Search for an app called **Terminal**.
- **Windows:** Open the Start menu, search for **PowerShell**, and open it.

Navigate to the folder where you downloaded this project before continuing.

---

### Step 1 — Install the Package Manager

This installs a tool called `uv` that handles the rest of the setup for you — including downloading the correct version of Python automatically.

- **Mac / Linux** — paste this into Terminal and press Enter:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **Windows** — paste this into PowerShell and press Enter:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> **Important:** Once this finishes, close your terminal window and open a fresh one before moving to the next step. This lets your computer recognise the new tool you just installed.

---

### Step 2 — Set Up the Project Environment

This creates a self-contained space on your computer for the software to live in, so it does not interfere with anything else.

Paste this command and press Enter:

```bash
uv venv --python 3.12 .venv
```

Next, activate the environment. **You will need to repeat this activation command each time you open a new terminal and want to use the software.**

- **Mac / Linux:**

```bash
source .venv/bin/activate
```

- **Windows:**

```powershell
.venv\Scripts\activate
```

---

### Step 3 — Install the Software

Paste this command and press Enter:

```bash
uv pip install -e ".[ome-models]"
```

---

### Step 4 — Install the AI Engine

This installs PyTorch, which powers the AI parts of the software. Choose the command that matches your computer:

- **Mac (any model):**

```bash
uv pip install torch torchvision
```

- **Windows or Linux — standard PC (no Nvidia graphics card):**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- **Windows or Linux — with an Nvidia graphics card:**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Not sure which you have? If you have never heard of an Nvidia GPU, choose the **standard PC** option.

---

### Step 5 — Test the Installation

Everything should now be installed. Run the following command to confirm everything is working:

```bash
python run_preproccess_pipeline.py config.yaml
```

If it runs without errors, you are all set!

---

---

## Developer Notes

This section is intended for developers and technically curious users who want to understand what the setup process is doing under the hood.

### Python Toolchain Management

Installing `uv` first removes the need for a pre-existing system-wide Python installation. The command:

```bash
uv venv --python 3.12 .venv
```

causes `uv` to check whether Python 3.12 is available and, if not, automatically fetch it via `uv python install` before creating the virtual environment.

### Virtual Environment Isolation

The `.venv` directory contains a fully isolated Python environment, preventing dependency conflicts with other projects or system packages.

### Editable Install and Extras

The command:

```bash
uv pip install -e ".[ome-models]"
```

installs the project in editable mode (`-e`), meaning changes to the source are reflected immediately without reinstalling. The `[ome-models]` extra pulls in the additional dependencies declared under that key in `pyproject.toml`.

### PyTorch Hardware Acceleration

- **macOS:** The standard PyTorch CPU wheel is used. On Apple Silicon hardware, PyTorch automatically routes computation through the MPS (Metal Performance Shaders) backend at runtime — no separate install is needed.
- **CPU fallback:** Pointing explicitly to the `whl/cpu` index avoids downloading large CUDA binaries on machines that cannot use them.
- **CUDA (Nvidia):** The `cu121` index targets CUDA 12.1, which offers broad compatibility across modern Nvidia GPUs on Windows and Linux.

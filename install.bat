@echo off
:: Thin launcher so users can double-click instead of opening PowerShell manually.
:: Bypasses execution policy for this process only — does NOT change system policy.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Installation failed. See output above.
    pause
    exit /b %ERRORLEVEL%
)
pause
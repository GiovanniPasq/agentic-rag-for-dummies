@echo off
setlocal enabledelayedexpansion

:: Navigate to repository root
cd /d "%~dp0"

echo ==========================================
echo   Starting Agentic RAG System
echo ==========================================

:: 1. Check/Prepare environment file
if not exist "project\.env" (
    if exist "project\.env.example" (
        echo Creating project\.env from project\.env.example...
        copy "project\.env.example" "project\.env" >nul
    )
)

:: 2. Informative check for Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama does not seem to be running on localhost:11434.
    echo           If using local models (default), remember to start Ollama.
)

:: 3. Launch application (prefer uv if available)
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo Using uv for fast environment management...
    uv run project\app.py
    goto :end
)

echo uv not detected, falling back to standard Python venv...

:: Detect Python executable (try py, then python)
set "PYTHON_CMD="
where py >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=python"
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python 3.11+ is required but neither py nor python was found in PATH.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment in .venv...
    %PYTHON_CMD% -m venv .venv
)

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Checking and installing dependencies...
pip install -q -r requirements.txt

echo Starting app...
python project\app.py

:end
endlocal

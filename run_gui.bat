@echo off
setlocal

:: Change to script directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\pythonw.exe" (
    echo Virtual environment not found!
    echo.
    echo Please run setup_venv.bat first to create the virtual environment.
    echo.
    pause
    exit /b 1
)

:: Run GUI silently (no console window)
start "" /B "venv\Scripts\pythonw.exe" run.py

endlocal

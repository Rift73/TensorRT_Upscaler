@echo off
setlocal

:: Change to script directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found!
    echo.
    echo Please run setup_venv.bat first to create the virtual environment.
    echo.
    pause
    exit /b 1
)

:: Activate venv and run CLI with all arguments passed through
call "venv\Scripts\activate.bat"
python -m tensorrt_upscaler.cli %*

endlocal

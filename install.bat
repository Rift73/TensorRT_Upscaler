@echo off
setlocal enabledelayedexpansion

title TensorRT Upscaler v2 - Installer

echo ===============================================
echo  TensorRT Upscaler v2 - One-Click Installer
echo ===============================================
echo.

:: Change to script directory
cd /d "%~dp0"

:: Find Python 3.13 or 3.12
set PYTHON_EXE=

:: Check common Python 3.13 locations
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe" (
    set PYTHON_EXE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe
    goto :found
)
if exist "C:\Python313\python.exe" (
    set PYTHON_EXE=C:\Python313\python.exe
    goto :found
)

:: Check common Python 3.12 locations
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON_EXE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
    goto :found
)
if exist "C:\Python312\python.exe" (
    set PYTHON_EXE=C:\Python312\python.exe
    goto :found
)

:: Try py launcher
py -3.13 --version >nul 2>&1 && (
    set PYTHON_EXE=py -3.13
    goto :found
)
py -3.12 --version >nul 2>&1 && (
    set PYTHON_EXE=py -3.12
    goto :found
)

:: Not found
echo ERROR: Python 3.13 or 3.12 not found!
echo.
echo Please install Python 3.13 or 3.12 from:
echo   https://www.python.org/downloads/
echo.
pause
exit /b 1

:found
echo [+] Found Python: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

:: Create venv if needed
if not exist venv (
    echo [+] Creating virtual environment...
    %PYTHON_EXE% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [+] Virtual environment already exists
)

echo.
echo [+] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [+] Upgrading pip...
python -m pip install --upgrade pip -q

echo.
echo [+] Installing core dependencies...
pip install numpy Pillow PySide6 cuda-python -q

echo.
echo [+] Installing performance dependencies...
pip install opencv-python numba fpng-py -q

echo.
echo [+] Installing package in development mode...
pip install -e . -q

echo.
echo ===============================================
echo  Installation Complete!
echo ===============================================
echo.
echo To run the GUI:
echo   - Double-click run_gui.bat
echo   - Or: venv\Scripts\python run.py
echo.
echo To use the CLI:
echo   - run_cli.bat [arguments]
echo   - Or: venv\Scripts\python -m tensorrt_upscaler.cli --help
echo.
echo NOTE: TensorRT must be installed separately:
echo   pip install tensorrt
echo.

:: Ask to create desktop shortcut
choice /c YN /m "Create desktop shortcut"
if errorlevel 2 goto :skip_shortcut

:: Create VBS script to make shortcut
echo [+] Creating desktop shortcut...
set SHORTCUT_VBS=%TEMP%\create_shortcut.vbs
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%SHORTCUT_VBS%"
echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\TensorRT Upscaler.lnk" >> "%SHORTCUT_VBS%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%SHORTCUT_VBS%"
echo oLink.TargetPath = "%~dp0run_gui.bat" >> "%SHORTCUT_VBS%"
echo oLink.WorkingDirectory = "%~dp0" >> "%SHORTCUT_VBS%"
echo oLink.Description = "TensorRT Image Upscaler" >> "%SHORTCUT_VBS%"
if exist "%~dp0icon.ico" (
    echo oLink.IconLocation = "%~dp0icon.ico" >> "%SHORTCUT_VBS%"
)
echo oLink.Save >> "%SHORTCUT_VBS%"
cscript //nologo "%SHORTCUT_VBS%"
del "%SHORTCUT_VBS%"
echo    Shortcut created on desktop!

:skip_shortcut
echo.
pause

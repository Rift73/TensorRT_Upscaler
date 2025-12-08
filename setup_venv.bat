@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo  TensorRT Upscaler v2 - Virtual Environment Setup
echo ===============================================
echo.

:: Try to find Python 3.13 first, then 3.12
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

:: Try python3.13 and python3.12 from PATH
where python3.13 >nul 2>&1 && (
    set PYTHON_EXE=python3.13
    goto :found
)
where python3.12 >nul 2>&1 && (
    set PYTHON_EXE=python3.12
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
echo Or set PYTHON_EXE environment variable to your Python executable.
pause
exit /b 1

:found
echo Found Python: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

:: Check if venv exists
if exist venv (
    echo Virtual environment already exists.
    choice /c YN /m "Delete and recreate"
    if errorlevel 2 goto :activate_only
    echo.
    echo Removing old venv...
    rmdir /s /q venv
)

echo.
echo Creating virtual environment...
%PYTHON_EXE% -m venv venv

if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

:activate_only
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo ===============================================
echo  Installing Dependencies
echo ===============================================
echo.

echo Installing core dependencies...
pip install numpy>=1.24.0 Pillow>=10.0.0 PySide6>=6.5.0

echo.
echo Installing performance dependencies...
pip install opencv-python>=4.8.0 numba>=0.58.0 fpng-py>=0.0.4

echo.
echo Installing CUDA/TensorRT dependencies...
pip install cuda-python

echo.
echo Installing CUDA/TensorRT dependencies...
pip install pycuda

echo.
echo Installing TensorRT...
pip install tensorrt
echo.

echo ===============================================
echo  Setup Complete!
echo ===============================================
echo.
echo To activate the virtual environment:
echo     venv\Scripts\activate.bat
echo.
echo To run the GUI:
echo     python run.py
echo.
echo To install TensorRT (if not already installed):
echo     pip install tensorrt
echo.
pause

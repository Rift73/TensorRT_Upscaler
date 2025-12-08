@echo off
setlocal

title TensorRT Upscaler v2 - Update Dependencies

echo ===============================================
echo  TensorRT Upscaler v2 - Update Dependencies
echo ===============================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo [+] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [+] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [+] Updating all dependencies...
pip install --upgrade numpy Pillow PySide6 cuda-python opencv-python numba fpng-py

echo.
echo [+] Reinstalling package...
pip install -e . --upgrade

echo.
echo ===============================================
echo  Update Complete!
echo ===============================================
echo.
pause

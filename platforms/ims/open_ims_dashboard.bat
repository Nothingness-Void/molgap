@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=0
pushd "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
    echo [Error] Could not find .venv\Scripts\python.exe in %CD%
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -S platforms\ims\check_queue.py --web --open %*
if errorlevel 1 (
    echo.
    echo [IMS Dashboard exited with error]
    pause
)
popd

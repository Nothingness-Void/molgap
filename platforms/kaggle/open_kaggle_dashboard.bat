@echo off
setlocal
cd /d "%~dp0..\.."
set PYTHONUTF8=0

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" platforms\kaggle\dashboard.py live --port 8766 --open %*
) else (
  python platforms\kaggle\dashboard.py live --port 8766 --open %*
)

@echo off
cd /d "%~dp0"
title ChatGPT Image Bridge 7960

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" -c "import playwright" >nul 2>&1
if errorlevel 1 (
  echo [bridge] Installing playwright ...
  "%PY%" -m pip install playwright
)

echo.
echo ChatGPT image bridge on http://127.0.0.1:7960
echo First run: log in to ChatGPT in the Chrome window that opens.
echo Keep this window open while using it.
echo.
"%PY%" -u -m bridges.chatgpt_image.server
pause

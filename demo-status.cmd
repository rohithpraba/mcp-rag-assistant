@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo.ps1" status
if errorlevel 1 pause
endlocal

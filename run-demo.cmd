@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo.ps1" start -Public
if errorlevel 1 (
  echo.
  echo Demo startup failed. Review the message above.
  pause
)
endlocal

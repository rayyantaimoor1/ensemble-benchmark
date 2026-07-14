@echo off
cd /d "%~dp0"

set PYEXE=%USERPROFILE%\anaconda3\envs\ensemble-bench\python.exe

echo Using Python: %PYEXE%
echo.

"%PYEXE%" -m streamlit run app.py

pause
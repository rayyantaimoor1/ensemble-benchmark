@echo off
REM ============================================================
REM  run_dashboard.bat
REM  Double-click this file to launch the Streamlit dashboard.
REM  Must be located in the project root (same folder as app.py).
REM ============================================================

cd /d "%~dp0"

echo ============================================
echo  Ensemble Learning Benchmark - Dashboard
echo ============================================
echo.

REM --- Activate a virtual environment if one exists ---
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating .venv...
    call .venv\Scripts\activate.bat
) else (
    echo No venv found - using system/conda Python on PATH.
)

REM --- Check Python is available ---
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not found on PATH.
    echo Install Python, or activate your conda environment first,
    echo then run this file again.
    echo.
    pause
    exit /b 1
)

REM --- Check Streamlit is installed; install requirements if not ---
python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Streamlit not found - installing requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install requirements. Check your internet
        echo connection and Python/pip installation, then try again.
        echo.
        pause
        exit /b 1
    )
)

REM --- Launch the dashboard ---
echo.
echo Starting dashboard at http://localhost:8501
echo Close this window to stop the server.
echo.
streamlit run app.py

pause

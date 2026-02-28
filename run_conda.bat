@echo off
set ENV_NAME=data_inspector_py311
set PY_VER=3.11

echo ==========================================================
echo [Conda Setup] Creating environment: %ENV_NAME% (Python %PY_VER%)
echo ==========================================================

:: Create conda environment
call conda create -n %ENV_NAME% python=%PY_VER% -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda environment creation failed.
    pause
    exit /b %ERRORLEVEL%
)

:: Install requirements if file exists
if exist requirements.txt (
    echo.
    echo [Pip Install] Installing dependencies from requirements.txt...
    call conda run -n %ENV_NAME% pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Pip installation failed.
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo.
    echo [SKIP] requirements.txt not found. Skipping package installation.
)

echo.
echo ==========================================================
echo [SUCCESS] Environment '%ENV_NAME%' is ready.
echo To activate: conda activate %ENV_NAME%
echo ==========================================================
pause

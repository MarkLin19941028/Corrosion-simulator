@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================================
echo   Corrosion Simulator - Windows PyInstaller Build Script
echo ============================================================

set "APP_NAME=CorrosionSimulator"

if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo [1/3] Prebuilding Parquet tables and all 6 ML models from data\raw ...
"%PY_EXE%" -m corrosion_simulator.build_artifacts --models all --no-csv
if errorlevel 1 (
    echo [ERROR] Failed to build artifacts from data\raw. Aborting.
    exit /b 1
)

echo [2/3] Running PyInstaller (onedir mode, external data excluded from _internal) ...
"%PY_EXE%" -m PyInstaller -D --noconfirm --clean ^
    --name "%APP_NAME%" ^
    --add-data "app.py;." ^
    --collect-submodules corrosion_simulator ^
    --collect-all streamlit ^
    --collect-all plotly ^
    --collect-all pyarrow ^
    --collect-all sklearn ^
    --collect-all matplotlib ^
    --collect-submodules scipy ^
    --collect-submodules joblib ^
    --copy-metadata streamlit ^
    --copy-metadata plotly ^
    --copy-metadata scikit-learn ^
    --copy-metadata pyarrow ^
    main.py
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed. Make sure pyinstaller is installed: "%PY_EXE%" -m pip install pyinstaller
    exit /b 1
)

echo [3/3] Copying external data\ folder next to %APP_NAME%.exe ...
set "DIST_DATA=dist\%APP_NAME%\data"
if exist "%DIST_DATA%" rmdir /s /q "%DIST_DATA%"
mkdir "%DIST_DATA%\raw"
mkdir "%DIST_DATA%\processed\models"

if exist "data\raw" (
    xcopy "data\raw\*" "%DIST_DATA%\raw\" /E /I /Y >nul
)
if exist "data\processed" (
    for %%F in ("data\processed\*.parquet" "data\processed\manifest.json" "data\processed\rejected_rows.csv") do (
        if exist "%%~F" copy /Y "%%~F" "%DIST_DATA%\processed\" >nul
    )
    if exist "data\processed\models" (
        xcopy "data\processed\models\*.joblib" "%DIST_DATA%\processed\models\" /Y >nul
    )
)

echo ============================================================
echo [SUCCESS] Build completed!
echo   Executable : dist\%APP_NAME%\%APP_NAME%.exe
echo   Raw CSVs   : dist\%APP_NAME%\data\raw\
echo   Model Cache: dist\%APP_NAME%\data\processed\
echo   Zip the entire "dist\%APP_NAME%" folder to distribute.
echo ============================================================
endlocal

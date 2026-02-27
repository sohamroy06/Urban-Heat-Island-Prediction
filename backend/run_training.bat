@echo off
echo Running ShadowMap training pipeline...
python -u "%~dp0train_model.py"
if errorlevel 1 (
    echo.
    echo ERROR: Training failed. Make sure Python and required packages are installed.
    echo Run: pip install -r requirements.txt
    pause
) else (
    echo.
    echo Training complete!
    pause
)

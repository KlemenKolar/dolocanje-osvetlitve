@echo off
REM Activate the virtual environment
call venv\Scripts\activate

REM Start the MLflow UI
start mlflow ui

REM Wait a few seconds to ensure MLflow UI is up and running
timeout /t 3 /nobreak

REM Open Chrome at port 5000
start chrome http://localhost:5000
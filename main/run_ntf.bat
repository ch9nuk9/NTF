@echo off
REM Activate conda environment
call C:\ProgramData\Anaconda3\Scripts\activate.bat ntf_env

REM Run the Python script
python "%~dp0ntf_app.py"

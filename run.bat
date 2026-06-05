@echo off

cd /d "%~dp0"



REM Hentikan instance Streamlit lama (jika ada)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1



REM Bersihkan cache Python agar modul terbaru dimuat

for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"



call venv\Scripts\activate.bat

streamlit run app.py


@echo off
chcp 65001 >nul
echo Testing Auto Data Collection System
echo ====================================

:: Check required files
echo Checking system files...

if not exist "src\Python\auto_data_collector.py" (
    echo ERROR: Python script not found
    exit /b 1
)

if not exist "src\Scripts\auto_data_collection.bat" (
    echo ERROR: Main script not found
    exit /b 1
)

echo OK: All required files exist

:: Test Python script
echo Testing Python script...
call conda activate getting-over-it-analysis
python src\Python\auto_data_collector.py --help

if %ERRORLEVEL% EQU 0 (
    echo OK: Python script help test passed
) else (
    echo ERROR: Python script help test failed
    exit /b 1
)

:: Test database creation
echo Testing database creation...
python -c "
import sys
sys.path.append('src/Python')
from auto_data_collector import AutoDataCollector
collector = AutoDataCollector('test_db.db')
collector.setup_database()
print('OK: Database creation test passed')
"

if %ERRORLEVEL% EQU 0 (
    echo OK: Database test passed
) else (
    echo ERROR: Database test failed
    exit /b 1
)

:: Clean up test database
del test_db.db 2>nul

echo.
echo Auto Data Collection System Test Complete!
echo.
echo Usage:
echo    1. Run auto_data_collection.bat
echo    2. Wait for game to start
echo    3. Select 'Restart'
echo    4. Press any key to start collection
echo.
pause

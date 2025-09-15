@echo off
setlocal enabledelayedexpansion

echo 🚀 Getting Over It 自动化数据采集系统
echo ========================================

:: 配置参数
set GAME_PATH="C:\Users\Symbol\software\game_store\steam\steamapps\common\Getting Over It\GettingOverIt.exe"
set PYTHON_SCRIPT="src\Python\auto_data_collector_v2.py"
set COLLECTION_DURATION=5
set COLLECTION_FREQUENCY=30

:: 检查游戏文件是否存在
if not exist %GAME_PATH% (
    echo ❌ 游戏文件不存在: %GAME_PATH%
    pause
    exit /b 1
)

:: 检查Python脚本是否存在
if not exist %PYTHON_SCRIPT% (
    echo ❌ Python脚本不存在: %PYTHON_SCRIPT%
    pause
    exit /b 1
)

echo 📁 游戏路径: %GAME_PATH%
echo 🐍 Python脚本: %PYTHON_SCRIPT%
echo ⏱️ 采集时长: %COLLECTION_DURATION%秒
echo 📊 采集频率: %COLLECTION_FREQUENCY%Hz
echo.

:: 启动游戏
echo 🎮 启动游戏...
start "" %GAME_PATH%

:: 等待游戏启动
echo ⏳ 等待游戏启动 (10秒)...
timeout /t 10 /nobreak >nul

:: 检查游戏进程是否运行
:check_process
tasklist /FI "IMAGENAME eq GettingOverIt.exe" 2>NUL | find /I /N "GettingOverIt.exe" >nul
if "%ERRORLEVEL%"=="0" (
    echo ✅ 游戏进程已启动
) else (
    echo ⏳ 等待游戏进程启动...
    timeout /t 2 /nobreak >nul
    goto check_process
)

:: 等待游戏完全加载
echo ⏳ 等待游戏完全加载 (15秒)...
timeout /t 15 /nobreak >nul

:: 提示用户操作
echo.
echo 🤖 自动化模式已启用:
echo    1. Unity插件将自动选择"重新开始"
echo    2. 自动等待游戏场景加载完成
echo    3. 自动开始数据采集
echo.
echo 按任意键开始自动化数据采集...
pause

:: 启动Python数据采集脚本
echo 🐍 启动Python数据采集脚本...
echo 📊 开始采集数据: %COLLECTION_DURATION%秒 @ %COLLECTION_FREQUENCY%Hz

:: 激活conda环境并运行Python脚本
call conda activate getting-over-it-analysis
python %PYTHON_SCRIPT% --duration %COLLECTION_DURATION% --frequency %COLLECTION_FREQUENCY%

:: 检查Python脚本执行结果
if %ERRORLEVEL% EQU 0 (
    echo ✅ 数据采集完成
) else (
    echo ❌ 数据采集失败
    pause
    exit /b 1
)

echo.
echo 🎉 自动化数据采集完成!
echo 📁 数据已保存到数据库
echo.

:: 询问是否关闭游戏
set /p close_game="是否关闭游戏? (y/n): "
if /i "%close_game%"=="y" (
    echo 🎮 关闭游戏...
    taskkill /F /IM GettingOverIt.exe >nul 2>&1
    echo ✅ 游戏已关闭
)

echo.
echo 📋 采集统计:
echo    - 采集时长: %COLLECTION_DURATION%秒
echo    - 采集频率: %COLLECTION_FREQUENCY%Hz
echo    - 预计数据点: %COLLECTION_DURATION% * %COLLECTION_FREQUENCY% = %COLLECTION_DURATION% * %COLLECTION_FREQUENCY%
echo.

pause

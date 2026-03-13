@echo off
chcp 65001 > nul
echo ========================================
echo 剂量-响应验证测试 - 快速启动
echo ========================================
echo.

cd /d "%~dp0"

echo 激活conda环境...
call conda activate getting-over-it-analysis

echo.
echo 选择操作:
echo   1. 生成测试文件
echo   2. 运行所有测试（半自动）
echo   3. 分析测试结果
echo.

set /p choice="请输入选项 (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo 生成测试文件...
    python dose_response_test.py
) else if "%choice%"=="2" (
    echo.
    echo 开始自动测试...
    python run_dose_response_tests.py
) else if "%choice%"=="3" (
    echo.
    echo 分析测试结果...
    python dose_response_test.py --analyze
) else (
    echo 无效选项
)

echo.
pause


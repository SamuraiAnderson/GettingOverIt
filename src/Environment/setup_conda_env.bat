@echo off
echo 🐍 === Getting Over It 数据分析环境设置 ===
echo.

echo 📦 创建conda环境...
conda env create -f environment.yml

echo.
echo ✅ 环境创建完成！
echo.
echo 💡 使用方法:
echo    conda activate getting-over-it-analysis
echo    python quick_analysis.py
echo    python analyze_tracking_data.py
echo.
echo 🔄 如需重新创建环境:
echo    conda env remove -n getting-over-it-analysis
echo    conda env create -f environment.yml
echo.
pause

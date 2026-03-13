"""
自动运行精密精度测试 - 无需交互
"""

import sys
import io
import subprocess
import shutil
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def run_single_test(value):
    """运行单个测试"""
    game_exe = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GettingOverIt.exe")
    game_data = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    result_folder = game_data / "GameResults"
    
    print(f"\n{'='*60}")
    print(f"测试输入值: {value}")
    print(f"{'='*60}")
    
    # 准备输入文件
    input_test_file = game_data / f"precision_{int(value)}.csv"
    input_command_file = game_data / "input_commands.csv"
    
    if not input_test_file.exists():
        print(f"  ❌ 找不到测试文件: {input_test_file}")
        return False
    
    print(f"  1. 复制测试文件...")
    shutil.copy(input_test_file, input_command_file)
    print(f"     ✓ precision_{int(value)}.csv → input_commands.csv")
    
    # 启动游戏
    print(f"  2. 启动游戏...")
    try:
        process = subprocess.Popen([str(game_exe)])
        print(f"     ✓ 游戏已启动 (PID: {process.pid})")
        print(f"     ⏳ 等待自动完成...")
        
        process.wait()
        print(f"     ✓ 游戏已关闭")
        time.sleep(2)
        
    except Exception as e:
        print(f"     ❌ 错误: {e}")
        return False
    
    # 保存结果
    result_latest = result_folder / "ContinuousTracking_latest.csv"
    result_saved = result_folder / f"precision_{int(value)}.csv"
    
    print(f"  3. 保存结果...")
    if result_latest.exists():
        shutil.copy(result_latest, result_saved)
        print(f"     ✓ precision_{int(value)}.csv")
        return True
    else:
        print(f"     ❌ 找不到结果文件")
        return False


def main():
    """主函数"""
    test_values = [49, 50, 51, 52]
    
    print("\n" + "🔬"*30)
    print("自动运行精密精度测试")
    print("🔬"*30)
    print(f"\n测试序列: {test_values}")
    print(f"预计耗时: 约 {len(test_values) * 0.5:.1f} 分钟")
    print("\n" + "="*60)
    
    start_time = time.time()
    success_count = 0
    
    for i, value in enumerate(test_values, 1):
        print(f"\n[{i}/{len(test_values)}] 开始测试...")
        
        if run_single_test(value):
            success_count += 1
            
            if i < len(test_values):
                print(f"\n⏸️  等待2秒...")
                time.sleep(2)
        else:
            print(f"\n⚠️  测试失败，继续下一个...")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("测试完成总结")
    print("="*60)
    print(f"\n  成功: {success_count}/{len(test_values)}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    
    if success_count == len(test_values):
        print("\n✅ 所有测试完成！")
        print("\n下一步: 运行分析")
        print("  cd src/tests")
        print("  python analyze_precision_50.py")
    else:
        print(f"\n⚠️  部分测试失败")


if __name__ == '__main__':
    main()


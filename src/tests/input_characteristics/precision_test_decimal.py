"""
小数精度测试
测试序列: 50.0, 50.1, 50.2, 50.5, 51.0
目标: 验证系统能否区分小数级别的输入差异
"""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import shutil
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def generate_test_input(value, duration_sec=3, fps=60):
    """生成测试输入"""
    n_frames = int(duration_sec * fps)
    commands = []
    for i in range(n_frames):
        commands.append({
            'timestamp': i / fps,
            'mouseXdelta': value,
            'mouseYdelta': 0
        })
    return pd.DataFrame(commands)


def generate_decimal_tests():
    """生成小数精度测试文件"""
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    
    # 小数测试序列
    test_values = [50.0, 50.1, 50.2, 50.5, 51.0]
    
    print("="*60)
    print("小数精度测试 - 生成测试文件")
    print("="*60)
    print(f"\n测试基准: 50")
    print(f"测试序列: {test_values}")
    print(f"间隔: 0.1, 0.1, 0.3, 0.5")
    print(f"\n目标: 验证系统能否区分小数差异")
    print("-"*60)
    
    for value in test_values:
        df = generate_test_input(value, duration_sec=3)
        # 使用特殊命名避免覆盖
        filename = f"decimal_{int(value*10):03d}.csv"  # 50.0->500, 50.1->501
        df.to_csv(game_path / filename, index=False)
        print(f"  ✓ {filename:20s} (输入值={value:.1f}, 共{len(df)}帧)")
    
    print(f"\n✅ 已生成 {len(test_values)} 个小数测试文件")
    return test_values


def run_single_test(value):
    """运行单个测试"""
    game_exe = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GettingOverIt.exe")
    game_data = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    result_folder = game_data / "GameResults"
    
    print(f"\n{'='*60}")
    print(f"测试输入值: {value:.1f}")
    print(f"{'='*60}")
    
    # 准备输入文件
    input_test_file = game_data / f"decimal_{int(value*10):03d}.csv"
    input_command_file = game_data / "input_commands.csv"
    
    if not input_test_file.exists():
        print(f"  ❌ 找不到测试文件: {input_test_file}")
        return False
    
    print(f"  1. 复制测试文件...")
    shutil.copy(input_test_file, input_command_file)
    print(f"     ✓ decimal_{int(value*10):03d}.csv → input_commands.csv")
    
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
    result_saved = result_folder / f"decimal_{int(value*10):03d}.csv"
    
    print(f"  3. 保存结果...")
    if result_latest.exists():
        shutil.copy(result_latest, result_saved)
        print(f"     ✓ decimal_{int(value*10):03d}.csv")
        return True
    else:
        print(f"     ❌ 找不到结果文件")
        return False


def main():
    """主函数"""
    test_values = [50.0, 50.1, 50.2, 50.5, 51.0]
    
    print("\n" + "🔬"*30)
    print("小数精度测试")
    print("🔬"*30)
    print(f"\n测试序列: {test_values}")
    print(f"预计耗时: 约 {len(test_values) * 0.5:.1f} 分钟")
    print("\n" + "="*60)
    
    # 先生成测试文件
    print("\n步骤1: 生成测试文件")
    generate_decimal_tests()
    
    print("\n步骤2: 运行测试")
    print("="*60)
    
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
        print("  python analyze_decimal_precision.py")
    else:
        print(f"\n⚠️  部分测试失败")


if __name__ == '__main__':
    main()


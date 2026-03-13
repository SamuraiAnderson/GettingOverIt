"""
精细化测试：80-100区间
目标：精确定位饱和点
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


def run_single_test(value):
    """运行单个测试"""
    game_exe = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GettingOverIt.exe")
    game_data = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    result_folder = game_data / "GameResults"
    
    print(f"\n{'='*60}")
    print(f"测试输入值: {value}")
    print(f"{'='*60}")
    
    # 生成并保存输入文件
    df = generate_test_input(value, duration_sec=3)
    input_file = game_data / "input_commands.csv"
    df.to_csv(input_file, index=False)
    print(f"  1. 已生成输入文件")
    
    # 运行游戏
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
    result_saved = result_folder / f"maxtest_{int(value):04d}.csv"
    
    print(f"  3. 保存结果...")
    if result_latest.exists():
        shutil.copy(result_latest, result_saved)
        print(f"     ✓ maxtest_{int(value):04d}.csv")
        return True
    else:
        print(f"     ❌ 找不到结果文件")
        return False


def analyze_immediate_response(csv_path):
    """分析第2帧响应"""
    df = pd.read_csv(csv_path)
    
    key_fields = ['hammerAngle', 'sliderAngle', 'handleX', 'handleY', 'tipX', 'tipY',
                  'handleVelX', 'handleVelY', 'tipVelX', 'tipVelY', 'hubSliderAngle']
    
    total_change = 0
    field_changes = {}
    
    for field in key_fields:
        if field in df.columns and len(df) > 2:
            change = abs(df[field].iloc[2] - df[field].iloc[0])
            total_change += change
            field_changes[field] = change
    
    return {
        'total_change': total_change,
        'fields': field_changes
    }


def main():
    """主函数：测试80-100区间"""
    # 精细测试序列
    test_values = [85, 90, 95]
    
    print("\n" + "🔬"*30)
    print("精细化测试：80-100区间")
    print("🔬"*30)
    print(f"\n测试序列: {test_values}")
    print(f"目标: 精确定位饱和点")
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
            print(f"\n⚠️  测试失败")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("测试完成总结")
    print("="*60)
    print(f"\n  成功: {success_count}/{len(test_values)}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    
    if success_count >= 1:
        print("\n✅ 开始分析结果...")
        time.sleep(1)
        
        # 分析所有数据（包括之前的80和100）
        result_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData/GameResults")
        all_values = [80, 85, 90, 95, 100]
        
        print("\n" + "="*60)
        print("完整区间分析 [80-100]")
        print("="*60)
        
        results = {}
        for value in all_values:
            csv_file = result_path / f"maxtest_{int(value):04d}.csv"
            if csv_file.exists():
                analysis = analyze_immediate_response(csv_file)
                results[value] = analysis['total_change']
                print(f"✓ 输入值 {value:3d}: 第2帧总变化 = {analysis['total_change']:10.6f}")
        
        # 找到跳跃点
        print("\n" + "="*60)
        print("饱和点定位")
        print("="*60)
        
        sorted_values = sorted(results.keys())
        for i in range(1, len(sorted_values)):
            value = sorted_values[i]
            prev_value = sorted_values[i-1]
            
            response = results[value]
            prev_response = results[prev_value]
            
            growth = response - prev_response
            growth_rate = (growth / prev_response * 100) if prev_response > 0 else 0
            
            print(f"\n{prev_value} → {value}:")
            print(f"  响应: {prev_response:.6f} → {response:.6f}")
            print(f"  增长: {growth:+.6f} ({growth_rate:+.1f}%)")
            
            if growth_rate > 50:
                print(f"  🎯 关键跳跃点")
            elif growth_rate < 5:
                print(f"  ✅ 已饱和")
        
        # 最终结论
        max_response = max(results.values())
        saturation_value = [v for v, r in results.items() if r >= max_response * 0.95][0]
        
        print("\n" + "="*60)
        print("🎯 精确结论")
        print("="*60)
        print(f"\n饱和点: {saturation_value}")
        print(f"最大响应: {max_response:.6f}")
        print(f"\n推荐配置:")
        print(f"  action_space = Box(low=-{saturation_value}, high={saturation_value})")
        
    else:
        print(f"\n⚠️  测试数据不足，无法分析")


if __name__ == '__main__':
    main()


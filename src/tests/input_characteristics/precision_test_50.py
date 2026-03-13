"""
精密精度测试 - 50附近区间
测试序列: 49, 50, 51, 52
目标: 确定真实的输入精度（1单位还是更大）
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


def generate_precision_tests():
    """生成精密测试文件"""
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    
    # 精密测试序列
    test_values = [49, 50, 51, 52]
    
    print("="*60)
    print("精密精度测试 - 生成测试文件")
    print("="*60)
    print(f"\n测试区间: 50附近")
    print(f"测试序列: {test_values}")
    print(f"间隔: 1单位")
    print(f"\n目标: 验证系统能否区分1单位的输入差异")
    print("-"*60)
    
    for value in test_values:
        df = generate_test_input(value, duration_sec=3)
        filename = f"precision_{int(value)}.csv"
        df.to_csv(game_path / filename, index=False)
        print(f"  ✓ {filename:20s} (输入值={value}, 共{len(df)}帧)")
    
    print(f"\n✅ 已生成 {len(test_values)} 个精密测试文件")
    return test_values


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


def analyze_precision_results(test_values):
    """分析精密测试结果"""
    result_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData/GameResults")
    
    print("\n" + "="*60)
    print("精密精度分析")
    print("="*60)
    
    # 收集所有测试的第2帧数据
    results = {}
    
    for value in test_values:
        csv_file = result_path / f"precision_{int(value)}.csv"
        
        if not csv_file.exists():
            print(f"⚠️  跳过: 输入值 {value} (文件不存在)")
            continue
        
        df = pd.read_csv(csv_file)
        
        # 计算第2帧相对第0帧的变化
        key_fields = ['hammerAngle', 'sliderAngle', 'handleX', 'tipX', 
                      'handleVelX', 'tipVelX', 'hubSliderAngle']
        
        frame2_change = 0
        field_changes = {}
        
        for field in key_fields:
            if field in df.columns and len(df) > 2:
                change = abs(df[field].iloc[2] - df[field].iloc[0])
                frame2_change += change
                field_changes[field] = change
        
        results[value] = {
            'total_change': frame2_change,
            'fields': field_changes
        }
        
        print(f"✓ 输入值 {value}: 第2帧总变化 = {frame2_change:10.4f}")
    
    # 分析连续值的可区分性
    print("\n" + "="*60)
    print("连续输入可区分性分析（核心）")
    print("="*60)
    
    threshold = 0.01  # 可区分阈值
    distinguishable_pairs = []
    indistinguishable_pairs = []
    
    sorted_values = sorted(results.keys())
    
    for i in range(len(sorted_values) - 1):
        val1 = sorted_values[i]
        val2 = sorted_values[i + 1]
        
        change1 = results[val1]['total_change']
        change2 = results[val2]['total_change']
        diff = abs(change2 - change1)
        
        if diff > threshold:
            status = "✓ 可区分"
            distinguishable_pairs.append((val1, val2, diff))
        else:
            status = "✗ 不可区分"
            indistinguishable_pairs.append((val1, val2, diff))
        
        print(f"\n输入 {val1} vs {val2} (差异=1单位):")
        print(f"  第2帧变化: {change1:8.4f} vs {change2:8.4f}")
        print(f"  效果差异: {diff:8.4f}")
        print(f"  → {status}")
    
    # 最终结论
    print("\n" + "="*60)
    print("🎯 精度验证结论")
    print("="*60)
    
    if len(distinguishable_pairs) == len(sorted_values) - 1:
        print("\n✅ 所有连续输入对都可区分")
        print("   → 真实精度 ≤ 1")
        print("   → 系统能分辨1单位的输入差异")
        print("\n推荐配置:")
        print("   action_space = Box(low=-80, high=80, dtype=np.float32)")
        print("   最小动作单位: 1.0")
        
    elif len(indistinguishable_pairs) == len(sorted_values) - 1:
        print("\n❌ 所有连续输入对都不可区分")
        print("   → 真实精度 > 1")
        print("   → 需要测试更大的间隔（如2, 5, 10）")
        
    else:
        print(f"\n⚠️  部分可区分 ({len(distinguishable_pairs)}/{len(sorted_values)-1})")
        print(f"   可区分对: {distinguishable_pairs}")
        print(f"   不可区分对: {indistinguishable_pairs}")
        print("\n   → 精度可能是非均匀的")
        print("   → 或需要更多测试确认")
    
    # 保存结果
    import json
    from datetime import datetime
    
    report = {
        'test_date': datetime.now().isoformat(),
        'test_region': '50附近',
        'test_values': test_values,
        'results': {int(k): v['total_change'] for k, v in results.items()},
        'distinguishable_pairs': len(distinguishable_pairs),
        'total_pairs': len(sorted_values) - 1,
        'conclusion': {
            'precision_verified': len(distinguishable_pairs) == len(sorted_values) - 1,
            'estimated_precision': 1 if len(distinguishable_pairs) == len(sorted_values) - 1 else '>1'
        }
    }
    
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    with open(game_path / "precision_test_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存: precision_test_report.json")


def main():
    """主函数"""
    print("\n" + "🔬"*30)
    print("精密精度测试 - 50附近区间")
    print("🔬"*30)
    
    # 生成测试文件
    test_values = generate_precision_tests()
    
    print("\n" + "="*60)
    print("准备开始测试")
    print("="*60)
    print(f"测试数量: {len(test_values)}")
    print(f"预计耗时: 约 {len(test_values) * 0.5:.1f} 分钟")
    
    confirm = input("\n开始测试？(y/n): ").lower()
    if confirm != 'y':
        print("已取消")
        return
    
    # 运行所有测试
    success_count = 0
    for value in test_values:
        if run_single_test(value):
            success_count += 1
            if value != test_values[-1]:
                print(f"\n⏸️  等待2秒...")
                time.sleep(2)
    
    print("\n" + "="*60)
    print(f"测试完成: {success_count}/{len(test_values)}")
    print("="*60)
    
    if success_count == len(test_values):
        print("\n✅ 所有测试完成！开始分析...")
        time.sleep(1)
        analyze_precision_results(test_values)
    else:
        print(f"\n⚠️  部分测试失败")


if __name__ == '__main__':
    main()


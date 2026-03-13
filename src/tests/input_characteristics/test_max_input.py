"""
输入最大值测试
目标：
1. 找到输入的绝对最大值（系统限制）
2. 精确标定饱和点（效果不再显著增长）

测试策略：
- 测试大值：100, 150, 200, 300, 500, 1000
- 分析第2帧响应
- 找到饱和点和系统限制
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


def analyze_max_input_results(test_values):
    """分析最大值测试结果"""
    result_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData/GameResults")
    
    print("\n" + "="*60)
    print("最大值测试分析")
    print("="*60)
    print(f"\n测试序列: {test_values}")
    
    # 收集所有测试结果
    results = {}
    
    print("\n" + "-"*60)
    print("数据收集")
    print("-"*60)
    
    for value in test_values:
        csv_file = result_path / f"maxtest_{int(value):04d}.csv"
        
        if not csv_file.exists():
            print(f"⚠️  跳过: 输入值 {value} (文件不存在)")
            continue
        
        analysis = analyze_immediate_response(csv_file)
        results[value] = analysis
        
        print(f"✓ 输入值 {value:6.0f}: 第2帧总变化 = {analysis['total_change']:10.6f}")
    
    if len(results) < 2:
        print("\n❌ 数据不足")
        return
    
    # 分析增长率
    print("\n" + "="*60)
    print("增长率分析（寻找饱和点）")
    print("="*60)
    
    sorted_values = sorted(results.keys())
    
    print(f"\n{'输入值':<10} {'第2帧响应':<15} {'vs前值增长':<15} {'增长率':<10}")
    print("-"*60)
    
    for i, value in enumerate(sorted_values):
        response = results[value]['total_change']
        
        if i == 0:
            print(f"{value:<10.0f} {response:<15.6f} {'-':<15} {'-':<10}")
        else:
            prev_value = sorted_values[i-1]
            prev_response = results[prev_value]['total_change']
            
            growth = response - prev_response
            growth_rate = (growth / prev_response * 100) if prev_response > 0 else 0
            
            status = ""
            if growth_rate < 5:
                status = "⚠️ 饱和"
            elif growth_rate < 10:
                status = "⚠️ 接近饱和"
            
            print(f"{value:<10.0f} {response:<15.6f} {growth:+14.6f} {growth_rate:>8.1f}% {status}")
    
    # 找到饱和点
    print("\n" + "="*60)
    print("饱和点分析")
    print("="*60)
    
    saturation_point = None
    max_growth_rate = 0
    
    for i in range(1, len(sorted_values)):
        value = sorted_values[i]
        prev_value = sorted_values[i-1]
        
        response = results[value]['total_change']
        prev_response = results[prev_value]['total_change']
        
        growth_rate = (response - prev_response) / prev_response * 100 if prev_response > 0 else 0
        
        if i == 1:
            max_growth_rate = growth_rate
        
        # 饱和定义：增长率降到最大增长率的20%以下
        if saturation_point is None and growth_rate < max_growth_rate * 0.2:
            saturation_point = prev_value
            print(f"\n饱和点: {saturation_point:.0f}")
            print(f"  - 在输入值{prev_value:.0f}之后，增长率从{max_growth_rate:.1f}%降至{growth_rate:.1f}%")
            print(f"  - 继续增加输入的边际收益显著降低")
    
    # 找到最大有效值
    print("\n" + "="*60)
    print("最大有效值分析")
    print("="*60)
    
    max_response = max(r['total_change'] for r in results.values())
    max_value = [v for v, r in results.items() if r['total_change'] == max_response][0]
    
    print(f"\n最大响应值: {max_response:.6f}")
    print(f"对应输入值: {max_value:.0f}")
    
    # 检查是否还在增长
    last_value = sorted_values[-1]
    last_growth = (results[last_value]['total_change'] - results[sorted_values[-2]]['total_change']) / results[sorted_values[-2]]['total_change'] * 100
    
    if last_growth > 5:
        print(f"\n⚠️ 最后一个测试值({last_value})仍在增长({last_growth:.1f}%)")
        print(f"   建议测试更大的值")
        recommended_max = None
    else:
        print(f"\n✅ 响应已趋于稳定")
        print(f"   推荐最大有效输入: {max_value:.0f}")
        recommended_max = max_value
    
    # 最终建议
    print("\n" + "="*60)
    print("🎯 最终结论")
    print("="*60)
    
    print(f"\n有效输入范围建议:")
    print(f"  - 最小有效值: 10 (已知)")
    print(f"  - 饱和点: {saturation_point if saturation_point else '> ' + str(sorted_values[-2])}")
    print(f"  - 最大有效值: {recommended_max if recommended_max else '> ' + str(last_value)}")
    print(f"\n推荐配置:")
    
    if saturation_point:
        print(f"  action_space = Box(low=-{saturation_point:.0f}, high={saturation_point:.0f})")
    else:
        print(f"  需要更多测试确定")
    
    # 保存报告
    import json
    from datetime import datetime
    
    report = {
        'test_date': datetime.now().isoformat(),
        'test_type': 'max_input',
        'test_values': test_values,
        'results': {
            str(k): {'total_change': float(v['total_change'])}
            for k, v in results.items()
        },
        'saturation_point': float(saturation_point) if saturation_point else None,
        'recommended_max': float(recommended_max) if recommended_max else None,
        'conclusion': {
            'min_effective': 10,
            'saturation_point': saturation_point,
            'max_effective': recommended_max
        }
    }
    
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    with open(game_path / "max_input_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存: max_input_report.json")


def main():
    """主函数"""
    # 测试序列：从已知的80开始，逐步增大
    test_values = [80, 100, 150, 200, 300, 500]
    
    print("\n" + "🔬"*30)
    print("输入最大值测试")
    print("🔬"*30)
    print(f"\n测试序列: {test_values}")
    print(f"目标: 找到饱和点和最大有效值")
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
    
    if success_count >= 2:
        print("\n✅ 开始分析结果...")
        time.sleep(1)
        analyze_max_input_results(test_values)
    else:
        print(f"\n⚠️  测试数据不足，无法分析")


if __name__ == '__main__':
    main()


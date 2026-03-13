"""
剂量-响应验证测试
验证输入精度是否真的是10，还是只是最小有效值

测试策略：
1. 测试多个输入值：5, 10, 15, 20, 25, 30, 50, 80, 100
2. 计算每个输入的"总效果强度"
3. 验证是否存在单调递增的剂量-响应关系
4. 如果单调递增 → 精度验证通过
5. 如果出现平台期 → 说明精度不是10，可能是更大的量化单位
"""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

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


def calculate_total_effect(csv_path):
    """
    计算总效果强度（所有关键字段的累计变化）
    
    Returns:
        dict: 包含总效果、各项指标和详细信息
    """
    df = pd.read_csv(csv_path)
    
    # 关键字段及其权重
    fields = {
        'hammerAngle': 1.0,
        'sliderAngle': 1.0,
        'hubSliderAngle': 1.0,
        'handleX': 10.0,
        'handleY': 10.0,
        'tipX': 10.0,
        'tipY': 10.0,
        'poleX': 10.0,
        'poleY': 10.0,
        'handleVelX': 5.0,
        'handleVelY': 5.0,
        'tipVelX': 5.0,
        'tipVelY': 5.0,
    }
    
    total_change = 0
    total_activity = 0
    field_count = 0
    
    details = {}
    
    for field, weight in fields.items():
        if field in df.columns:
            change = abs(df[field].iloc[-1] - df[field].iloc[0])
            std = df[field].std()
            
            weighted_change = change * weight
            weighted_activity = std * weight
            
            total_change += weighted_change
            total_activity += weighted_activity
            field_count += 1
            
            details[field] = {
                'change': change,
                'std': std,
                'weighted_change': weighted_change,
                'weighted_activity': weighted_activity
            }
    
    return {
        'total_effect': total_change + total_activity,
        'total_change': total_change,
        'total_activity': total_activity,
        'field_count': field_count,
        'details': details
    }


def generate_test_files(test_values):
    """
    生成所有测试输入文件
    """
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    
    print("="*60)
    print("剂量-响应验证测试 - 生成测试文件")
    print("="*60)
    print(f"\n测试值序列: {test_values}")
    print(f"每个测试: 3秒 @ 60Hz = 180帧")
    print(f"\n保存路径: {game_path}")
    print("\n" + "-"*60)
    
    for value in test_values:
        df = generate_test_input(value, duration_sec=3)
        filename = f"input_test_{int(value):03d}.csv"
        df.to_csv(game_path / filename, index=False)
        print(f"  ✓ {filename:25s} (输入值={value:6.1f}, 共{len(df)}帧)")
    
    print("\n" + "="*60)
    print(f"✅ 已生成 {len(test_values)} 个测试文件")
    print("="*60)


def show_test_instructions(test_values):
    """
    显示测试操作指南
    """
    print("\n" + "="*60)
    print("📋 手动测试流程")
    print("="*60)
    
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    
    print(f"\n游戏数据目录: {game_path}")
    print("\n对于每个测试值，执行以下步骤：\n")
    
    for i, value in enumerate(test_values, 1):
        print(f"【测试 {i}/{len(test_values)}】输入值 = {value}")
        print(f"  1. 重命名: input_test_{int(value):03d}.csv → input_commands.csv")
        print(f"  2. 运行游戏（自动采集3秒后关闭）")
        print(f"  3. 重命名结果: ContinuousTracking_latest.csv → result_{int(value):03d}.csv")
        print()
    
    print("="*60)
    print("⏱️ 预计耗时: 约 {} 分钟".format(len(test_values) * 0.5))
    print("="*60)
    
    print("\n💡 提示：")
    print("  - 游戏会自动加载 input_commands.csv")
    print("  - 采集3秒后自动关闭")
    print("  - 记得及时重命名结果文件，避免被覆盖")
    
    print("\n完成所有测试后，运行分析：")
    print("  python dose_response_test.py --analyze")


def analyze_results(test_values):
    """
    分析所有测试结果
    """
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    result_path = game_path / "GameResults"
    
    results = []
    missing = []
    
    print("="*60)
    print("剂量-响应分析 - 数据收集")
    print("="*60)
    
    for value in test_values:
        csv_file = result_path / f"result_{int(value):03d}.csv"
        
        if not csv_file.exists():
            print(f"⚠️  输入值 {value:6.1f} - 未找到结果文件")
            missing.append(value)
            continue
        
        effect = calculate_total_effect(csv_file)
        results.append({
            'input': value,
            'total_effect': effect['total_effect'],
            'total_change': effect['total_change'],
            'total_activity': effect['total_activity'],
            'field_count': effect['field_count']
        })
        
        print(f"✓  输入值 {value:6.1f} → 总效果: {effect['total_effect']:10.2f} "
              f"(变化={effect['total_change']:8.2f}, 活动={effect['total_activity']:8.2f})")
    
    if missing:
        print(f"\n⚠️  缺失 {len(missing)} 个测试结果: {missing}")
        print("     请完成这些测试后重新运行分析")
    
    if len(results) < 3:
        print("\n❌ 测试数据不足（至少需要3个数据点）")
        return None
    
    print(f"\n✅ 成功收集 {len(results)} 个有效数据点")
    
    # 分析单调性
    print("\n" + "="*60)
    print("📊 单调性检验（核心验证）")
    print("="*60)
    print("\n如果效果随输入单调递增 → 精度验证通过")
    print("如果出现平台期 → 精度可能不是10\n")
    
    is_monotonic = True
    monotonic_violations = []
    
    for i in range(1, len(results)):
        curr_input = results[i]['input']
        prev_input = results[i-1]['input']
        curr_effect = results[i]['total_effect']
        prev_effect = results[i-1]['total_effect']
        
        input_increase = curr_input - prev_input
        effect_increase = curr_effect - prev_effect
        
        if prev_effect > 0.1:
            change_rate = effect_increase / prev_effect * 100
        else:
            change_rate = 0
        
        if effect_increase > 0:
            status = "✓"
        else:
            status = "✗"
            is_monotonic = False
            monotonic_violations.append({
                'from': prev_input,
                'to': curr_input,
                'effect_from': prev_effect,
                'effect_to': curr_effect
            })
        
        print(f"{status} [{prev_input:6.1f} → {curr_input:6.1f}] 输入+{input_increase:4.1f}: "
              f"效果 {prev_effect:8.2f} → {curr_effect:8.2f} "
              f"(增长 {effect_increase:+8.2f}, {change_rate:+6.1f}%)")
    
    # 计算线性相关性
    print("\n" + "="*60)
    print("📈 线性度分析")
    print("="*60)
    
    inputs = np.array([r['input'] for r in results])
    effects = np.array([r['total_effect'] for r in results])
    
    # 皮尔逊相关系数
    correlation = np.corrcoef(inputs, effects)[0, 1]
    
    # 线性拟合
    slope, intercept = np.polyfit(inputs, effects, 1)
    predicted = slope * inputs + intercept
    r_squared = 1 - np.sum((effects - predicted)**2) / np.sum((effects - effects.mean())**2)
    
    print(f"\n  相关系数 (r):     {correlation:.4f}")
    print(f"  决定系数 (R²):    {r_squared:.4f}")
    print(f"  拟合斜率:         {slope:.4f}")
    print(f"  拟合截距:         {intercept:.4f}")
    
    # 分析饱和现象
    print("\n" + "="*60)
    print("📉 饱和点分析")
    print("="*60)
    
    saturation_detected = False
    saturation_point = None
    
    for i in range(1, len(results)):
        if results[i-1]['input'] < 40:  # 只在高值区域检测饱和
            continue
        
        curr_effect = results[i]['total_effect']
        prev_effect = results[i-1]['total_effect']
        
        if prev_effect > 0.1:
            growth_rate = (curr_effect - prev_effect) / prev_effect * 100
            
            if growth_rate < 5:  # 增长率低于5%即认为饱和
                if not saturation_detected:
                    saturation_detected = True
                    saturation_point = results[i-1]['input']
                print(f"  ⚠️  输入 {results[i-1]['input']} → {results[i]['input']}: "
                      f"增长率仅 {growth_rate:.1f}% (接近饱和)")
    
    if saturation_detected:
        print(f"\n  → 饱和点约在: {saturation_point:.1f}")
    else:
        print(f"\n  → 未检测到明显饱和（最大测试值: {results[-1]['input']:.1f}）")
    
    # 最终结论
    print("\n" + "="*60)
    print("🎯 验证结论")
    print("="*60)
    
    if is_monotonic:
        print("\n✅ 单调性验证: 通过")
        print("   效果随输入单调递增，无平台期")
        
        if correlation > 0.95:
            print(f"\n✅ 线性度验证: 优秀 (r={correlation:.4f})")
            print("   高度线性响应")
        elif correlation > 0.85:
            print(f"\n✅ 线性度验证: 良好 (r={correlation:.4f})")
            print("   接近线性响应")
        elif correlation > 0.7:
            print(f"\n⚠️  线性度验证: 一般 (r={correlation:.4f})")
            print("   存在一定非线性（可能是饱和效应）")
        else:
            print(f"\n⚠️  线性度验证: 较差 (r={correlation:.4f})")
            print("   显著非线性")
        
        print("\n" + "="*60)
        print("📌 最终结论:")
        print("="*60)
        print("✅ 输入精度验证通过！")
        print("   系统能够分辨不同的输入值")
        print("   精度10是有效的，不仅仅是最小有效值")
        print(f"\n推荐配置:")
        print(f"  - 最小有效输入: ~10")
        print(f"  - 饱和点: {saturation_point if saturation_detected else '> ' + str(results[-1]['input'])}")
        print(f"  - 有效范围: [10, {saturation_point if saturation_detected else int(results[-1]['input'])}]")
        
    else:
        print("\n❌ 单调性验证: 失败")
        print(f"   检测到 {len(monotonic_violations)} 处违反单调性:")
        for v in monotonic_violations:
            print(f"   - 输入 {v['from']} → {v['to']}: "
                  f"效果 {v['effect_from']:.2f} → {v['effect_to']:.2f} (未增长！)")
        
        print("\n" + "="*60)
        print("📌 最终结论:")
        print("="*60)
        print("⚠️  精度验证存在问题！")
        print("   可能原因:")
        print("   1. 系统存在未知的量化机制")
        print("   2. 精度不是简单的数值精度")
        print("   3. 测试数据有误差")
        print("\n   建议: 重新检查测试过程和数据")
    
    # 保存结果
    save_results_to_json(results, correlation, r_squared, is_monotonic, saturation_point)
    
    return results


def save_results_to_json(results, correlation, r_squared, is_monotonic, saturation_point):
    """保存分析结果到JSON"""
    import json
    from datetime import datetime
    
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    output_file = game_path / "dose_response_analysis.json"
    
    report = {
        'test_date': datetime.now().isoformat(),
        'test_values': [r['input'] for r in results],
        'results': results,
        'statistics': {
            'correlation': correlation,
            'r_squared': r_squared,
            'is_monotonic': is_monotonic,
            'saturation_point': saturation_point
        },
        'conclusion': {
            'precision_valid': is_monotonic,
            'linearity': 'good' if correlation > 0.85 else 'fair' if correlation > 0.7 else 'poor',
            'saturation_detected': saturation_point is not None
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存到: {output_file}")


# 主程序
if __name__ == '__main__':
    # 测试值序列
    test_values = [5, 10, 15, 20, 25, 30, 50, 80, 100]
    
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        # 分析模式
        print("\n🔬 开始分析测试结果...\n")
        analyze_results(test_values)
    else:
        # 生成测试文件模式
        generate_test_files(test_values)
        show_test_instructions(test_values)
        
        print("\n" + "="*60)
        print("🚀 准备开始测试")
        print("="*60)
        input("\n按 Enter 键开始第一个测试...")


"""
分析小数精度测试结果
判断系统能否区分0.1、0.5等小数差异
"""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_decimal_results():
    """分析小数精度测试结果"""
    test_values = [50.0, 50.1, 50.2, 50.5, 51.0]
    result_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData/GameResults")
    
    print("="*60)
    print("小数精度分析")
    print("="*60)
    print(f"\n测试序列: {test_values}")
    print(f"差异: 0.1, 0.1, 0.3, 0.5")
    
    # 收集所有测试的第2帧数据
    results = {}
    
    print("\n" + "-"*60)
    print("数据收集")
    print("-"*60)
    
    for value in test_values:
        csv_file = result_path / f"decimal_{int(value*10):03d}.csv"
        
        if not csv_file.exists():
            print(f"⚠️  跳过: 输入值 {value:.1f} (文件不存在)")
            continue
        
        df = pd.read_csv(csv_file)
        
        # 计算第2帧相对第0帧的变化
        key_fields = ['hammerAngle', 'sliderAngle', 'handleX', 'handleY', 'tipX', 'tipY',
                      'handleVelX', 'handleVelY', 'tipVelX', 'tipVelY', 'hubSliderAngle']
        
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
        
        print(f"✓ 输入值 {value:5.1f}: 第2帧总变化 = {frame2_change:10.6f}")
    
    if len(results) < 2:
        print("\n❌ 数据不足，无法进行分析")
        return
    
    # 详细字段分析
    print("\n" + "="*60)
    print("第2帧详细字段分析")
    print("="*60)
    
    sorted_values = sorted(results.keys())
    
    print(f"\n{'输入值':<8}", end='')
    key_fields_display = ['hammerAngle', 'sliderAngle', 'handleX', 'tipX', 'tipVelX']
    for field in key_fields_display:
        print(f"{field:<15}", end='')
    print()
    print("-"*90)
    
    for value in sorted_values:
        print(f"{value:<8.1f}", end='')
        for field in key_fields_display:
            if field in results[value]['fields']:
                change = results[value]['fields'][field]
                print(f"{change:<15.6f}", end='')
            else:
                print(f"{'N/A':<15}", end='')
        print()
    
    # 连续输入可区分性分析
    print("\n" + "="*60)
    print("小数差异可区分性分析（核心）")
    print("="*60)
    
    threshold = 0.01  # 可区分阈值
    distinguishable_pairs = []
    indistinguishable_pairs = []
    
    print(f"\n判断阈值: {threshold}")
    print()
    
    for i in range(len(sorted_values) - 1):
        val1 = sorted_values[i]
        val2 = sorted_values[i + 1]
        input_diff = val2 - val1
        
        change1 = results[val1]['total_change']
        change2 = results[val2]['total_change']
        effect_diff = abs(change2 - change1)
        relative_diff = (effect_diff / change1 * 100) if change1 > 0 else 0
        
        if effect_diff > threshold:
            status = "✓ 可区分"
            distinguishable_pairs.append((val1, val2, input_diff, effect_diff))
        else:
            status = "✗ 不可区分"
            indistinguishable_pairs.append((val1, val2, input_diff, effect_diff))
        
        print(f"输入 {val1:5.1f} vs {val2:5.1f} (差异={input_diff:.1f}):")
        print(f"  第2帧变化: {change1:10.6f} vs {change2:10.6f}")
        print(f"  效果差异:   {effect_diff:10.6f} ({relative_diff:6.2f}%)")
        print(f"  → {status}")
        print()
    
    # 精度估算
    print("="*60)
    print("🎯 小数精度验证结论")
    print("="*60)
    
    total_pairs = len(sorted_values) - 1
    distinguishable_count = len(distinguishable_pairs)
    
    print(f"\n可区分对: {distinguishable_count}/{total_pairs}")
    
    # 找到最小可区分差异
    if distinguishable_pairs:
        min_distinguishable_diff = min(d[2] for d in distinguishable_pairs)
        print(f"最小可区分差异: {min_distinguishable_diff:.1f}")
    
    # 找到最大不可区分差异
    if indistinguishable_pairs:
        max_indistinguishable_diff = max(d[2] for d in indistinguishable_pairs)
        print(f"最大不可区分差异: {max_indistinguishable_diff:.1f}")
    
    print()
    
    if distinguishable_count == total_pairs:
        print("✅ 所有小数差异都可区分！")
        
        if any(d[2] <= 0.1 for d in distinguishable_pairs):
            print("\n【结论】")
            print("  → 真实精度 ≤ 0.1")
            print("  → 系统能够分辨0.1单位的输入差异")
            print("  → **精度比预期更高！**")
            
            precision_result = {
                'precision_verified': True,
                'precision_value': '≤0.1',
                'note': 'System can distinguish 0.1 unit differences'
            }
        else:
            print("\n【结论】")
            print("  → 真实精度在 0.1 - 1.0 之间")
            print("  → 需要进一步测试更小的间隔")
            
            precision_result = {
                'precision_verified': 'partial',
                'precision_value': '0.1-1.0',
                'note': 'Need smaller intervals to determine exact precision'
            }
    
    elif distinguishable_count == 0:
        print("❌ 所有小数差异都不可区分")
        print("\n【结论】")
        print("  → 真实精度 > 1.0")
        print("  → 之前的整数测试可能接近测量误差边界")
        print("  → **精度可能就是1.0**")
        
        precision_result = {
            'precision_verified': True,
            'precision_value': '~1.0',
            'note': 'Cannot distinguish decimal differences, precision is likely 1.0'
        }
    
    else:
        print(f"⚠️  部分可区分 ({distinguishable_count}/{total_pairs})")
        
        print("\n可区分对:")
        for v1, v2, diff, eff_diff in distinguishable_pairs:
            print(f"  - {v1:.1f} vs {v2:.1f}: 输入差={diff:.1f}, 效果差={eff_diff:.6f}")
        
        print("\n不可区分对:")
        for v1, v2, diff, eff_diff in indistinguishable_pairs:
            print(f"  - {v1:.1f} vs {v2:.1f}: 输入差={diff:.1f}, 效果差={eff_diff:.6f}")
        
        # 分析临界点
        if distinguishable_pairs and indistinguishable_pairs:
            min_dist_diff = min(d[2] for d in distinguishable_pairs)
            max_indist_diff = max(d[2] for d in indistinguishable_pairs)
            
            if min_dist_diff > max_indist_diff:
                estimated_precision = (min_dist_diff + max_indist_diff) / 2
                print(f"\n【估算精度】")
                print(f"  介于 {max_indist_diff:.1f} 和 {min_dist_diff:.1f} 之间")
                print(f"  估算值: {estimated_precision:.2f}")
                
                precision_result = {
                    'precision_verified': 'estimated',
                    'precision_value': f'~{estimated_precision:.2f}',
                    'precision_range': [max_indist_diff, min_dist_diff]
                }
            else:
                print(f"\n【结论】")
                print(f"  精度不均匀或存在测量噪声")
                
                precision_result = {
                    'precision_verified': 'uncertain',
                    'precision_value': 'non-uniform',
                    'note': 'Precision may vary or measurement has noise'
                }
        else:
            precision_result = {
                'precision_verified': 'partial',
                'precision_value': 'unknown'
            }
    
    # 最终建议
    print("\n" + "="*60)
    print("💡 建议")
    print("="*60)
    
    if distinguishable_count == total_pairs and any(d[2] <= 0.1 for d in distinguishable_pairs):
        print("\n✅ 系统精度足够高")
        print("   - 可以使用小数精度（如0.1）的动作")
        print("   - 训练时可以更细粒度控制")
        print("   - 推荐动作空间: Box(low=-80.0, high=80.0, dtype=np.float32)")
    
    elif distinguishable_count == 0:
        print("\n✅ 确认整数精度")
        print("   - 系统精度为整数级别（1.0）")
        print("   - 不需要使用小数精度")
        print("   - 推荐动作空间: Box(low=-80.0, high=80.0, dtype=np.float32)")
        print("   - 或简化为: 整数动作")
    
    else:
        print("\n⚠️  需要更多测试")
        print("   - 在可区分和不可区分的临界区域密集测试")
        print("   - 确定精确的精度阈值")
    
    # 保存报告
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    
    report = {
        'test_date': datetime.now().isoformat(),
        'test_type': 'decimal_precision',
        'test_values': test_values,
        'threshold': threshold,
        'results': {
            str(k): {
                'total_change': float(v['total_change']),
                'fields': {fk: float(fv) for fk, fv in v['fields'].items()}
            } for k, v in results.items()
        },
        'distinguishable_pairs': len(distinguishable_pairs),
        'total_pairs': total_pairs,
        'conclusion': precision_result
    }
    
    report_file = game_path / "decimal_precision_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存: {report_file}")
    
    return precision_result


if __name__ == '__main__':
    print("\n" + "🔬"*30)
    print("小数精度分析")
    print("🔬"*30)
    print()
    
    result = analyze_decimal_results()


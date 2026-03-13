"""
分析精密精度测试结果 - 50附近区间
判断系统能否区分1单位的输入差异
"""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_precision_results():
    """分析精密测试结果"""
    test_values = [49, 50, 51, 52]
    result_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData/GameResults")
    
    print("="*60)
    print("精密精度分析 - 50附近区间")
    print("="*60)
    print(f"\n测试序列: {test_values}")
    print(f"间隔: 1单位")
    print(f"目标: 验证系统能否区分1单位的输入差异")
    
    # 收集所有测试的第2帧数据
    results = {}
    
    print("\n" + "-"*60)
    print("数据收集")
    print("-"*60)
    
    for value in test_values:
        csv_file = result_path / f"precision_{int(value)}.csv"
        
        if not csv_file.exists():
            print(f"⚠️  跳过: 输入值 {value} (文件不存在)")
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
            'fields': field_changes,
            'data': df
        }
        
        print(f"✓ 输入值 {value}: 第2帧总变化 = {frame2_change:10.6f}")
    
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
    print("-"*80)
    
    for value in sorted_values:
        print(f"{value:<8.0f}", end='')
        for field in key_fields_display:
            if field in results[value]['fields']:
                change = results[value]['fields'][field]
                print(f"{change:<15.6f}", end='')
            else:
                print(f"{'N/A':<15}", end='')
        print()
    
    # 连续输入可区分性分析
    print("\n" + "="*60)
    print("连续输入可区分性分析（核心）")
    print("="*60)
    
    threshold = 0.01  # 可区分阈值
    distinguishable_pairs = []
    indistinguishable_pairs = []
    
    print(f"\n判断阈值: {threshold}")
    print(f"如果两个输入的第2帧效果差异 > {threshold} → 可区分")
    print()
    
    for i in range(len(sorted_values) - 1):
        val1 = sorted_values[i]
        val2 = sorted_values[i + 1]
        
        change1 = results[val1]['total_change']
        change2 = results[val2]['total_change']
        diff = abs(change2 - change1)
        relative_diff = (diff / change1 * 100) if change1 > 0 else 0
        
        if diff > threshold:
            status = "✓ 可区分"
            distinguishable_pairs.append((val1, val2, diff))
        else:
            status = "✗ 不可区分"
            indistinguishable_pairs.append((val1, val2, diff))
        
        print(f"输入 {val1:.0f} vs {val2:.0f} (差异=1单位):")
        print(f"  第2帧变化: {change1:10.6f} vs {change2:10.6f}")
        print(f"  效果差异:   {diff:10.6f} ({relative_diff:6.2f}%)")
        print(f"  → {status}")
        print()
    
    # 最终结论
    print("="*60)
    print("🎯 精度验证结论")
    print("="*60)
    
    total_pairs = len(sorted_values) - 1
    distinguishable_count = len(distinguishable_pairs)
    
    print(f"\n可区分对: {distinguishable_count}/{total_pairs}")
    
    if distinguishable_count == total_pairs:
        print("\n✅ 所有连续输入对都可区分！")
        print("\n【结论】")
        print("  → 真实精度 ≤ 1")
        print("  → 系统能够分辨1单位的输入差异")
        print("\n【推荐配置】")
        print("  最小有效输入: ~5-10 (存在死区)")
        print("  输入精度: 1.0")
        print("  有效范围: [10, 80]")
        print("  动作空间: Box(low=-80, high=80, dtype=np.float32)")
        print("\n【训练建议】")
        print("  - 可以使用连续动作空间")
        print("  - 精度足够支持细粒度控制")
        print("  - 建议动作值归一化到[-1, 1]再映射到[-80, 80]")
        
        precision_result = {
            'precision_verified': True,
            'precision_value': 1.0,
            'min_effective_input': 10,
            'max_effective_input': 80,
            'recommended_action_space': 'continuous'
        }
        
    elif distinguishable_count == 0:
        print("\n❌ 所有连续输入对都不可区分")
        print("\n【结论】")
        print("  → 真实精度 > 1")
        print("  → 系统无法分辨1单位的输入差异")
        print("\n【建议】")
        print("  需要测试更大的间隔:")
        print("  - 测试间隔2: [48, 50, 52, 54]")
        print("  - 测试间隔5: [45, 50, 55, 60]")
        print("  - 测试间隔10: [40, 50, 60, 70]")
        
        precision_result = {
            'precision_verified': False,
            'precision_value': '>1',
            'note': 'Need to test larger intervals'
        }
        
    else:
        print(f"\n⚠️  部分可区分 ({distinguishable_count}/{total_pairs})")
        print("\n可区分对:")
        for v1, v2, d in distinguishable_pairs:
            print(f"  - {v1} vs {v2}: 差异={d:.6f}")
        print("\n不可区分对:")
        for v1, v2, d in indistinguishable_pairs:
            print(f"  - {v1} vs {v2}: 差异={d:.6f}")
        
        print("\n【可能原因】")
        print("  1. 精度在某些区间不均匀")
        print("  2. 存在测量噪声")
        print("  3. 阈值设置过高")
        
        print("\n【建议】")
        print("  - 检查不可区分对的详细字段")
        print("  - 考虑降低阈值重新判断")
        print("  - 或增加测试样本")
        
        precision_result = {
            'precision_verified': 'partial',
            'precision_value': '~1',
            'distinguishable_pairs': distinguishable_count,
            'total_pairs': total_pairs
        }
    
    # 保存报告
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    
    report = {
        'test_date': datetime.now().isoformat(),
        'test_region': '50附近',
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
    
    report_file = game_path / "precision_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存: {report_file}")
    
    return precision_result


if __name__ == '__main__':
    print("\n" + "🔬"*30)
    print("精密精度分析")
    print("🔬"*30)
    print()
    
    result = analyze_precision_results()


"""
分析第一次状态变化 - 真正的精度验证

核心思想：
精度不是看累积效果，而是看即时响应！
- 输入10 vs 输入11，第1-3帧的状态是否不同？
- 如果不同 → 精度 ≤ 1
- 如果相同，但输入10 vs 输入20不同 → 精度在10-20之间
"""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_first_n_frames(csv_path, n_frames=5):
    """
    分析前N帧的状态变化
    
    Args:
        csv_path: CSV文件路径
        n_frames: 分析前N帧
    
    Returns:
        dict: 包含各帧的状态变化
    """
    df = pd.read_csv(csv_path)
    
    if len(df) < n_frames:
        print(f"⚠️ 数据点不足：只有{len(df)}帧")
        n_frames = len(df)
    
    # 关键字段
    key_fields = [
        'hammerAngle', 'hammerAngularVel',
        'sliderAngle', 
        'handleX', 'handleY', 'handleVelX', 'handleVelY',
        'tipX', 'tipY', 'tipVelX', 'tipVelY',
        'hubSliderAngle'
    ]
    
    # 计算每一帧相对于第0帧的变化
    frame_changes = []
    
    baseline = df.iloc[0]
    
    for frame_idx in range(1, n_frames + 1):
        if frame_idx >= len(df):
            break
        
        current = df.iloc[frame_idx]
        
        frame_change = {
            'frame': frame_idx,
            'timestamp': current['timestamp'],
            'total_change': 0,
            'field_changes': {}
        }
        
        for field in key_fields:
            if field not in df.columns:
                continue
            
            change = abs(current[field] - baseline[field])
            frame_change['field_changes'][field] = change
            frame_change['total_change'] += change
        
        frame_changes.append(frame_change)
    
    return frame_changes


def compare_inputs_first_response(test_values, n_frames=5):
    """
    比较不同输入值的第一次响应
    核心：找到首次产生不同效果的最小输入差异
    """
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    result_path = game_path / "GameResults"
    
    print("="*60)
    print("第一次响应分析 - 真正的精度验证")
    print("="*60)
    
    results = {}
    
    # 收集所有测试的前N帧数据
    for value in test_values:
        csv_file = result_path / f"result_{int(value):03d}.csv"
        
        if not csv_file.exists():
            print(f"⚠️  跳过: 输入值 {value} (文件不存在)")
            continue
        
        frame_changes = analyze_first_n_frames(csv_file, n_frames)
        results[value] = frame_changes
        
        print(f"✓ 输入值 {value:6.1f} - 已加载前{len(frame_changes)}帧数据")
    
    if len(results) < 2:
        print("\n❌ 测试数据不足（至少需要2个数据点）")
        return None
    
    # 分析每一帧的变化
    print("\n" + "="*60)
    print("逐帧变化分析")
    print("="*60)
    
    sorted_values = sorted(results.keys())
    
    for frame_idx in range(n_frames):
        print(f"\n【第 {frame_idx + 1} 帧】")
        print("-"*60)
        
        has_data = False
        
        for value in sorted_values:
            if frame_idx < len(results[value]):
                frame_data = results[value][frame_idx]
                print(f"  输入 {value:6.1f} → 总变化: {frame_data['total_change']:10.4f}")
                has_data = True
        
        if not has_data:
            break
    
    # 关键分析：找到首次出现差异的帧
    print("\n" + "="*60)
    print("首次响应差异分析（核心）")
    print("="*60)
    
    # 对于每对连续的输入值，找到它们首次产生不同效果的帧
    first_diff_frames = []
    
    for i in range(len(sorted_values) - 1):
        val1 = sorted_values[i]
        val2 = sorted_values[i + 1]
        input_diff = val2 - val1
        
        print(f"\n比较: 输入 {val1:6.1f} vs {val2:6.1f} (差异={input_diff:4.1f})")
        
        found_diff = False
        threshold = 0.01  # 变化阈值
        
        for frame_idx in range(min(len(results[val1]), len(results[val2]))):
            change1 = results[val1][frame_idx]['total_change']
            change2 = results[val2][frame_idx]['total_change']
            diff = abs(change2 - change1)
            
            if diff > threshold:
                print(f"  → 第 {frame_idx + 1} 帧首次出现差异:")
                print(f"     输入{val1}: 变化={change1:8.4f}")
                print(f"     输入{val2}: 变化={change2:8.4f}")
                print(f"     差异={diff:8.4f}")
                
                first_diff_frames.append({
                    'input1': val1,
                    'input2': val2,
                    'input_diff': input_diff,
                    'first_diff_frame': frame_idx + 1,
                    'effect_diff': diff
                })
                
                found_diff = True
                break
        
        if not found_diff:
            print(f"  → 前{n_frames}帧无明显差异（可能精度不足）")
            first_diff_frames.append({
                'input1': val1,
                'input2': val2,
                'input_diff': input_diff,
                'first_diff_frame': None,
                'effect_diff': 0
            })
    
    # 估算精度
    print("\n" + "="*60)
    print("精度估算")
    print("="*60)
    
    # 找到能产生差异的最小输入差
    valid_diffs = [d for d in first_diff_frames if d['first_diff_frame'] is not None]
    
    if valid_diffs:
        min_effective_diff = min(d['input_diff'] for d in valid_diffs)
        print(f"\n✓ 能产生即时差异的最小输入差: {min_effective_diff:.1f}")
        print(f"  → 估算精度: 约 {min_effective_diff:.1f}")
        
        # 检查所有小于此值的输入差是否都无效
        invalid_diffs = [d for d in first_diff_frames if d['first_diff_frame'] is None]
        if invalid_diffs:
            print(f"\n  以下输入差未产生即时差异:")
            for d in invalid_diffs:
                print(f"    输入差 {d['input_diff']:4.1f} ({d['input1']:4.1f} vs {d['input2']:4.1f})")
    else:
        print(f"\n❌ 所有输入值在前{n_frames}帧都未产生明显差异")
        print(f"  可能原因:")
        print(f"  1. 精度非常大（> {max(d['input_diff'] for d in first_diff_frames):.1f}）")
        print(f"  2. 测试值范围不足")
        print(f"  3. 需要更多帧才能观察到差异")
    
    # 最终结论
    print("\n" + "="*60)
    print("🎯 结论")
    print("="*60)
    
    if valid_diffs:
        min_diff = min(d['input_diff'] for d in valid_diffs)
        
        # 检查是否有更小的输入差也有效
        smaller_valid = [d for d in valid_diffs if d['input_diff'] <= min_diff * 1.5]
        
        if len(smaller_valid) >= 2:
            avg_min_diff = np.mean([d['input_diff'] for d in smaller_valid])
            print(f"\n估算的最小有效精度: {avg_min_diff:.1f}")
        else:
            print(f"\n估算的最小有效精度: {min_diff:.1f}")
        
        print(f"\n验证方法:")
        print(f"  - 测试输入差 < {min_diff:.1f} 的值对")
        print(f"  - 例如: {min_diff/2:.1f}, {min_diff:.1f}, {min_diff*1.5:.1f}")
        print(f"  - 找到首次产生差异的最小间隔")
    
    return first_diff_frames


def detailed_field_analysis(test_values, frame_idx=1):
    """
    详细分析某一帧各个字段的变化
    """
    game_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
    result_path = game_path / "GameResults"
    
    print("\n" + "="*60)
    print(f"第{frame_idx}帧详细字段分析")
    print("="*60)
    
    key_fields = [
        'hammerAngle', 'hammerAngularVel',
        'sliderAngle', 
        'handleX', 'handleY', 'handleVelX', 'handleVelY',
        'tipX', 'tipY', 'tipVelX', 'tipVelY',
    ]
    
    print(f"\n{'输入值':<8}", end='')
    for field in key_fields:
        print(f"{field:<15}", end='')
    print()
    print("-"*120)
    
    for value in sorted(test_values):
        csv_file = result_path / f"result_{int(value):03d}.csv"
        
        if not csv_file.exists():
            continue
        
        df = pd.read_csv(csv_file)
        
        if frame_idx >= len(df):
            continue
        
        baseline = df.iloc[0]
        current = df.iloc[frame_idx]
        
        print(f"{value:<8.1f}", end='')
        
        for field in key_fields:
            if field in df.columns:
                change = abs(current[field] - baseline[field])
                print(f"{change:<15.4f}", end='')
            else:
                print(f"{'N/A':<15}", end='')
        
        print()


if __name__ == '__main__':
    test_values = [5, 10, 15, 20, 25, 30, 50, 80, 100]
    
    print("\n" + "🔬"*30)
    print("第一次响应分析 - 精度验证")
    print("🔬"*30)
    
    # 主分析
    compare_inputs_first_response(test_values, n_frames=10)
    
    # 详细字段分析（第1帧）
    detailed_field_analysis(test_values, frame_idx=1)
    
    print("\n" + "="*60)
    print("💡 提示")
    print("="*60)
    print("\n精度定义:")
    print("  系统能够分辨的最小输入差异")
    print("  = 能在第1-3帧产生不同效果的最小输入间隔")
    print("\n如果输入10和输入20在第1帧就不同:")
    print("  → 精度 ≤ 10")
    print("\n如果输入10和输入20在前几帧相同:")
    print("  → 精度可能 > 10")


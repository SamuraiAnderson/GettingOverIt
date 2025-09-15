#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Getting Over It 真实数据分析脚本
专门处理从游戏导出的真实CSV数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_real_data(csv_file_path):
    """加载真实游戏数据，跳过头部注释"""
    try:
        # 找到数据开始行
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data_start_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('timestamp,deltaTime'):
                data_start_line = i
                break
        
        if data_start_line == 0:
            print("❌ 未找到数据开始行")
            return None
        
        # 读取数据
        data = pd.read_csv(csv_file_path, skiprows=data_start_line)
        print(f"✅ 成功加载真实数据: {len(data)} 个数据点")
        print(f"📊 时间跨度: {data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]:.2f} 秒")
        print(f"⚡ 实际采样频率: {len(data) / (data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]):.1f} Hz")
        return data
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def analyze_real_input_dof(data):
    """分析真实输入自由度 (3DOF)"""
    print("\n🎮 === 真实输入分析 (3DOF) ===")
    
    # 1. 鼠标轨迹分析
    mouse_range_x = data['mouseX'].max() - data['mouseX'].min()
    mouse_range_y = data['mouseY'].max() - data['mouseY'].min()
    
    # 2. 鼠标移动模式
    mouse_movement = np.sqrt(data['mouseDeltaX']**2 + data['mouseDeltaY']**2)
    avg_movement = mouse_movement.mean()
    max_movement = mouse_movement.max()
    
    # 3. 主动控制模式分析 (Getting Over It没有抓握按钮，isGripping表示主动控制状态)
    grip_changes = data['grippingChanged'].sum()
    grip_time = data['isGripping'].mean()
    
    # 4. 输入活跃度
    active_frames = (mouse_movement > 1.0).sum()
    activity_ratio = active_frames / len(data)
    
    print(f"📱 鼠标使用范围: X={mouse_range_x:.0f}px, Y={mouse_range_y:.0f}px")
    print(f"🖱️ 平均移动速度: {avg_movement:.1f}px/frame")
    print(f"⚡ 最大移动速度: {max_movement:.1f}px/frame")
    print(f"🤏 主动控制变化次数: {grip_changes}次")
    print(f"⏱️ 主动控制时间比例: {grip_time*100:.1f}%")
    print(f"🎯 输入活跃度: {activity_ratio*100:.1f}%")
    
    return {
        'mouse_range_x': mouse_range_x,
        'mouse_range_y': mouse_range_y,
        'avg_movement': avg_movement,
        'max_movement': max_movement,
        'grip_changes': grip_changes,
        'grip_time_ratio': grip_time,
        'activity_ratio': activity_ratio
    }

def analyze_real_state_dof(data):
    """分析真实状态自由度 (8DOF)"""
    print("\n🎯 === 真实状态分析 (8DOF) ===")
    
    # 1. 位置分析 (2DOF)
    pos_range_x = data['playerX'].max() - data['playerX'].min()
    pos_range_y = data['playerY'].max() - data['playerY'].min()
    
    # 2. 速度分析 (2DOF)
    speed = np.sqrt(data['velocityX']**2 + data['velocityY']**2)
    avg_speed = speed.mean()
    max_speed = speed.max()
    speed_std = speed.std()
    
    # 3. 锤子角度分析 (1DOF)
    angle_range = data['hammerAngle'].max() - data['hammerAngle'].min()
    avg_angular_vel = abs(data['hammerAngularVel']).mean()
    max_angular_vel = abs(data['hammerAngularVel']).max()
    
    # 4. 离散状态分析 (2DOF)
    grounded_ratio = data['isGrounded'].mean()
    
    # 5. 状态变化率
    avg_pos_delta = data['positionDelta'].mean()
    avg_vel_delta = data['velocityDelta'].mean()
    avg_angle_delta = abs(data['angleDelta']).mean()
    
    print(f"📍 玩家移动范围: X={pos_range_x:.2f}, Y={pos_range_y:.2f}")
    print(f"🏃 平均速度: {avg_speed:.2f} ± {speed_std:.2f}")
    print(f"⚡ 最大速度: {max_speed:.2f}")
    print(f"🔨 锤子角度范围: {angle_range:.1f}°")
    print(f"🌪️ 平均角速度: {avg_angular_vel:.1f}°/s")
    print(f"⚡ 最大角速度: {max_angular_vel:.1f}°/s")
    print(f"🏔️ 接地时间比例: {grounded_ratio*100:.1f}%")
    print(f"📈 平均状态变化: 位置={avg_pos_delta:.4f}, 速度={avg_vel_delta:.3f}, 角度={avg_angle_delta:.2f}°")
    
    return {
        'pos_range_x': pos_range_x,
        'pos_range_y': pos_range_y,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'speed_std': speed_std,
        'angle_range': angle_range,
        'avg_angular_vel': avg_angular_vel,
        'max_angular_vel': max_angular_vel,
        'grounded_ratio': grounded_ratio,
        'avg_pos_delta': avg_pos_delta,
        'avg_vel_delta': avg_vel_delta,
        'avg_angle_delta': avg_angle_delta
    }

def analyze_real_correlations(data):
    """分析真实输入-状态关联性"""
    print("\n⚡ === 真实输入-状态关联分析 ===")
    
    # 计算关键相关性
    correlations = {}
    
    # 鼠标位置 vs 玩家位置
    correlations['mouseX_playerX'] = data['mouseX'].corr(data['playerX'])
    correlations['mouseY_playerY'] = data['mouseY'].corr(data['playerY'])
    
    # 鼠标移动 vs 玩家速度
    mouse_movement = np.sqrt(data['mouseDeltaX']**2 + data['mouseDeltaY']**2)
    player_speed = np.sqrt(data['velocityX']**2 + data['velocityY']**2)
    correlations['mouse_movement_player_speed'] = mouse_movement.corr(player_speed)
    
    # 抓握状态 vs 游戏状态
    correlations['gripping_grounded'] = data['isGripping'].corr(data['isGrounded'])
    
    # 分析响应延迟
    delays = {}
    for delay in range(1, min(11, len(data)//10)):  # 限制延迟范围
        if len(data) > delay:
            delayed_corr = data['mouseX'].corr(data['playerX'].shift(-delay))
            delays[f'delay_{delay}'] = delayed_corr
    
    best_delay = max(delays.items(), key=lambda x: abs(x[1])) if delays else ('delay_1', 0)
    
    print(f"🎮 鼠标X ↔ 玩家X: {correlations['mouseX_playerX']:.3f}")
    print(f"🎮 鼠标Y ↔ 玩家Y: {correlations['mouseY_playerY']:.3f}")
    print(f"⚡ 鼠标移动 ↔ 玩家速度: {correlations['mouse_movement_player_speed']:.3f}")
    print(f"🤏 抓握 ↔ 接地: {correlations['gripping_grounded']:.3f}")
    print(f"⏱️ 最佳响应延迟: {best_delay[0]} (相关性: {best_delay[1]:.3f})")
    
    return correlations, delays

def analyze_real_patterns(data):
    """分析真实操作模式"""
    print("\n🎭 === 真实操作模式分析 ===")
    
    # 1. 抓握模式分析
    gripping_sequences = []
    current_sequence = 0
    for grip in data['isGripping']:
        if grip:
            current_sequence += 1
        else:
            if current_sequence > 0:
                gripping_sequences.append(current_sequence)
            current_sequence = 0
    
    if current_sequence > 0:
        gripping_sequences.append(current_sequence)
    
    # 2. 移动模式分析  
    mouse_movement = np.sqrt(data['mouseDeltaX']**2 + data['mouseDeltaY']**2)
    movement_peaks, _ = find_peaks(mouse_movement, height=mouse_movement.mean() + mouse_movement.std())
    
    # 3. 速度模式分析
    player_speed = np.sqrt(data['velocityX']**2 + data['velocityY']**2)
    speed_peaks, _ = find_peaks(player_speed, height=player_speed.mean() + player_speed.std())
    
    # 4. 碰撞检测
    collisions = (data['velocityDelta'] > player_speed.mean() + 2*player_speed.std()).sum()
    
    print(f"🤏 平均抓握持续: {np.mean(gripping_sequences):.1f} 帧" if gripping_sequences else "🤏 平均抓握持续: 0 帧")
    print(f"🕐 最长抓握时间: {max(gripping_sequences)} 帧" if gripping_sequences else "🕐 最长抓握时间: 0 帧")
    print(f"🔄 抓握序列数: {len(gripping_sequences)}")
    print(f"📈 移动峰值数: {len(movement_peaks)}")
    print(f"⚡ 速度峰值数: {len(speed_peaks)}")
    print(f"💥 估计碰撞次数: {collisions}")
    
    return {
        'avg_grip_duration': np.mean(gripping_sequences) if gripping_sequences else 0,
        'max_grip_duration': max(gripping_sequences) if gripping_sequences else 0,
        'grip_sequences': len(gripping_sequences),
        'movement_peaks': len(movement_peaks),
        'speed_peaks': len(speed_peaks),
        'estimated_collisions': collisions
    }

def generate_real_conclusions(input_results, state_results, correlations, patterns):
    """生成真实数据分析结论"""
    print("\n" + "="*80)
    print("🎯 === Getting Over It 真实数据分析结论 ===")
    print("="*80)
    
    print("\n📋 === 核心发现 ===")
    
    # 1. 输入特征分析
    print(f"\n🎮 【真实输入控制特征】")
    print(f"   • 输入空间利用: {input_results.get('mouse_range_x', 0):.0f}×{input_results.get('mouse_range_y', 0):.0f} 像素")
    print(f"   • 操作强度: {input_results.get('activity_ratio', 0)*100:.1f}% 时间在主动操作")
    print(f"   • 抓握策略: {input_results.get('grip_changes', 0)} 次抓握变化，{input_results.get('grip_time_ratio', 0)*100:.1f}% 时间保持抓握")
    print(f"   • 操作精度: 平均移动{input_results.get('avg_movement', 0):.1f}px/帧，最大{input_results.get('max_movement', 0):.1f}px/帧")
    
    # 2. 游戏状态特征
    print(f"\n🎯 【真实游戏状态特征】")
    print(f"   • 移动范围: X轴{state_results.get('pos_range_x', 0):.2f}，Y轴{state_results.get('pos_range_y', 0):.2f}")
    print(f"   • 运动强度: 平均速度{state_results.get('avg_speed', 0):.2f}，最大速度{state_results.get('max_speed', 0):.2f}")
    print(f"   • 锤子控制: 角度范围{state_results.get('angle_range', 0):.1f}°，平均角速度{state_results.get('avg_angular_vel', 0):.1f}°/s")
    print(f"   • 接地比例: {state_results.get('grounded_ratio', 0)*100:.1f}% 时间接触地面")
    
    # 3. 输入-状态关联
    print(f"\n⚡ 【真实控制效率分析】")
    mouseX_playerX = correlations.get('mouseX_playerX', 0)
    mouseY_playerY = correlations.get('mouseY_playerY', 0)
    movement_speed = correlations.get('mouse_movement_player_speed', 0)
    
    print(f"   • 水平控制效率: {abs(mouseX_playerX):.3f} ({'强' if abs(mouseX_playerX) > 0.5 else '中' if abs(mouseX_playerX) > 0.3 else '弱'})")
    print(f"   • 垂直控制效率: {abs(mouseY_playerY):.3f} ({'强' if abs(mouseY_playerY) > 0.5 else '中' if abs(mouseY_playerY) > 0.3 else '弱'})")
    print(f"   • 动态响应性: {abs(movement_speed):.3f} ({'优秀' if abs(movement_speed) > 0.4 else '良好' if abs(movement_speed) > 0.2 else '一般'})")
    
    # 4. 操作模式
    print(f"\n🎭 【真实操作模式特征】")
    avg_grip = patterns.get('avg_grip_duration', 0)
    max_grip = patterns.get('max_grip_duration', 0)
    collisions = patterns.get('estimated_collisions', 0)
    
    print(f"   • 抓握模式: 平均持续{avg_grip:.1f}帧，最长{max_grip}帧")
    print(f"   • 操作节奏: {patterns.get('movement_peaks', 0)}个移动峰值，{patterns.get('speed_peaks', 0)}个速度峰值")
    print(f"   • 碰撞频率: 估计{collisions}次显著碰撞")
    
    # 5. 综合评估
    print(f"\n🏆 === 综合评估 ===")
    
    # 控制精度评级
    control_precision = (abs(mouseX_playerX) + abs(mouseY_playerY)) / 2
    if control_precision > 0.6:
        precision_rating = "精确"
    elif control_precision > 0.4:
        precision_rating = "良好"
    elif control_precision > 0.2:
        precision_rating = "一般"
    else:
        precision_rating = "较差"
        
    # 操作流畅度评级
    smoothness = 1 - (state_results.get('speed_std', 0) / max(state_results.get('avg_speed', 1), 0.1))
    if smoothness > 0.7:
        smoothness_rating = "流畅"
    elif smoothness > 0.5:
        smoothness_rating = "较流畅"
    else:
        smoothness_rating = "不够流畅"
        
    # 策略效率评级
    strategy_efficiency = input_results.get('activity_ratio', 0) * abs(movement_speed)
    if strategy_efficiency > 0.3:
        strategy_rating = "高效"
    elif strategy_efficiency > 0.15:
        strategy_rating = "中等"
    else:
        strategy_rating = "低效"
        
    print(f"   • 控制精度: {precision_rating} (相关性系数: {control_precision:.3f})")
    print(f"   • 操作流畅度: {smoothness_rating} (平滑度: {smoothness:.3f})")
    print(f"   • 策略效率: {strategy_rating} (效率指数: {strategy_efficiency:.3f})")
    
    # 6. 改进建议
    print(f"\n💡 === 改进建议 ===")
    
    suggestions = []
    
    if control_precision < 0.4:
        suggestions.append("• 提高鼠标精度设置，练习精确控制")
        
    if input_results.get('activity_ratio', 0) < 0.3:
        suggestions.append("• 增加操作频率，保持持续的微调控制")
        
    if state_results.get('grounded_ratio', 0) < 0.2:
        suggestions.append("• 提高接地稳定性，减少空中失控时间")
        
    if avg_grip < 5:
        suggestions.append("• 延长抓握持续时间，提高控制稳定性")
        
    if abs(movement_speed) < 0.2:
        suggestions.append("• 优化输入时机，提高动作响应性")
        
    if smoothness < 0.5:
        suggestions.append("• 练习平滑操作，减少急速变化")
        
    if not suggestions:
        suggestions.append("• 当前操作已达到较高水平，继续保持！")
        
    for suggestion in suggestions:
        print(f"   {suggestion}")
    
    print("\n" + "="*80)
    print("✅ 真实数据分析完成！")
    print("="*80)

def main():
    """主函数"""
    import glob
    import os
    
    # 自动查找最新的真实数据文件
    csv_files = glob.glob("../Data/ContinuousTracking_*.csv")
    
    if not csv_files:
        print("❌ 未找到真实数据文件")
        print("💡 请确保CSV文件在 ../Data/ 目录中")
        return
    
    # 选择最新的文件
    latest_csv = max(csv_files, key=os.path.getctime)
    csv_file = os.path.basename(latest_csv)
    
    print("🚀 === Getting Over It 真实数据分析 ===")
    print(f"📁 分析文件: {csv_file}")
    
    # 1. 加载数据
    data = load_real_data(latest_csv)
    if data is None:
        return
    
    # 2. 执行各项分析
    input_results = analyze_real_input_dof(data)
    state_results = analyze_real_state_dof(data)
    correlations, delays = analyze_real_correlations(data)
    patterns = analyze_real_patterns(data)
    
    # 3. 生成结论
    generate_real_conclusions(input_results, state_results, correlations, patterns)

if __name__ == "__main__":
    main()

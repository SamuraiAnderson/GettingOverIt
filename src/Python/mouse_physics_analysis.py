#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Getting Over It 鼠标移动-物理响应分析
专注于分析鼠标移动模式与游戏物理响应的映射关系
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib的字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_mouse_physics_data(csv_file_path):
    """加载鼠标-物理响应数据"""
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
        print(f"✅ 成功加载鼠标-物理响应数据: {len(data)} 个数据点")
        
        # 计算时间跨度
        if len(data) > 0:
            duration = data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]
            actual_freq = len(data) / duration
            print(f"📊 时间跨度: {duration:.2f} 秒")
            print(f"⚡ 实际采样频率: {actual_freq:.1f} Hz")
        
        return data
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def analyze_mouse_movement_patterns(data):
    """分析鼠标移动模式"""
    print("\n🖱️ === 鼠标移动模式分析 ===")
    
    # 1. 鼠标移动轨迹分析
    mouse_range_x = data['mouseX'].max() - data['mouseX'].min()
    mouse_range_y = data['mouseY'].max() - data['mouseY'].min()
    
    # 2. 鼠标移动速度和距离分析
    mouse_move_distance = data['mouseMoveDistance']
    mouse_move_speed = data['mouseMoveSpeed']
    
    # 3. 移动模式统计
    avg_move_distance = mouse_move_distance.mean()
    max_move_distance = mouse_move_distance.max()
    avg_move_speed = mouse_move_speed.mean()
    max_move_speed = mouse_move_speed.max()
    
    # 4. 移动方向分析
    mouse_delta_x = data['mouseDeltaX']
    mouse_delta_y = data['mouseDeltaY']
    
    # 计算主要移动方向
    total_x_movement = mouse_delta_x.abs().sum()
    total_y_movement = mouse_delta_y.abs().sum()
    dominant_direction = "水平" if total_x_movement > total_y_movement else "垂直"
    
    # 5. 移动节奏分析
    # 检测移动峰值（快速移动）
    move_peaks, _ = find_peaks(mouse_move_speed, height=mouse_move_speed.mean() + mouse_move_speed.std())
    num_rapid_movements = len(move_peaks)
    
    # 静止时间分析
    stationary_frames = (mouse_move_distance < 0.5).sum()
    stationary_ratio = stationary_frames / len(data)
    
    print(f"📱 鼠标使用范围: X={mouse_range_x:.0f}px, Y={mouse_range_y:.0f}px")
    print(f"🖱️ 平均移动距离: {avg_move_distance:.2f}px/frame")
    print(f"⚡ 最大移动距离: {max_move_distance:.2f}px/frame")
    print(f"🏃 平均移动速度: {avg_move_speed:.1f}px/s")
    print(f"🚀 最大移动速度: {max_move_speed:.1f}px/s")
    print(f"🧭 主要移动方向: {dominant_direction} (X:{total_x_movement:.0f} vs Y:{total_y_movement:.0f})")
    print(f"📈 快速移动次数: {num_rapid_movements}")
    print(f"⏸️ 静止时间比例: {stationary_ratio:.1%}")
    
    return {
        'mouse_range_x': mouse_range_x,
        'mouse_range_y': mouse_range_y,
        'avg_move_distance': avg_move_distance,
        'max_move_distance': max_move_distance,
        'avg_move_speed': avg_move_speed,
        'max_move_speed': max_move_speed,
        'dominant_direction': dominant_direction,
        'num_rapid_movements': num_rapid_movements,
        'stationary_ratio': stationary_ratio
    }

def analyze_physics_response(data):
    """分析物理响应"""
    print("\n⚡ === 物理响应分析 ===")
    
    # 1. 角色运动分析
    player_speed = np.sqrt(data['velocityX']**2 + data['velocityY']**2)
    avg_speed = player_speed.mean()
    max_speed = player_speed.max()
    speed_std = player_speed.std()
    
    # 2. 锤子控制分析
    hammer_angle_range = data['hammerAngle'].max() - data['hammerAngle'].min()
    avg_angular_vel = data['hammerAngularVel'].abs().mean()
    max_angular_vel = data['hammerAngularVel'].abs().max()
    
    # 3. 物理变化分析
    position_delta = data['positionDelta']
    velocity_delta = data['velocityDelta']
    angle_delta = data['angleDelta']
    
    avg_position_change = position_delta.mean()
    avg_velocity_change = velocity_delta.mean()
    avg_angle_change = angle_delta.abs().mean()
    
    # 4. 接地状态分析
    grounded_ratio = data['isGrounded'].mean()
    
    # 5. 物理响应延迟分析
    response_delay_ratio = data['physicsResponseDelay'].mean()
    
    print(f"🏃 角色平均速度: {avg_speed:.2f} ± {speed_std:.2f}")
    print(f"⚡ 角色最大速度: {max_speed:.2f}")
    print(f"🔨 锤子角度范围: {hammer_angle_range:.1f}°")
    print(f"🌪️ 平均角速度: {avg_angular_vel:.1f}°/s")
    print(f"🚀 最大角速度: {max_angular_vel:.1f}°/s")
    print(f"📈 平均位置变化: {avg_position_change:.4f}")
    print(f"⚡ 平均速度变化: {avg_velocity_change:.3f}")
    print(f"🔄 平均角度变化: {avg_angle_change:.2f}°")
    print(f"🏔️ 接地时间比例: {grounded_ratio:.1%}")
    print(f"⏱️ 响应延迟比例: {response_delay_ratio:.1%}")
    
    return {
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'speed_std': speed_std,
        'hammer_angle_range': hammer_angle_range,
        'avg_angular_vel': avg_angular_vel,
        'max_angular_vel': max_angular_vel,
        'avg_position_change': avg_position_change,
        'avg_velocity_change': avg_velocity_change,
        'avg_angle_change': avg_angle_change,
        'grounded_ratio': grounded_ratio,
        'response_delay_ratio': response_delay_ratio
    }

def analyze_mouse_physics_correlation(data):
    """分析鼠标移动与物理响应的相关性"""
    print("\n🔗 === 鼠标-物理响应关联分析 ===")
    
    # 1. 直接相关性分析
    mouse_move_distance = data['mouseMoveDistance']
    mouse_move_speed = data['mouseMoveSpeed']
    player_speed = np.sqrt(data['velocityX']**2 + data['velocityY']**2)
    hammer_angular_vel = data['hammerAngularVel'].abs()
    
    # 计算相关系数
    corr_move_distance_speed, _ = pearsonr(mouse_move_distance, player_speed)
    corr_move_speed_speed, _ = pearsonr(mouse_move_speed, player_speed)
    corr_move_speed_angular, _ = pearsonr(mouse_move_speed, hammer_angular_vel)
    
    # 2. 方向性分析
    mouse_delta_x = data['mouseDeltaX']
    mouse_delta_y = data['mouseDeltaY']
    velocity_x = data['velocityX']
    velocity_y = data['velocityY']
    
    corr_x_direction, _ = pearsonr(mouse_delta_x, velocity_x)
    corr_y_direction, _ = pearsonr(mouse_delta_y, velocity_y)
    
    # 3. 响应延迟分析
    # 分析鼠标移动与后续物理响应的相关性
    max_delay = 10  # 最大延迟帧数
    best_correlation = 0
    best_delay = 0
    
    for delay in range(0, max_delay + 1):
        if delay == 0:
            corr = pearsonr(mouse_move_speed, player_speed)[0]
        else:
            if len(mouse_move_speed) > delay:
                corr = pearsonr(mouse_move_speed[:-delay], player_speed[delay:])[0]
            else:
                corr = 0
        
        if not np.isnan(corr) and abs(corr) > abs(best_correlation):
            best_correlation = corr
            best_delay = delay
    
    print(f"🖱️ 鼠标移动距离 ↔ 角色速度: {corr_move_distance_speed:.3f}")
    print(f"⚡ 鼠标移动速度 ↔ 角色速度: {corr_move_speed_speed:.3f}")
    print(f"🌪️ 鼠标移动速度 ↔ 锤子角速度: {corr_move_speed_angular:.3f}")
    print(f"↔️ 水平方向控制效率: {corr_x_direction:.3f}")
    print(f"↕️ 垂直方向控制效率: {corr_y_direction:.3f}")
    print(f"⏱️ 最佳响应延迟: {best_delay}帧 (相关性: {best_correlation:.3f})")
    
    return {
        'corr_move_distance_speed': corr_move_distance_speed,
        'corr_move_speed_speed': corr_move_speed_speed,
        'corr_move_speed_angular': corr_move_speed_angular,
        'corr_x_direction': corr_x_direction,
        'corr_y_direction': corr_y_direction,
        'best_delay': best_delay,
        'best_correlation': best_correlation
    }

def analyze_control_efficiency(data, mouse_results, physics_results, correlation_results):
    """分析控制效率"""
    print("\n🎯 === 控制效率分析 ===")
    
    # 1. 控制精度评估
    control_precision = (abs(correlation_results['corr_x_direction']) + 
                        abs(correlation_results['corr_y_direction'])) / 2
    
    # 2. 响应效率评估
    response_efficiency = abs(correlation_results['corr_move_speed_speed'])
    
    # 3. 操作流畅度评估
    # 基于速度变化的平滑度
    velocity_smoothness = 1 - (physics_results['avg_velocity_change'] / physics_results['max_speed'])
    velocity_smoothness = max(0, min(1, velocity_smoothness))
    
    # 4. 策略效率评估
    # 基于鼠标移动与物理响应的匹配度
    strategy_efficiency = (abs(correlation_results['corr_move_distance_speed']) + 
                          abs(correlation_results['corr_move_speed_angular'])) / 2
    
    # 5. 综合控制评分
    overall_score = (control_precision * 0.3 + 
                    response_efficiency * 0.3 + 
                    velocity_smoothness * 0.2 + 
                    strategy_efficiency * 0.2)
    
    print(f"🎯 控制精度: {'优秀' if control_precision > 0.7 else '良好' if control_precision > 0.4 else '一般' if control_precision > 0.2 else '较差'} ({control_precision:.3f})")
    print(f"⚡ 响应效率: {'优秀' if response_efficiency > 0.7 else '良好' if response_efficiency > 0.4 else '一般' if response_efficiency > 0.2 else '较差'} ({response_efficiency:.3f})")
    print(f"🌊 操作流畅度: {'优秀' if velocity_smoothness > 0.8 else '良好' if velocity_smoothness > 0.6 else '一般' if velocity_smoothness > 0.4 else '较差'} ({velocity_smoothness:.3f})")
    print(f"🧠 策略效率: {'优秀' if strategy_efficiency > 0.7 else '良好' if strategy_efficiency > 0.4 else '一般' if strategy_efficiency > 0.2 else '较差'} ({strategy_efficiency:.3f})")
    print(f"🏆 综合控制评分: {overall_score:.3f} ({'优秀' if overall_score > 0.7 else '良好' if overall_score > 0.4 else '一般' if overall_score > 0.2 else '较差'})")
    
    return {
        'control_precision': control_precision,
        'response_efficiency': response_efficiency,
        'velocity_smoothness': velocity_smoothness,
        'strategy_efficiency': strategy_efficiency,
        'overall_score': overall_score
    }

def generate_improvement_suggestions(mouse_results, physics_results, correlation_results, efficiency_results):
    """生成改进建议"""
    print("\n💡 === 改进建议 ===")
    
    suggestions = []
    
    # 基于鼠标移动模式的建议
    if mouse_results['stationary_ratio'] > 0.5:
        suggestions.append("• 增加鼠标移动频率，保持持续的微调控制")
    
    if mouse_results['max_move_speed'] < 10:
        suggestions.append("• 练习快速鼠标移动，提高操作强度")
    
    if mouse_results['dominant_direction'] == "水平" and mouse_results['mouse_range_y'] < 100:
        suggestions.append("• 加强垂直方向的控制练习")
    elif mouse_results['dominant_direction'] == "垂直" and mouse_results['mouse_range_x'] < 100:
        suggestions.append("• 加强水平方向的控制练习")
    
    # 基于物理响应的建议
    if physics_results['response_delay_ratio'] > 0.3:
        suggestions.append("• 优化输入时机，减少响应延迟")
    
    if physics_results['avg_velocity_change'] > 5:
        suggestions.append("• 练习平滑操作，减少急速变化")
    
    if physics_results['grounded_ratio'] < 0.2:
        suggestions.append("• 提高接地控制，增加稳定性")
    
    # 基于控制效率的建议
    if efficiency_results['control_precision'] < 0.3:
        suggestions.append("• 提高鼠标精度设置，练习精确控制")
    
    if efficiency_results['response_efficiency'] < 0.3:
        suggestions.append("• 优化鼠标移动与角色动作的匹配度")
    
    if efficiency_results['velocity_smoothness'] < 0.5:
        suggestions.append("• 练习连续平滑的鼠标移动")
    
    if not suggestions:
        suggestions.append("• 控制表现良好，继续保持当前操作风格")
    
    for suggestion in suggestions:
        print(suggestion)

def analyze_sampling_frequency(data):
    """分析采样频率"""
    print("\n📊 === 采样频率分析 ===")
    
    # 1. 计算实际采样频率
    if len(data) > 1:
        duration = data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]
        actual_freq = len(data) / duration
        
        # 2. 分析deltaTime分布
        delta_times = data['deltaTime']
        avg_delta_time = delta_times.mean()
        min_delta_time = delta_times.min()
        max_delta_time = delta_times.max()
        std_delta_time = delta_times.std()
        
        # 3. 计算理论vs实际频率
        theoretical_freq = 1.0 / avg_delta_time if avg_delta_time > 0 else 0
        
        print(f"⏱️ 记录时长: {duration:.2f} 秒")
        print(f"📊 数据点数: {len(data)}")
        print(f"⚡ 实际采样频率: {actual_freq:.1f} Hz")
        print(f"🎯 理论采样频率: {theoretical_freq:.1f} Hz")
        print(f"📈 平均帧间隔: {avg_delta_time*1000:.1f} ms")
        print(f"📉 最小帧间隔: {min_delta_time*1000:.1f} ms")
        print(f"📊 最大帧间隔: {max_delta_time*1000:.1f} ms")
        print(f"📊 帧间隔标准差: {std_delta_time*1000:.1f} ms")
        
        # 4. 频率稳定性分析
        frequency_stability = 1 - (std_delta_time / avg_delta_time) if avg_delta_time > 0 else 0
        stability_rating = "优秀" if frequency_stability > 0.95 else "良好" if frequency_stability > 0.9 else "一般" if frequency_stability > 0.8 else "较差"
        
        print(f"🎯 频率稳定性: {stability_rating} ({frequency_stability:.3f})")
        
        return {
            'duration': duration,
            'data_points': len(data),
            'actual_freq': actual_freq,
            'theoretical_freq': theoretical_freq,
            'avg_delta_time': avg_delta_time,
            'frequency_stability': frequency_stability
        }
    else:
        print("❌ 数据不足，无法分析采样频率")
        return None

def main():
    """主函数"""
    print("🚀 === Getting Over It 鼠标移动-物理响应分析 ===")
    
    # 自动查找最新的数据文件
    csv_files = glob.glob("../Data/ContinuousTracking_*.csv")
    
    if not csv_files:
        print("❌ 未找到数据文件")
        print("💡 请确保CSV文件在 ../Data/ 目录中")
        return
    
    # 选择最新的文件
    latest_csv = max(csv_files, key=os.path.getctime)
    csv_file = os.path.basename(latest_csv)
    
    print(f"📁 分析文件: {csv_file}")
    
    # 1. 加载数据
    data = load_mouse_physics_data(latest_csv)
    if data is None:
        return
    
    # 2. 执行各项分析
    sampling_results = analyze_sampling_frequency(data)
    mouse_results = analyze_mouse_movement_patterns(data)
    physics_results = analyze_physics_response(data)
    correlation_results = analyze_mouse_physics_correlation(data)
    efficiency_results = analyze_control_efficiency(data, mouse_results, physics_results, correlation_results)
    
    # 3. 生成改进建议
    generate_improvement_suggestions(mouse_results, physics_results, correlation_results, efficiency_results)
    
    print("\n" + "="*80)
    print("✅ 鼠标移动-物理响应分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()

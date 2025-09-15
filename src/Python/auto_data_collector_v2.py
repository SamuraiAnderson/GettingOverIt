#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Getting Over It 自动化数据采集器 V2
专注于数据集保存，不使用SQLite数据库
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AutoDataCollectorV2:
    """自动化数据采集器 V2 - 专注于数据集保存"""
    
    def __init__(self, output_dir: str = "src/Data/AI_Training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据采集参数
        self.collection_duration = 5  # 秒
        self.collection_frequency = 30  # Hz
        self.expected_data_points = 0
        
        # 数据采集状态
        self.is_collecting = False
        self.collected_data = []
        self.session_id = 0
        
        print("🤖 自动化数据采集器 V2 初始化完成")
    
    def start_collection(self, duration: int, frequency: int) -> int:
        """开始数据采集会话"""
        self.collection_duration = duration
        self.collection_frequency = frequency
        self.expected_data_points = duration * frequency
        self.session_id = int(time.time())
        
        print(f"🔄 开始数据采集会话 #{self.session_id}")
        print(f"   - 采集时长: {duration}秒")
        print(f"   - 采集频率: {frequency}Hz")
        print(f"   - 预计数据点: {self.expected_data_points}")
        
        return self.session_id
    
    def collect_data_from_unity(self, session_id: int) -> bool:
        """从Unity插件采集数据"""
        print("📊 开始从Unity插件采集数据...")
        
        # 1. 启动自动化模式
        if not self._start_auto_mode(session_id):
            print("❌ 启动自动化模式失败")
            return False
        
        # 2. 等待Unity插件完成自动流程
        if not self._wait_for_auto_mode_completion():
            print("❌ 自动化模式执行超时")
            return False
        
        # 3. 等待采集完成
        print(f"⏳ 等待数据采集完成 ({self.collection_duration}秒)...")
        time.sleep(self.collection_duration)
        
        # 4. 停止自动化模式
        self._stop_auto_mode()
        
        # 5. 读取Unity插件导出的数据
        return self._load_and_convert_unity_data(session_id)
    
    def _start_auto_mode(self, session_id: int) -> bool:
        """启动Unity插件自动化模式"""
        signal_file = Path("src/Data/auto_mode_signal.json")
        signal_data = {
            "enable_auto_mode": True,
            "duration": self.collection_duration,
            "frequency": self.collection_frequency,
            "timestamp": time.time()
        }
        
        with open(signal_file, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, indent=2)
        
        print(f"📤 启动Unity插件自动化模式")
        return True
    
    def _stop_auto_mode(self):
        """停止Unity插件自动化模式"""
        signal_file = Path("src/Data/auto_mode_signal.json")
        signal_data = {
            "enable_auto_mode": False,
            "duration": 0,
            "frequency": 0,
            "timestamp": time.time()
        }
        
        with open(signal_file, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, indent=2)
        
        print(f"📤 停止Unity插件自动化模式")
    
    def _wait_for_auto_mode_completion(self, timeout: int = 60) -> bool:
        """等待自动化模式完成"""
        response_file = Path("src/Data/auto_mode_response.json")
        start_time = time.time()
        
        print("⏳ 等待Unity插件完成自动流程...")
        
        while time.time() - start_time < timeout:
            if response_file.exists():
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)
                    
                    if response_data.get("response") == "data_collection_started":
                        print(f"✅ Unity插件自动化流程完成，开始数据采集")
                        return True
                except (json.JSONDecodeError, KeyError):
                    pass
            
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            print(f"   等待中... {elapsed}/{timeout}秒")
        
        return False
    
    def _send_collection_signal(self, signal: str, session_id: int):
        """向Unity插件发送采集信号（保留兼容性）"""
        signal_file = Path("src/Data/collection_signal.json")
        signal_data = {
            "signal": signal,
            "session_id": session_id,
            "duration": self.collection_duration,
            "frequency": self.collection_frequency,
            "timestamp": time.time()
        }
        
        with open(signal_file, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, indent=2)
        
        print(f"📤 发送信号到Unity插件: {signal}")
    
    def _wait_for_unity_response(self, expected_response: str, timeout: int = 10) -> bool:
        """等待Unity插件响应"""
        response_file = Path("src/Data/unity_response.json")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if response_file.exists():
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)
                    
                    if response_data.get("response") == expected_response:
                        print(f"✅ Unity插件响应: {expected_response}")
                        return True
                except (json.JSONDecodeError, KeyError):
                    pass
            
            time.sleep(0.5)
        
        return False
    
    def _load_and_convert_unity_data(self, session_id: int) -> bool:
        """加载并转换Unity插件导出的数据"""
        # 查找最新的数据文件
        data_dir = Path("src/Data")
        csv_files = list(data_dir.glob("ContinuousTracking_*.csv"))
        
        if not csv_files:
            print("❌ 未找到Unity插件导出的数据文件")
            return False
        
        # 选择最新的文件
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        print(f"📁 加载数据文件: {latest_file.name}")
        
        try:
            # 读取CSV数据
            raw_data = pd.read_csv(latest_file, skiprows=13)
            
            # 验证数据质量
            if len(raw_data) < self.expected_data_points * 0.8:  # 允许20%的数据丢失
                print(f"⚠️ 数据点不足: 期望{self.expected_data_points}, 实际{len(raw_data)}")
            
            # 转换为AI训练格式
            self._convert_to_ai_format(session_id, raw_data)
            
            print(f"✅ 数据采集完成: {len(raw_data)}个数据点")
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def _convert_to_ai_format(self, session_id: int, raw_data: pd.DataFrame):
        """转换为AI训练格式"""
        print("🔄 转换数据为AI训练格式...")
        
        # 提取观察空间 (7维)
        observation_features = ['playerX', 'playerY', 'velocityX', 'velocityY', 
                              'hammerAngle', 'hammerAngularVel', 'isGrounded']
        observations = raw_data[observation_features].values
        
        # 提取动作空间 (2维)
        action_features = ['mouseDeltaX', 'mouseDeltaY']
        actions = raw_data[action_features].values
        
        # 数据归一化
        observations = self._normalize_data(observations, observation_features)
        actions = self._normalize_data(actions, action_features)
        
        # 创建时间序列
        obs_sequences, action_sequences = self._create_sequences(observations, actions)
        
        # 计算奖励
        rewards = self._calculate_rewards(raw_data)
        
        # 保存转换后的数据
        self._save_ai_training_data(session_id, obs_sequences, action_sequences, rewards, raw_data)
    
    def _normalize_data(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """数据归一化到[-1, 1]范围"""
        normalized_data = data.copy()
        
        # 归一化范围定义
        normalization_ranges = {
            'playerX': (-100, 100),
            'playerY': (-100, 100), 
            'velocityX': (-50, 50),
            'velocityY': (-50, 50),
            'hammerAngle': (-180, 180),
            'hammerAngularVel': (-1000, 1000),
            'isGrounded': (0, 1),
            'mouseDeltaX': (-10, 10),
            'mouseDeltaY': (-10, 10)
        }
        
        for i, feature in enumerate(feature_names):
            if feature in normalization_ranges:
                min_val, max_val = normalization_ranges[feature]
                # 归一化到[-1, 1]
                normalized_data[:, i] = 2 * (data[:, i] - min_val) / (max_val - min_val) - 1
                # 限制在[-1, 1]范围内
                normalized_data[:, i] = np.clip(normalized_data[:, i], -1, 1)
        
        return normalized_data
    
    def _create_sequences(self, observations: np.ndarray, actions: np.ndarray, 
                         sequence_length: int = 10, overlap: int = 5) -> tuple:
        """创建时间序列数据"""
        if len(observations) < sequence_length:
            raise ValueError(f"数据长度不足: 需要至少{sequence_length}个数据点")
        
        # 计算序列数量
        step_size = sequence_length - overlap
        num_sequences = (len(observations) - sequence_length) // step_size + 1
        
        obs_sequences = np.zeros((num_sequences, sequence_length, observations.shape[1]))
        action_sequences = np.zeros((num_sequences, sequence_length, actions.shape[1]))
        
        for i in range(num_sequences):
            start_idx = i * step_size
            end_idx = start_idx + sequence_length
            
            obs_sequences[i] = observations[start_idx:end_idx]
            action_sequences[i] = actions[start_idx:end_idx]
        
        print(f"📈 序列数据创建完成:")
        print(f"   - 序列数量: {num_sequences}")
        print(f"   - 序列长度: {sequence_length}")
        print(f"   - 观察序列形状: {obs_sequences.shape}")
        print(f"   - 动作序列形状: {action_sequences.shape}")
        
        return obs_sequences, action_sequences
    
    def _calculate_rewards(self, raw_data: pd.DataFrame) -> np.ndarray:
        """计算奖励信号"""
        rewards = np.zeros(len(raw_data))
        
        # 基于物理响应的奖励设计
        if 'positionDelta' in raw_data.columns:
            position_delta = raw_data['positionDelta'].values
        else:
            position_delta = np.zeros(len(raw_data))
        
        if 'velocityDelta' in raw_data.columns:
            velocity_delta = raw_data['velocityDelta'].values
        else:
            velocity_delta = np.zeros(len(raw_data))
        
        if 'angleDelta' in raw_data.columns:
            angle_delta = raw_data['angleDelta'].values
        else:
            angle_delta = np.zeros(len(raw_data))
        
        if 'physicsResponseDelay' in raw_data.columns:
            response_delay = raw_data['physicsResponseDelay'].values
        else:
            response_delay = np.zeros(len(raw_data))
        
        # 奖励函数设计
        progress_reward = position_delta * 1.0  # 位置进步奖励
        stability_penalty = -velocity_delta * 0.5  # 速度稳定性惩罚
        control_penalty = -np.abs(angle_delta) * 0.3  # 控制精度惩罚
        efficiency_bonus = (1 - response_delay) * 0.2  # 响应效率奖励
        
        rewards = progress_reward + stability_penalty + control_penalty + efficiency_bonus
        
        print(f"🏆 奖励计算完成:")
        print(f"   - 奖励范围: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"   - 平均奖励: {rewards.mean():.3f}")
        
        return rewards
    
    def _save_ai_training_data(self, session_id: int, obs_sequences: np.ndarray, 
                              action_sequences: np.ndarray, rewards: np.ndarray, 
                              raw_data: pd.DataFrame):
        """保存AI训练数据"""
        # 保存为numpy格式
        npz_file = self.output_dir / f"session_{session_id}_ai_training.npz"
        np.savez_compressed(
            npz_file,
            observation_sequences=obs_sequences,
            action_sequences=action_sequences,
            rewards=rewards
        )
        
        # 保存元数据
        metadata = {
            "session_id": session_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration": self.collection_duration,
            "frequency": self.collection_frequency,
            "raw_data_points": len(raw_data),
            "sequence_count": len(obs_sequences),
            "observation_features": ['playerX', 'playerY', 'velocityX', 'velocityY', 
                                   'hammerAngle', 'hammerAngularVel', 'isGrounded'],
            "action_features": ['mouseDeltaX', 'mouseDeltaY'],
            "sequence_length": 10,
            "overlap": 5
        }
        
        metadata_file = self.output_dir / f"session_{session_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 AI训练数据已保存:")
        print(f"   - 数据文件: {npz_file}")
        print(f"   - 元数据: {metadata_file}")
    
    def update_unified_dataset(self):
        """更新统一训练数据集"""
        print("📊 更新统一训练数据集...")
        
        # 查找所有会话数据文件
        session_files = list(self.output_dir.glob("session_*_ai_training.npz"))
        
        if not session_files:
            print("⚠️ 没有找到会话数据文件")
            return
        
        all_obs_sequences = []
        all_action_sequences = []
        all_rewards = []
        
        for file_path in session_files:
            data = np.load(file_path)
            all_obs_sequences.append(data['observation_sequences'])
            all_action_sequences.append(data['action_sequences'])
            all_rewards.append(data['rewards'])
        
        # 合并所有数据
        combined_obs = np.concatenate(all_obs_sequences, axis=0)
        combined_actions = np.concatenate(all_action_sequences, axis=0)
        combined_rewards = np.concatenate(all_rewards, axis=0)
        
        # 保存统一数据集
        dataset_file = self.output_dir / "unified_training_dataset.npz"
        np.savez_compressed(
            dataset_file,
            observation_sequences=combined_obs,
            action_sequences=combined_actions,
            rewards=combined_rewards
        )
        
        print(f"✅ 统一训练数据集已更新:")
        print(f"   - 数据集文件: {dataset_file}")
        print(f"   - 总序列数: {len(combined_obs)}")
        print(f"   - 观察空间形状: {combined_obs.shape}")
        print(f"   - 动作空间形状: {combined_actions.shape}")
        print(f"   - 奖励数据形状: {combined_rewards.shape}")
    
    def run_auto_collection(self, duration: int, frequency: int) -> bool:
        """运行自动化数据采集"""
        try:
            # 开始采集会话
            session_id = self.start_collection(duration, frequency)
            
            # 从Unity插件采集数据
            success = self.collect_data_from_unity(session_id)
            
            if success:
                # 更新统一数据集
                self.update_unified_dataset()
                
                print(f"\n🎉 自动化数据采集完成!")
                print(f"📁 输出目录: {self.output_dir}")
                print(f"📊 会话ID: #{session_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ 自动化采集失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Getting Over It 自动化数据采集器 V2')
    parser.add_argument('--duration', type=int, default=5, help='采集时长(秒)')
    parser.add_argument('--frequency', type=int, default=30, help='采集频率(Hz)')
    parser.add_argument('--output-dir', type=str, default='src/Data/AI_Training', help='输出目录')
    
    args = parser.parse_args()
    
    print("🚀 Getting Over It 自动化数据采集器 V2")
    print("=" * 50)
    print(f"⏱️ 采集时长: {args.duration}秒")
    print(f"📊 采集频率: {args.frequency}Hz")
    print(f"📁 输出目录: {args.output_dir}")
    print()
    
    # 创建采集器并运行
    collector = AutoDataCollectorV2(args.output_dir)
    success = collector.run_auto_collection(args.duration, args.frequency)
    
    if success:
        print("\n🎉 自动化数据采集完成!")
        exit(0)
    else:
        print("\n❌ 自动化数据采集失败!")
        exit(1)

if __name__ == "__main__":
    main()

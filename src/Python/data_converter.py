#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Getting Over It AI训练数据转换器
将Unity ContinuousTracker采集的原始数据转换为AI训练格式
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DataConverter:
    """数据转换器 - 将原始数据转换为AI训练格式"""
    
    def __init__(self, data_dir: str = "src/Data", output_dir: str = "src/Data/AI_Training"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # AI训练数据格式定义
        self.observation_space = {
            'player_position': ['playerX', 'playerY'],
            'player_velocity': ['velocityX', 'velocityY'], 
            'hammer_state': ['hammerAngle', 'hammerAngularVel'],
            'environment_state': ['isGrounded']
        }
        
        self.action_space = {
            'mouse_movement': ['mouseDeltaX', 'mouseDeltaY']
        }
        
        # 数据归一化范围
        self.normalization_ranges = {
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
        
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """加载Unity采集的原始数据"""
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        print(f"📊 加载原始数据: {filename}")
        
        # 检测数据格式并跳过相应的头部行
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找数据开始行
        data_start_line = 0
        for i, line in enumerate(lines):
            if line.startswith('timestamp,'):
                data_start_line = i
                break
        
        if data_start_line == 0:
            raise ValueError("无法找到数据开始行")
        
        # 从数据开始行读取CSV
        raw_data = pd.read_csv(file_path, skiprows=data_start_line)
        
        print(f"✅ 原始数据加载完成:")
        print(f"   - 数据点数: {len(raw_data)}")
        if 'timestamp' in raw_data.columns:
            print(f"   - 时间跨度: {raw_data['timestamp'].iloc[-1] - raw_data['timestamp'].iloc[0]:.2f}秒")
        
        return raw_data
    
    def extract_observation_space(self, raw_data: pd.DataFrame) -> np.ndarray:
        """提取观察空间数据 (7维)"""
        observation_features = []
        
        # 按顺序提取观察特征
        for category, features in self.observation_space.items():
            for feature in features:
                if feature in raw_data.columns:
                    observation_features.append(feature)
        
        # 检查是否有足够的特征
        if len(observation_features) < 6:
            raise ValueError(f"观察空间特征不足: 需要至少6个，实际{len(observation_features)}个")
        
        # 如果特征超过7个，只取前7个
        if len(observation_features) > 7:
            observation_features = observation_features[:7]
            print(f"⚠️ 观察空间特征过多，只取前7个: {observation_features}")
        
        observations = raw_data[observation_features].values
        print(f"📊 观察空间提取完成: {observations.shape}")
        print(f"   - 特征列表: {observation_features}")
        
        return observations
    
    def extract_action_space(self, raw_data: pd.DataFrame) -> np.ndarray:
        """提取动作空间数据 (2维)"""
        action_features = []
        
        for category, features in self.action_space.items():
            for feature in features:
                if feature in raw_data.columns:
                    action_features.append(feature)
        
        if len(action_features) != 2:
            raise ValueError(f"动作空间特征数量不匹配: 期望2个，实际{len(action_features)}个")
        
        actions = raw_data[action_features].values
        print(f"🎯 动作空间提取完成: {actions.shape}")
        
        return actions
    
    def normalize_data(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """数据归一化到[-1, 1]范围"""
        normalized_data = data.copy()
        
        for i, feature in enumerate(feature_names):
            if feature in self.normalization_ranges:
                min_val, max_val = self.normalization_ranges[feature]
                # 归一化到[-1, 1]
                normalized_data[:, i] = 2 * (data[:, i] - min_val) / (max_val - min_val) - 1
                # 限制在[-1, 1]范围内
                normalized_data[:, i] = np.clip(normalized_data[:, i], -1, 1)
        
        return normalized_data
    
    def create_sequences(self, observations: np.ndarray, actions: np.ndarray, 
                        sequence_length: int = 10, overlap: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def calculate_rewards(self, raw_data: pd.DataFrame) -> np.ndarray:
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
    
    def convert_single_file(self, filename: str) -> Dict[str, Any]:
        """转换单个数据文件"""
        print(f"\n🔄 开始转换文件: {filename}")
        
        # 1. 加载原始数据
        raw_data = self.load_raw_data(filename)
        
        # 2. 提取观察空间和动作空间
        observations = self.extract_observation_space(raw_data)
        actions = self.extract_action_space(raw_data)
        
        # 3. 数据归一化
        obs_features = [f for features in self.observation_space.values() for f in features]
        action_features = [f for features in self.action_space.values() for f in features]
        
        normalized_obs = self.normalize_data(observations, obs_features)
        normalized_actions = self.normalize_data(actions, action_features)
        
        # 4. 创建时间序列
        obs_sequences, action_sequences = self.create_sequences(normalized_obs, normalized_actions)
        
        # 5. 计算奖励
        rewards = self.calculate_rewards(raw_data)
        
        # 6. 保存转换后的数据
        output_data = {
            'filename': filename,
            'raw_data_points': len(raw_data),
            'sequence_count': len(obs_sequences),
            'observation_sequences': obs_sequences,
            'action_sequences': action_sequences,
            'rewards': rewards,
            'metadata': {
                'observation_features': obs_features,
                'action_features': action_features,
                'sequence_length': 10,
                'overlap': 5,
                'normalization_ranges': self.normalization_ranges
            }
        }
        
        return output_data
    
    def save_converted_data(self, converted_data: Dict[str, Any]) -> str:
        """保存转换后的数据"""
        filename = converted_data['filename']
        base_name = Path(filename).stem
        
        # 保存为numpy格式
        npz_file = self.output_dir / f"{base_name}_ai_training.npz"
        np.savez_compressed(
            npz_file,
            observation_sequences=converted_data['observation_sequences'],
            action_sequences=converted_data['action_sequences'],
            rewards=converted_data['rewards']
        )
        
        # 保存元数据
        metadata_file = self.output_dir / f"{base_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data['metadata'], f, indent=2, ensure_ascii=False)
        
        print(f"💾 转换数据已保存:")
        print(f"   - 数据文件: {npz_file}")
        print(f"   - 元数据: {metadata_file}")
        
        return str(npz_file)
    
    def batch_convert(self, file_pattern: str = "ContinuousTracking_*.csv") -> List[str]:
        """批量转换数据文件"""
        print(f"🔄 开始批量转换数据文件...")
        
        # 查找匹配的文件
        data_files = list(self.data_dir.glob(file_pattern))
        
        if not data_files:
            print(f"⚠️ 未找到匹配的文件: {file_pattern}")
            return []
        
        print(f"📁 找到 {len(data_files)} 个数据文件")
        
        converted_files = []
        
        for data_file in data_files:
            try:
                # 转换单个文件
                converted_data = self.convert_single_file(data_file.name)
                
                # 保存转换后的数据
                output_file = self.save_converted_data(converted_data)
                converted_files.append(output_file)
                
            except Exception as e:
                print(f"❌ 转换文件失败 {data_file.name}: {e}")
                continue
        
        print(f"\n✅ 批量转换完成:")
        print(f"   - 成功转换: {len(converted_files)} 个文件")
        print(f"   - 输出目录: {self.output_dir}")
        
        return converted_files
    
    def create_training_dataset(self, converted_files: List[str]) -> str:
        """创建统一的训练数据集"""
        if not converted_files:
            raise ValueError("没有转换后的数据文件")
        
        print(f"📊 创建统一训练数据集...")
        
        all_obs_sequences = []
        all_action_sequences = []
        all_rewards = []
        
        for file_path in converted_files:
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
        
        print(f"✅ 统一训练数据集创建完成:")
        print(f"   - 数据集文件: {dataset_file}")
        print(f"   - 总序列数: {len(combined_obs)}")
        print(f"   - 观察空间形状: {combined_obs.shape}")
        print(f"   - 动作空间形状: {combined_actions.shape}")
        print(f"   - 奖励数据形状: {combined_rewards.shape}")
        
        return str(dataset_file)

def main():
    """主函数"""
    print("🚀 Getting Over It AI训练数据转换器")
    print("=" * 50)
    
    # 创建转换器
    converter = DataConverter()
    
    try:
        # 批量转换数据文件
        converted_files = converter.batch_convert()
        
        if converted_files:
            # 创建统一训练数据集
            dataset_file = converter.create_training_dataset(converted_files)
            
            print(f"\n🎉 数据转换完成!")
            print(f"📁 输出目录: {converter.output_dir}")
            print(f"📊 统一数据集: {dataset_file}")
            
        else:
            print("⚠️ 没有成功转换的数据文件")
            
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()

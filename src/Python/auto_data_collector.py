#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Getting Over It 自动化数据采集器
与Unity插件通信，控制数据采集流程
"""

import argparse
import time
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AutoDataCollector:
    """自动化数据采集器"""
    
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
        
        print("🤖 自动化数据采集器初始化完成")
    
    def setup_database(self):
        """设置数据库表结构"""
        self.db_connection = sqlite3.connect(self.db_path)
        cursor = self.db_connection.cursor()
        
        # 创建数据采集会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                duration REAL NOT NULL,
                frequency INTEGER NOT NULL,
                data_points INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建训练数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                delta_time REAL NOT NULL,
                mouse_x REAL NOT NULL,
                mouse_y REAL NOT NULL,
                player_x REAL NOT NULL,
                player_y REAL NOT NULL,
                velocity_x REAL NOT NULL,
                velocity_y REAL NOT NULL,
                hammer_angle REAL NOT NULL,
                hammer_angular_vel REAL NOT NULL,
                is_grounded INTEGER NOT NULL,
                mouse_delta_x REAL NOT NULL,
                mouse_delta_y REAL NOT NULL,
                mouse_move_distance REAL NOT NULL,
                mouse_move_speed REAL NOT NULL,
                position_delta REAL NOT NULL,
                velocity_delta REAL NOT NULL,
                angle_delta REAL NOT NULL,
                physics_response_delay REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES collection_sessions (session_id)
            )
        ''')
        
        self.db_connection.commit()
        print("✅ 数据库表结构设置完成")
    
    def start_collection(self, duration: int, frequency: int) -> int:
        """开始数据采集会话"""
        self.collection_duration = duration
        self.collection_frequency = frequency
        self.expected_data_points = duration * frequency
        
        # 创建新的采集会话
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO collection_sessions (timestamp, duration, frequency, data_points, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (time.strftime('%Y-%m-%d %H:%M:%S'), duration, frequency, 0, 'started'))
        
        session_id = cursor.lastrowid
        self.db_connection.commit()
        
        print(f"🔄 开始数据采集会话 #{session_id}")
        print(f"   - 采集时长: {duration}秒")
        print(f"   - 采集频率: {frequency}Hz")
        print(f"   - 预计数据点: {self.expected_data_points}")
        
        return session_id
    
    def collect_data_from_unity(self, session_id: int) -> bool:
        """从Unity插件采集数据"""
        print("📊 开始从Unity插件采集数据...")
        
        # 这里需要与Unity插件通信
        # 由于Unity插件是独立的，我们通过文件系统进行通信
        
        # 1. 发送采集开始信号
        self._send_collection_signal("start", session_id)
        
        # 2. 等待Unity插件开始采集
        if not self._wait_for_unity_response("started", timeout=10):
            print("❌ Unity插件响应超时")
            return False
        
        # 3. 等待采集完成
        print(f"⏳ 等待数据采集完成 ({self.collection_duration}秒)...")
        time.sleep(self.collection_duration)
        
        # 4. 发送采集停止信号
        self._send_collection_signal("stop", session_id)
        
        # 5. 等待Unity插件完成数据导出
        if not self._wait_for_unity_response("completed", timeout=30):
            print("❌ Unity插件数据导出超时")
            return False
        
        # 6. 读取Unity插件导出的数据
        return self._load_unity_data(session_id)
    
    def _send_collection_signal(self, signal: str, session_id: int):
        """向Unity插件发送采集信号"""
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
    
    def _load_unity_data(self, session_id: int) -> bool:
        """加载Unity插件导出的数据"""
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
            
            # 保存数据到数据库
            self._save_data_to_database(session_id, raw_data)
            
            # 更新会话状态
            self._update_session_status(session_id, "completed", len(raw_data))
            
            print(f"✅ 数据采集完成: {len(raw_data)}个数据点")
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            self._update_session_status(session_id, "failed", 0)
            return False
    
    def _save_data_to_database(self, session_id: int, data: pd.DataFrame):
        """保存数据到数据库"""
        cursor = self.db_connection.cursor()
        
        # 准备数据
        data_rows = []
        for _, row in data.iterrows():
            data_rows.append((
                session_id,
                row['timestamp'],
                row['deltaTime'],
                row['mouseX'],
                row['mouseY'],
                row['playerX'],
                row['playerY'],
                row['velocityX'],
                row['velocityY'],
                row['hammerAngle'],
                row['hammerAngularVel'],
                int(row['isGrounded']),
                row['mouseDeltaX'],
                row['mouseDeltaY'],
                row['mouseMoveDistance'],
                row['mouseMoveSpeed'],
                row['positionDelta'],
                row['velocityDelta'],
                row['angleDelta'],
                row['physicsResponseDelay']
            ))
        
        # 批量插入数据
        cursor.executemany('''
            INSERT INTO training_data (
                session_id, timestamp, delta_time, mouse_x, mouse_y,
                player_x, player_y, velocity_x, velocity_y, hammer_angle,
                hammer_angular_vel, is_grounded, mouse_delta_x, mouse_delta_y,
                mouse_move_distance, mouse_move_speed, position_delta,
                velocity_delta, angle_delta, physics_response_delay
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_rows)
        
        self.db_connection.commit()
        print(f"💾 数据已保存到数据库: {len(data_rows)}行")
    
    def _update_session_status(self, session_id: int, status: str, data_points: int):
        """更新会话状态"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            UPDATE collection_sessions 
            SET status = ?, data_points = ?
            WHERE session_id = ?
        ''', (status, data_points, session_id))
        
        self.db_connection.commit()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取采集统计信息"""
        cursor = self.db_connection.cursor()
        
        # 获取会话统计
        cursor.execute('''
            SELECT COUNT(*) as total_sessions,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_sessions,
                   SUM(data_points) as total_data_points
            FROM collection_sessions
        ''')
        
        session_stats = cursor.fetchone()
        
        # 获取最新会话信息
        cursor.execute('''
            SELECT session_id, timestamp, duration, frequency, data_points, status
            FROM collection_sessions
            ORDER BY session_id DESC
            LIMIT 1
        ''')
        
        latest_session = cursor.fetchone()
        
        return {
            "total_sessions": session_stats[0] or 0,
            "completed_sessions": session_stats[1] or 0,
            "total_data_points": session_stats[2] or 0,
            "latest_session": {
                "session_id": latest_session[0] if latest_session else None,
                "timestamp": latest_session[1] if latest_session else None,
                "duration": latest_session[2] if latest_session else None,
                "frequency": latest_session[3] if latest_session else None,
                "data_points": latest_session[4] if latest_session else None,
                "status": latest_session[5] if latest_session else None
            }
        }
    
    def run_auto_collection(self, duration: int, frequency: int) -> bool:
        """运行自动化数据采集"""
        try:
            # 设置数据库
            self.setup_database()
            
            # 开始采集会话
            session_id = self.start_collection(duration, frequency)
            
            # 从Unity插件采集数据
            success = self.collect_data_from_unity(session_id)
            
            if success:
                # 显示统计信息
                stats = self.get_collection_stats()
                print(f"\n📊 采集统计:")
                print(f"   - 总会话数: {stats['total_sessions']}")
                print(f"   - 完成会话数: {stats['completed_sessions']}")
                print(f"   - 总数据点数: {stats['total_data_points']}")
                print(f"   - 最新会话: #{stats['latest_session']['session_id']}")
                print(f"   - 最新状态: {stats['latest_session']['status']}")
            
            return success
            
        except Exception as e:
            print(f"❌ 自动化采集失败: {e}")
            return False
        
        finally:
            if self.db_connection:
                self.db_connection.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Getting Over It 自动化数据采集器')
    parser.add_argument('--duration', type=int, default=5, help='采集时长(秒)')
    parser.add_argument('--frequency', type=int, default=30, help='采集频率(Hz)')
    parser.add_argument('--db-path', type=str, default='src/Data/ai_training_database.db', help='数据库路径')
    
    args = parser.parse_args()
    
    print("🚀 Getting Over It 自动化数据采集器")
    print("=" * 50)
    print(f"⏱️ 采集时长: {args.duration}秒")
    print(f"📊 采集频率: {args.frequency}Hz")
    print(f"💾 数据库路径: {args.db_path}")
    print()
    
    # 创建采集器并运行
    collector = AutoDataCollector(args.db_path)
    success = collector.run_auto_collection(args.duration, args.frequency)
    
    if success:
        print("\n🎉 自动化数据采集完成!")
        exit(0)
    else:
        print("\n❌ 自动化数据采集失败!")
        exit(1)

if __name__ == "__main__":
    main()

"""
状态一致性验证测试
测试目标：验证环境的确定性和可复现性
测试方法：
1. 从同一初始状态开始
2. 执行相同的动作序列
3. 多次重复验证最终状态是否完全一致
4. 测试不同类型的一致性（位置、速度、角度等）
"""

import os
import sys
import time
import json
import random
import subprocess
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

class ConsistencyTester:
    def __init__(self, config_path="test_config.json"):
        """初始化测试器"""
        self.config = self.load_config(config_path)
        self.game_process = None
        self.test_results = []
        
    def load_config(self, config_path):
        """加载配置文件"""
        config_file = Path(__file__).parent.parent / config_path
        if not config_file.exists():
            return self.get_default_config()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_default_config(self):
        """默认配置"""
        return {
            "game_exe_path": "C:/Program Files/Steam/steamapps/common/Getting Over It/Getting Over It.exe",
            "game_data_dir": "C:/Program Files/Steam/steamapps/common/Getting Over It/GoiData",
            "num_trials": 5,  # 重复次数
            "action_duration": 3.0,  # 动作执行时长
            "consistency_threshold": 1e-5  # 一致性阈值
        }
    
    def get_signal_dir(self):
        return os.path.join(self.config['game_data_dir'], 'TestSignals')
    
    def get_state_dir(self):
        return os.path.join(self.config['game_data_dir'], 'GameStates')
    
    def get_result_dir(self):
        return os.path.join(self.config['game_data_dir'], 'TestResults')
    
    def ensure_directories(self):
        os.makedirs(self.get_signal_dir(), exist_ok=True)
        os.makedirs(self.get_state_dir(), exist_ok=True)
        os.makedirs(self.get_result_dir(), exist_ok=True)
    
    def start_game(self):
        """启动游戏"""
        print("🎮 启动游戏...")
        
        if not os.path.exists(self.config['game_exe_path']):
            print(f"❌ 游戏可执行文件不存在: {self.config['game_exe_path']}")
            return False
        
        try:
            self.game_process = subprocess.Popen([self.config['game_exe_path']])
            print(f"✅ 游戏已启动 (PID: {self.game_process.pid})")
            time.sleep(10)
            return True
        except Exception as e:
            print(f"❌ 启动游戏失败: {e}")
            return False
    
    def stop_game(self):
        """停止游戏"""
        if self.game_process:
            print("🛑 停止游戏...")
            try:
                self.game_process.terminate()
                self.game_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.game_process.kill()
            print("✅ 游戏已关闭")
    
    def send_signal(self, signal_file, content=""):
        """发送信号文件"""
        signal_path = os.path.join(self.get_signal_dir(), signal_file)
        with open(signal_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def wait_for_result(self, operation, timeout=5.0):
        """等待操作结果"""
        result_dir = self.get_result_dir()
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            success_file = os.path.join(result_dir, f"{operation}_success.result")
            if os.path.exists(success_file):
                with open(success_file, 'r') as f:
                    content = f.read()
                os.remove(success_file)
                return True, self.parse_result(content)
            
            failed_file = os.path.join(result_dir, f"{operation}_failed.result")
            if os.path.exists(failed_file):
                with open(failed_file, 'r') as f:
                    content = f.read()
                os.remove(failed_file)
                return False, self.parse_result(content)
            
            time.sleep(0.1)
        
        return False, {"error": "timeout"}
    
    def parse_result(self, content):
        """解析结果文件内容"""
        result = {}
        for line in content.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                result[key] = value
        return result
    
    def save_initial_state(self):
        """保存初始状态"""
        print("💾 保存初始状态...")
        self.send_signal("save_initial.signal")
        success, result = self.wait_for_result("save_initial")
        return success
    
    def reset_to_initial(self):
        """重置到初始状态"""
        self.send_signal("reset.signal")
        success, result = self.wait_for_result("reset")
        return success
    
    def export_current_state(self, filename):
        """导出当前状态"""
        self.send_signal("export_state.signal", filename)
        success, result = self.wait_for_result("export_state")
        
        if success:
            return os.path.join(self.get_state_dir(), filename)
        return None
    
    def compare_states(self, state_file):
        """比较当前状态与文件中的状态"""
        self.send_signal("compare_state.signal", os.path.basename(state_file))
        success, result = self.wait_for_result("compare_state")
        
        if success and 'data' in result:
            return float(result['data'])
        return None
    
    def load_state_from_file(self, filepath):
        """从文件加载状态数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载状态文件失败: {e}")
            return None
    
    def simulate_actions(self, duration):
        """模拟动作执行"""
        time.sleep(duration)
    
    def analyze_state_details(self, state_files):
        """详细分析多个状态文件的差异"""
        states = []
        for filepath in state_files:
            state = self.load_state_from_file(filepath)
            if state:
                states.append(state)
        
        if len(states) < 2:
            return None
        
        # 分析每个刚体的一致性
        rigidbody_consistency = {}
        
        if 'rigidbodyStates' not in states[0]:
            return None
        
        num_rigidbodies = len(states[0]['rigidbodyStates'])
        
        for rb_idx in range(num_rigidbodies):
            rb_name = states[0]['rigidbodyStates'][rb_idx]['name']
            
            # 收集所有trial的该刚体数据
            positions = []
            rotations = []
            velocities = []
            angular_velocities = []
            
            for state in states:
                if rb_idx < len(state['rigidbodyStates']):
                    rb = state['rigidbodyStates'][rb_idx]
                    positions.append([rb['position']['x'], rb['position']['y']])
                    rotations.append(rb['rotation'])
                    velocities.append([rb['velocity']['x'], rb['velocity']['y']])
                    angular_velocities.append(rb['angularVelocity'])
            
            # 计算标准差
            positions = np.array(positions)
            velocities = np.array(velocities)
            rotations = np.array(rotations)
            angular_velocities = np.array(angular_velocities)
            
            rigidbody_consistency[rb_name] = {
                'position_std': np.std(positions, axis=0).tolist() if len(positions) > 0 else [0, 0],
                'rotation_std': float(np.std(rotations)) if len(rotations) > 0 else 0.0,
                'velocity_std': np.std(velocities, axis=0).tolist() if len(velocities) > 0 else [0, 0],
                'angular_velocity_std': float(np.std(angular_velocities)) if len(angular_velocities) > 0 else 0.0,
                'position_range': (np.max(positions, axis=0) - np.min(positions, axis=0)).tolist() if len(positions) > 0 else [0, 0],
                'rotation_range': float(np.max(rotations) - np.min(rotations)) if len(rotations) > 0 else 0.0
            }
        
        return rigidbody_consistency
    
    def test_basic_consistency(self):
        """测试基本一致性"""
        print(f"\n{'='*60}")
        print(f"🔬 基本一致性测试")
        print(f"{'='*60}")
        
        state_files = []
        
        # 执行多次相同的实验
        for trial in range(self.config['num_trials']):
            print(f"\n--- Trial {trial + 1}/{self.config['num_trials']} ---")
            
            # 1. 重置到初始状态
            if not self.reset_to_initial():
                print(f"❌ Trial {trial + 1}: 重置失败")
                continue
            
            time.sleep(0.5)
            
            # 2. 执行固定动作
            print(f"🎲 执行动作 {self.config['action_duration']}秒...")
            self.simulate_actions(self.config['action_duration'])
            
            # 3. 导出最终状态
            state_file = self.export_current_state(f"consistency_trial_{trial}.json")
            if state_file:
                state_files.append(state_file)
            
            time.sleep(0.3)
        
        # 4. 比较所有状态
        if len(state_files) < 2:
            return {"success": False, "error": "状态文件数量不足"}
        
        print(f"\n🔍 比较 {len(state_files)} 个状态...")
        
        # 使用第一个状态作为参考
        reference_state = state_files[0]
        differences = []
        
        for i, state_file in enumerate(state_files[1:], 1):
            diff = self.compare_states(reference_state)
            if diff is not None:
                differences.append(diff)
                print(f"   Trial {i+1} vs Trial 1: {diff:.6e}")
        
        if not differences:
            return {"success": False, "error": "状态比较失败"}
        
        # 5. 详细分析
        detailed_analysis = self.analyze_state_details(state_files)
        
        # 6. 计算统计数据
        avg_diff = sum(differences) / len(differences)
        max_diff = max(differences)
        min_diff = min(differences)
        
        threshold = self.config['consistency_threshold']
        is_consistent = max_diff < threshold
        
        result = {
            "success": True,
            "test_type": "basic_consistency",
            "num_trials": len(state_files),
            "average_difference": avg_diff,
            "max_difference": max_diff,
            "min_difference": min_diff,
            "is_consistent": is_consistent,
            "threshold": threshold,
            "state_files": state_files,
            "detailed_analysis": detailed_analysis
        }
        
        print(f"\n📊 一致性结果:")
        print(f"   平均差异: {avg_diff:.6e}")
        print(f"   最大差异: {max_diff:.6e}")
        print(f"   最小差异: {min_diff:.6e}")
        print(f"   阈值: {threshold:.6e}")
        
        if is_consistent:
            print(f"✅ 状态高度一致！")
        else:
            print(f"⚠️ 状态存在差异，可能不是完全确定性的")
        
        return result
    
    def test_long_sequence_consistency(self):
        """测试长序列的一致性传播"""
        print(f"\n{'='*60}")
        print(f"🔬 长序列一致性测试")
        print(f"{'='*60}")
        
        # 测试更长的动作序列，看误差是否累积
        durations = [1, 2, 5, 10]
        results = []
        
        for duration in durations:
            print(f"\n--- 测试 {duration}秒序列 ---")
            
            state_files = []
            
            for trial in range(min(3, self.config['num_trials'])):
                # 重置
                if not self.reset_to_initial():
                    continue
                
                time.sleep(0.5)
                
                # 执行动作
                self.simulate_actions(duration)
                
                # 导出状态
                state_file = self.export_current_state(f"long_seq_{duration}s_trial{trial}.json")
                if state_file:
                    state_files.append(state_file)
                
                time.sleep(0.3)
            
            # 比较状态
            if len(state_files) >= 2:
                reference_state = state_files[0]
                diffs = []
                
                for state_file in state_files[1:]:
                    diff = self.compare_states(reference_state)
                    if diff is not None:
                        diffs.append(diff)
                
                if diffs:
                    avg_diff = sum(diffs) / len(diffs)
                    results.append({
                        "duration": duration,
                        "average_difference": avg_diff,
                        "max_difference": max(diffs),
                        "num_trials": len(state_files)
                    })
                    print(f"   平均差异: {avg_diff:.6e}")
        
        # 分析误差累积趋势
        if len(results) > 1:
            print(f"\n📈 误差累积分析:")
            for r in results:
                print(f"   {r['duration']}秒: {r['average_difference']:.6e}")
            
            # 检查是否有明显的误差累积
            first_diff = results[0]['average_difference']
            last_diff = results[-1]['average_difference']
            
            if last_diff > first_diff * 10:
                print(f"⚠️ 检测到误差累积：{last_diff/first_diff:.1f}倍增长")
            else:
                print(f"✅ 误差累积不明显")
        
        return {
            "success": True,
            "test_type": "long_sequence_consistency",
            "results": results
        }
    
    def run_test(self):
        """运行完整测试"""
        print("\n" + "="*60)
        print("🧪 状态一致性验证测试")
        print("="*60)
        
        # 1. 准备环境
        self.ensure_directories()
        
        # 2. 启动游戏
        if not self.start_game():
            return
        
        try:
            # 3. 等待进入游戏场景
            print("⏳ 等待进入游戏场景...")
            time.sleep(15)
            
            # 4. 保存初始状态
            if not self.save_initial_state():
                print("❌ 无法保存初始状态，测试终止")
                return
            
            time.sleep(1)
            
            # 5. 运行基本一致性测试
            basic_result = self.test_basic_consistency()
            self.test_results.append(basic_result)
            
            # 6. 运行长序列一致性测试
            long_seq_result = self.test_long_sequence_consistency()
            self.test_results.append(long_seq_result)
            
            # 7. 分析总体结果
            self.analyze_results()
            
            # 8. 保存报告
            self.save_report()
            
        finally:
            # 9. 清理
            self.stop_game()
    
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "="*60)
        print("📊 测试结果总结")
        print("="*60)
        
        basic_tests = [r for r in self.test_results if r.get('test_type') == 'basic_consistency']
        long_seq_tests = [r for r in self.test_results if r.get('test_type') == 'long_sequence_consistency']
        
        if basic_tests:
            for test in basic_tests:
                if test.get('is_consistent'):
                    print(f"\n✅ 基本一致性测试通过")
                    print(f"   最大差异: {test['max_difference']:.6e}")
                    print(f"   平均差异: {test['average_difference']:.6e}")
                else:
                    print(f"\n⚠️ 基本一致性测试未达到预期")
                    print(f"   最大差异: {test['max_difference']:.6e} > 阈值 {test['threshold']:.6e}")
                
                # 打印详细分析
                if test.get('detailed_analysis'):
                    print(f"\n📋 各刚体一致性分析:")
                    for rb_name, analysis in test['detailed_analysis'].items():
                        if rb_name:  # 跳过空名称
                            pos_std = analysis['position_std']
                            rot_std = analysis['rotation_std']
                            print(f"   {rb_name}:")
                            print(f"     位置标准差: [{pos_std[0]:.2e}, {pos_std[1]:.2e}]")
                            print(f"     旋转标准差: {rot_std:.2e}")
        
        if long_seq_tests:
            print(f"\n📈 长序列一致性测试:")
            for test in long_seq_tests:
                if test.get('results'):
                    for r in test['results']:
                        print(f"   {r['duration']}秒序列: 平均差异 {r['average_difference']:.6e}")
        
        print(f"\n{'='*60}")
    
    def save_report(self):
        """保存测试报告"""
        report_file = os.path.join(self.get_result_dir(), "consistency_test_report.json")
        
        report = {
            "test_name": "State Consistency Test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "results": self.test_results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 测试报告已保存: {report_file}")

def main():
    tester = ConsistencyTester()
    tester.run_test()

if __name__ == "__main__":
    main()


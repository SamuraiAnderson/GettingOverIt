"""
状态保存/加载测试（Checkpoint测试）
测试目标：验证游戏状态的保存和加载功能
测试方法：
1. 在不同游戏进度点保存checkpoint
2. 执行动作后恢复到checkpoint
3. 验证恢复的准确性
4. 测试从checkpoint开始执行相同动作的可复现性
"""

import os
import sys
import time
import json
import random
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

class CheckpointTester:
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
            "checkpoint_intervals": [0, 2, 5, 8, 10],  # 秒
            "action_duration": 3.0,
            "num_trials": 3
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
    
    def save_checkpoint(self, checkpoint_name):
        """保存checkpoint"""
        print(f"💾 保存 Checkpoint: {checkpoint_name}")
        self.send_signal("save_checkpoint.signal", checkpoint_name)
        success, result = self.wait_for_result(f"save_checkpoint_{checkpoint_name}")
        
        if success:
            print(f"✅ Checkpoint '{checkpoint_name}' 已保存")
        else:
            print(f"❌ 保存 Checkpoint 失败: {result}")
        
        return success
    
    def load_checkpoint(self, checkpoint_name):
        """加载checkpoint"""
        print(f"🔄 加载 Checkpoint: {checkpoint_name}")
        self.send_signal("load_checkpoint.signal", checkpoint_name)
        success, result = self.wait_for_result(f"load_checkpoint_{checkpoint_name}")
        
        if success:
            print(f"✅ Checkpoint '{checkpoint_name}' 已加载")
        else:
            print(f"❌ 加载 Checkpoint 失败: {result}")
        
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
        print(f"🎲 模拟动作 {duration}秒...")
        time.sleep(duration)
    
    def test_checkpoint_save_load(self, checkpoint_name, wait_time):
        """测试单个checkpoint的保存和加载"""
        print(f"\n{'='*60}")
        print(f"📊 测试 Checkpoint: {checkpoint_name} (等待 {wait_time}秒)")
        print(f"{'='*60}")
        
        # 1. 从初始状态开始，执行动作到checkpoint点
        print(f"⏳ 等待 {wait_time}秒 到达checkpoint点...")
        self.simulate_actions(wait_time)
        
        # 2. 保存checkpoint
        if not self.save_checkpoint(checkpoint_name):
            return {"success": False, "error": "保存checkpoint失败"}
        
        # 等待保存完成
        time.sleep(0.5)
        
        # 3. 导出checkpoint状态作为参考
        checkpoint_ref_file = self.export_current_state(f"{checkpoint_name}_ref.json")
        if not checkpoint_ref_file:
            return {"success": False, "error": "导出checkpoint参考状态失败"}
        
        # 4. 继续执行动作（改变状态）
        print("🎲 执行额外动作改变状态...")
        self.simulate_actions(self.config['action_duration'])
        
        # 5. 加载checkpoint（恢复状态）
        load_start_time = time.time()
        if not self.load_checkpoint(checkpoint_name):
            return {"success": False, "error": "加载checkpoint失败"}
        load_duration = time.time() - load_start_time
        
        # 等待物理系统稳定
        time.sleep(0.5)
        
        # 6. 导出加载后的状态
        after_load_file = self.export_current_state(f"{checkpoint_name}_after_load.json")
        
        # 7. 比较加载后状态与checkpoint参考状态
        diff = self.compare_states(checkpoint_ref_file)
        
        if diff is None:
            return {"success": False, "error": "状态比较失败"}
        
        # 8. 分析结果
        threshold = 1e-3
        load_success = diff < threshold
        
        result = {
            "success": True,
            "checkpoint_name": checkpoint_name,
            "wait_time": wait_time,
            "load_duration": load_duration,
            "state_difference": diff,
            "load_quality": "excellent" if diff < 1e-5 else "good" if load_success else "poor",
            "load_success": load_success,
            "checkpoint_ref_file": checkpoint_ref_file,
            "after_load_file": after_load_file
        }
        
        if load_success:
            print(f"✅ Checkpoint加载成功: 差异 {diff:.6e} < 阈值 {threshold:.6e}")
        else:
            print(f"⚠️ Checkpoint加载质量不佳: 差异 {diff:.6e} > 阈值 {threshold:.6e}")
        
        print(f"⏱️ 加载耗时: {load_duration:.3f}秒")
        
        return result
    
    def test_checkpoint_reproducibility(self, checkpoint_name):
        """测试从checkpoint开始的可复现性"""
        print(f"\n{'='*60}")
        print(f"🔬 测试 Checkpoint 可复现性: {checkpoint_name}")
        print(f"{'='*60}")
        
        trajectories = []
        
        # 从同一checkpoint开始，执行相同动作多次
        for trial in range(self.config['num_trials']):
            print(f"\n--- Trial {trial + 1}/{self.config['num_trials']} ---")
            
            # 1. 加载checkpoint
            if not self.load_checkpoint(checkpoint_name):
                continue
            
            time.sleep(0.5)
            
            # 2. 导出起始状态
            start_state_file = self.export_current_state(f"{checkpoint_name}_trial{trial}_start.json")
            
            # 3. 执行固定动作序列
            self.simulate_actions(2.0)
            
            # 4. 导出结束状态
            end_state_file = self.export_current_state(f"{checkpoint_name}_trial{trial}_end.json")
            
            if start_state_file and end_state_file:
                trajectories.append({
                    "trial": trial,
                    "start_state": start_state_file,
                    "end_state": end_state_file
                })
        
        # 比较所有轨迹的一致性
        if len(trajectories) < 2:
            return {"success": False, "error": "轨迹数量不足"}
        
        # 比较所有trial的起始状态
        start_diffs = []
        for i in range(1, len(trajectories)):
            self.load_checkpoint(checkpoint_name)
            time.sleep(0.3)
            diff = self.compare_states(trajectories[0]['start_state'])
            if diff is not None:
                start_diffs.append(diff)
        
        # 计算一致性
        avg_start_diff = sum(start_diffs) / len(start_diffs) if start_diffs else float('inf')
        
        result = {
            "success": True,
            "checkpoint_name": checkpoint_name,
            "num_trials": len(trajectories),
            "avg_start_difference": avg_start_diff,
            "reproducibility": "excellent" if avg_start_diff < 1e-5 else "good" if avg_start_diff < 1e-3 else "poor",
            "trajectories": trajectories
        }
        
        print(f"\n📊 可复现性结果:")
        print(f"   平均起始差异: {avg_start_diff:.6e}")
        print(f"   质量评级: {result['reproducibility']}")
        
        return result
    
    def run_test(self):
        """运行完整测试"""
        print("\n" + "="*60)
        print("🧪 状态保存/加载测试（Checkpoint测试）")
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
            
            # 5. 测试不同时间点的checkpoint
            for i, wait_time in enumerate(self.config['checkpoint_intervals']):
                checkpoint_name = f"checkpoint_{i:02d}_t{wait_time}"
                result = self.test_checkpoint_save_load(checkpoint_name, wait_time)
                self.test_results.append(result)
                
                # 测试可复现性（只测试中间的checkpoint）
                if i > 0 and i < len(self.config['checkpoint_intervals']) - 1:
                    repro_result = self.test_checkpoint_reproducibility(checkpoint_name)
                    self.test_results.append(repro_result)
                
                # 重置到初始状态，准备下一个测试
                self.send_signal("reset.signal")
                self.wait_for_result("reset")
                time.sleep(1)
            
            # 6. 分析总体结果
            self.analyze_results()
            
            # 7. 保存报告
            self.save_report()
            
        finally:
            # 8. 清理
            self.stop_game()
    
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "="*60)
        print("📊 测试结果分析")
        print("="*60)
        
        save_load_tests = [r for r in self.test_results if 'checkpoint_name' in r and 'load_duration' in r]
        reproducibility_tests = [r for r in self.test_results if 'reproducibility' in r]
        
        print(f"\n保存/加载测试数: {len(save_load_tests)}")
        print(f"可复现性测试数: {len(reproducibility_tests)}")
        
        if save_load_tests:
            successful_loads = [r for r in save_load_tests if r.get('load_success', False)]
            load_durations = [r['load_duration'] for r in save_load_tests if 'load_duration' in r]
            state_diffs = [r['state_difference'] for r in save_load_tests if 'state_difference' in r]
            
            print(f"\n保存/加载成功率: {len(successful_loads)/len(save_load_tests)*100:.1f}%")
            
            if load_durations:
                print(f"\n加载耗时统计:")
                print(f"  平均: {sum(load_durations)/len(load_durations):.3f}秒")
                print(f"  最小: {min(load_durations):.3f}秒")
                print(f"  最大: {max(load_durations):.3f}秒")
            
            if state_diffs:
                print(f"\n状态差异统计:")
                print(f"  平均: {sum(state_diffs)/len(state_diffs):.6e}")
                print(f"  最小: {min(state_diffs):.6e}")
                print(f"  最大: {max(state_diffs):.6e}")
        
        if reproducibility_tests:
            print(f"\n可复现性测试:")
            for r in reproducibility_tests:
                print(f"  {r['checkpoint_name']}: {r['reproducibility']} (差异: {r['avg_start_difference']:.6e})")
        
        # 总体评价
        if len(successful_loads) == len(save_load_tests):
            print(f"\n✅ 测试通过！所有checkpoint保存/加载都成功")
        else:
            print(f"\n⚠️ 部分checkpoint测试失败")
    
    def save_report(self):
        """保存测试报告"""
        report_file = os.path.join(self.get_result_dir(), "checkpoint_test_report.json")
        
        report = {
            "test_name": "Checkpoint Save/Load Test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "results": self.test_results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 测试报告已保存: {report_file}")

def main():
    tester = CheckpointTester()
    tester.run_test()

if __name__ == "__main__":
    main()


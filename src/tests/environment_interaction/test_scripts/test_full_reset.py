"""
完全重置测试
测试目标：验证游戏能否完全重置到初始状态
测试方法：
1. 启动游戏到初始状态
2. 记录初始物理状态
3. 执行随机动作序列
4. 触发重置
5. 对比重置后状态与初始状态
6. 重复多次验证一致性
"""

import os
import sys
import time
import json
import random
import subprocess
from pathlib import Path

# 添加父目录到路径以导入共享模块
sys.path.append(str(Path(__file__).parent.parent))

class FullResetTester:
    def __init__(self, config_path="test_config.json"):
        """初始化测试器"""
        self.config = self.load_config(config_path)
        self.game_process = None
        self.test_results = []
        
    def load_config(self, config_path):
        """加载配置文件"""
        config_file = Path(__file__).parent.parent / config_path
        if not config_file.exists():
            print(f"⚠️ 配置文件不存在: {config_file}")
            print("使用默认配置")
            return self.get_default_config()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_default_config(self):
        """默认配置"""
        return {
            "game_exe_path": "C:/Program Files/Steam/steamapps/common/Getting Over It/Getting Over It.exe",
            "game_data_dir": "C:/Program Files/Steam/steamapps/common/Getting Over It/GoiData",
            "test_duration": 5.0,
            "action_interval": 0.1,
            "num_episodes": 10
        }
    
    def get_signal_dir(self):
        """获取信号目录"""
        return os.path.join(self.config['game_data_dir'], 'TestSignals')
    
    def get_state_dir(self):
        """获取状态目录"""
        return os.path.join(self.config['game_data_dir'], 'GameStates')
    
    def get_result_dir(self):
        """获取结果目录"""
        return os.path.join(self.config['game_data_dir'], 'TestResults')
    
    def ensure_directories(self):
        """确保目录存在"""
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
            
            # 等待游戏完全加载
            print("⏳ 等待游戏加载...")
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
                # 尝试优雅关闭
                self.game_process.terminate()
                self.game_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 强制关闭
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
            # 检查成功文件
            success_file = os.path.join(result_dir, f"{operation}_success.result")
            if os.path.exists(success_file):
                with open(success_file, 'r') as f:
                    content = f.read()
                os.remove(success_file)
                return True, self.parse_result(content)
            
            # 检查失败文件
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
        
        if success:
            print("✅ 初始状态已保存")
            return True
        else:
            print(f"❌ 保存初始状态失败: {result}")
            return False
    
    def reset_to_initial(self):
        """重置到初始状态"""
        print("🔄 重置到初始状态...")
        self.send_signal("reset.signal")
        success, result = self.wait_for_result("reset")
        
        if success:
            print("✅ 重置成功")
            return True
        else:
            print(f"❌ 重置失败: {result}")
            return False
    
    def export_current_state(self, filename):
        """导出当前状态"""
        print(f"📤 导出当前状态: {filename}")
        self.send_signal("export_state.signal", filename)
        success, result = self.wait_for_result("export_state")
        
        if success:
            return os.path.join(self.get_state_dir(), filename)
        else:
            print(f"❌ 导出状态失败: {result}")
            return None
    
    def compare_states(self, state_file):
        """比较当前状态与文件中的状态"""
        print(f"🔍 比较状态: {state_file}")
        self.send_signal("compare_state.signal", os.path.basename(state_file))
        success, result = self.wait_for_result("compare_state")
        
        if success and 'data' in result:
            diff = float(result['data'])
            print(f"   状态差异: {diff:.6e}")
            return diff
        else:
            print(f"❌ 比较状态失败: {result}")
            return None
    
    def simulate_random_actions(self, duration):
        """模拟随机动作（通过等待实现，实际动作由游戏内部处理）"""
        print(f"🎲 模拟随机动作 {duration}秒...")
        # 这里可以扩展为实际的鼠标输入模拟
        time.sleep(duration)
    
    def run_single_episode(self, episode_num):
        """运行单个episode测试"""
        print(f"\n{'='*60}")
        print(f"📊 Episode {episode_num + 1}")
        print(f"{'='*60}")
        
        # 1. 导出初始状态作为参考
        initial_state_file = self.export_current_state(f"initial_ref_{episode_num}.json")
        if not initial_state_file:
            return {"success": False, "error": "无法导出初始状态"}
        
        # 等待状态稳定
        time.sleep(0.5)
        
        # 2. 执行随机动作
        self.simulate_random_actions(self.config['test_duration'])
        
        # 3. 导出执行动作后的状态
        after_action_file = self.export_current_state(f"after_action_{episode_num}.json")
        
        # 4. 重置到初始状态
        reset_start_time = time.time()
        if not self.reset_to_initial():
            return {"success": False, "error": "重置失败"}
        reset_duration = time.time() - reset_start_time
        
        # 等待物理系统稳定
        time.sleep(0.5)
        
        # 5. 导出重置后的状态
        after_reset_file = self.export_current_state(f"after_reset_{episode_num}.json")
        
        # 6. 比较重置后状态与初始状态
        diff = self.compare_states(initial_state_file)
        
        if diff is None:
            return {"success": False, "error": "状态比较失败"}
        
        # 7. 分析结果
        result = {
            "success": True,
            "episode": episode_num,
            "reset_duration": reset_duration,
            "state_difference": diff,
            "initial_state_file": initial_state_file,
            "after_action_file": after_action_file,
            "after_reset_file": after_reset_file
        }
        
        # 判断重置是否成功（差异小于阈值）
        threshold = 1e-3
        if diff < threshold:
            print(f"✅ 重置成功: 差异 {diff:.6e} < 阈值 {threshold:.6e}")
            result['reset_success'] = True
        else:
            print(f"⚠️ 重置质量不佳: 差异 {diff:.6e} > 阈值 {threshold:.6e}")
            result['reset_success'] = False
        
        print(f"⏱️ 重置耗时: {reset_duration:.3f}秒")
        
        return result
    
    def run_test(self):
        """运行完整测试"""
        print("\n" + "="*60)
        print("🧪 完全重置测试")
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
            
            # 等待状态保存完成
            time.sleep(1)
            
            # 5. 运行多个episode
            for i in range(self.config['num_episodes']):
                result = self.run_single_episode(i)
                self.test_results.append(result)
                
                # episode之间等待一下
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
        
        successful_tests = [r for r in self.test_results if r.get('success', False)]
        successful_resets = [r for r in successful_tests if r.get('reset_success', False)]
        
        print(f"\n总测试数: {len(self.test_results)}")
        print(f"成功测试数: {len(successful_tests)}")
        print(f"重置成功数: {len(successful_resets)}")
        print(f"成功率: {len(successful_resets)/len(self.test_results)*100:.1f}%")
        
        if successful_tests:
            reset_times = [r['reset_duration'] for r in successful_tests]
            state_diffs = [r['state_difference'] for r in successful_tests]
            
            print(f"\n重置耗时统计:")
            print(f"  平均: {sum(reset_times)/len(reset_times):.3f}秒")
            print(f"  最小: {min(reset_times):.3f}秒")
            print(f"  最大: {max(reset_times):.3f}秒")
            
            print(f"\n状态差异统计:")
            print(f"  平均: {sum(state_diffs)/len(state_diffs):.6e}")
            print(f"  最小: {min(state_diffs):.6e}")
            print(f"  最大: {max(state_diffs):.6e}")
        
        # 判断测试是否通过
        if len(successful_resets) == len(self.test_results):
            print(f"\n✅ 测试通过！所有重置都成功且状态一致")
        elif len(successful_resets) >= len(self.test_results) * 0.9:
            print(f"\n⚠️ 测试基本通过，但有少量重置质量不佳")
        else:
            print(f"\n❌ 测试失败！重置成功率过低")
    
    def save_report(self):
        """保存测试报告"""
        report_file = os.path.join(self.get_result_dir(), "full_reset_test_report.json")
        
        report = {
            "test_name": "Full Reset Test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len([r for r in self.test_results if r.get('success', False)]),
                "successful_resets": len([r for r in self.test_results if r.get('reset_success', False)])
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 测试报告已保存: {report_file}")

def main():
    tester = FullResetTester()
    tester.run_test()

if __name__ == "__main__":
    main()


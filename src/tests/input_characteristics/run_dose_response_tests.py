"""
剂量-响应测试 - 半自动化执行脚本
自动处理文件重命名和游戏启动
"""

import sys
import io
import time
import subprocess
import shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class DoseResponseTestRunner:
    def __init__(self):
        self.game_exe = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GettingOverIt.exe")
        self.game_data = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/GoiData")
        self.result_folder = self.game_data / "GameResults"
        
        self.test_values = [5, 10, 15, 20, 25, 30, 50, 80, 100]
        self.current_test = 0
        
    def run_single_test(self, value):
        """运行单个测试"""
        print("\n" + "="*60)
        print(f"测试 {self.current_test + 1}/{len(self.test_values)}: 输入值 = {value}")
        print("="*60)
        
        # 步骤1: 准备输入文件
        input_test_file = self.game_data / f"input_test_{int(value):03d}.csv"
        input_command_file = self.game_data / "input_commands.csv"
        
        if not input_test_file.exists():
            print(f"❌ 错误: 找不到测试文件 {input_test_file}")
            return False
        
        print(f"\n步骤1: 准备输入文件")
        print(f"  复制: {input_test_file.name} → input_commands.csv")
        shutil.copy(input_test_file, input_command_file)
        print(f"  ✓ 完成")
        
        # 步骤2: 启动游戏
        print(f"\n步骤2: 启动游戏")
        print(f"  游戏路径: {self.game_exe}")
        print(f"  ⏳ 启动中...")
        
        try:
            process = subprocess.Popen([str(self.game_exe)])
            
            print(f"  ✓ 游戏已启动 (PID: {process.pid})")
            print(f"\n  等待游戏自动完成...")
            print(f"  - 游戏会自动进入场景")
            print(f"  - 自动采集3秒数据")
            print(f"  - 自动关闭")
            
            # 等待游戏结束
            process.wait()
            
            print(f"\n  ✓ 游戏已关闭")
            
            # 等待文件系统同步
            time.sleep(2)
            
        except FileNotFoundError:
            print(f"  ❌ 错误: 找不到游戏可执行文件")
            return False
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            return False
        
        # 步骤3: 保存结果
        result_latest = self.result_folder / "ContinuousTracking_latest.csv"
        result_saved = self.result_folder / f"result_{int(value):03d}.csv"
        
        print(f"\n步骤3: 保存结果")
        print(f"  查找结果文件...")
        
        if not result_latest.exists():
            print(f"  ❌ 错误: 找不到结果文件 {result_latest}")
            return False
        
        print(f"  复制: ContinuousTracking_latest.csv → result_{int(value):03d}.csv")
        shutil.copy(result_latest, result_saved)
        print(f"  ✓ 完成")
        
        print("\n" + "="*60)
        print(f"✅ 测试 {self.current_test + 1}/{len(self.test_values)} 完成！")
        print("="*60)
        
        self.current_test += 1
        return True
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "🔬"*30)
        print("剂量-响应验证测试 - 自动执行")
        print("🔬"*30)
        
        print(f"\n测试配置:")
        print(f"  测试序列: {self.test_values}")
        print(f"  测试数量: {len(self.test_values)}")
        print(f"  预计耗时: 约 {len(self.test_values) * 0.5:.1f} 分钟")
        
        print(f"\n游戏路径: {self.game_exe}")
        print(f"数据目录: {self.game_data}")
        
        if not self.game_exe.exists():
            print(f"\n❌ 错误: 找不到游戏可执行文件")
            print(f"请检查路径: {self.game_exe}")
            return
        
        confirm = input("\n确认开始测试？(y/n): ").lower()
        if confirm != 'y':
            print("已取消")
            return
        
        start_time = time.time()
        success_count = 0
        
        for value in self.test_values:
            success = self.run_single_test(value)
            
            if success:
                success_count += 1
                
                if self.current_test < len(self.test_values):
                    print(f"\n⏸️  准备下一个测试...")
                    time.sleep(2)
            else:
                print(f"\n⚠️  测试失败，是否继续？")
                cont = input("继续下一个测试？(y/n): ").lower()
                if cont != 'y':
                    break
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("测试完成总结")
        print("="*60)
        print(f"\n  成功: {success_count}/{len(self.test_values)}")
        print(f"  耗时: {elapsed/60:.1f} 分钟")
        
        if success_count == len(self.test_values):
            print("\n✅ 所有测试完成！")
            print("\n下一步: 运行分析")
            print("  python dose_response_test.py --analyze")
        else:
            print(f"\n⚠️  部分测试未完成")
            print(f"  已完成: {success_count}/{len(self.test_values)}")


if __name__ == '__main__':
    runner = DoseResponseTestRunner()
    runner.run_all_tests()


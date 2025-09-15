#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试Unity插件状态
"""

import json
import time
import os
from pathlib import Path

def check_signal_files():
    """检查信号文件状态"""
    print("🔍 检查信号文件状态")
    print("=" * 30)
    
    signal_files = [
        "src/Data/auto_mode_signal.json",
        "src/Data/auto_mode_response.json",
        "src/Data/collection_signal.json",
        "src/Data/unity_response.json"
    ]
    
    for file_path in signal_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} 存在")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                print(f"   内容: {content}")
            except Exception as e:
                print(f"   读取错误: {e}")
        else:
            print(f"❌ {file_path} 不存在")

def send_test_signal():
    """发送测试信号"""
    print("\n📤 发送测试信号")
    print("=" * 30)
    
    test_signal = {
        "enable_auto_mode": True,
        "duration": 5,
        "frequency": 30,
        "timestamp": time.time()
    }
    
    signal_file = Path("src/Data/auto_mode_signal.json")
    with open(signal_file, 'w', encoding='utf-8') as f:
        json.dump(test_signal, f, indent=2)
    
    print(f"✅ 测试信号已发送: {signal_file}")
    print(f"   内容: {test_signal}")

def wait_for_response(timeout=10):
    """等待响应"""
    print(f"\n⏳ 等待Unity响应 (超时: {timeout}秒)")
    print("=" * 30)
    
    response_file = Path("src/Data/auto_mode_response.json")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if response_file.exists():
            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    response = json.load(f)
                print(f"✅ 收到响应: {response}")
                return True
            except Exception as e:
                print(f"❌ 响应解析错误: {e}")
        
        time.sleep(1)
        elapsed = int(time.time() - start_time)
        print(f"   等待中... {elapsed}/{timeout}秒")
    
    print("❌ 响应超时")
    return False

def check_game_process():
    """检查游戏进程"""
    print("\n🎮 检查游戏进程")
    print("=" * 30)
    
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq GettingOverIt.exe'], 
                              capture_output=True, text=True)
        if 'GettingOverIt.exe' in result.stdout:
            print("✅ 游戏进程正在运行")
            return True
        else:
            print("❌ 游戏进程未运行")
            return False
    except Exception as e:
        print(f"❌ 检查游戏进程失败: {e}")
        return False

def check_plugin_file():
    """检查插件文件"""
    print("\n🔌 检查Unity插件文件")
    print("=" * 30)
    
    plugin_path = Path("C:/Users/Symbol/software/game_store/steam/steamapps/common/Getting Over It/BepInEx/plugins/GoiHitboxLogger.dll")
    
    if plugin_path.exists():
        print(f"✅ 插件文件存在: {plugin_path}")
        print(f"   文件大小: {plugin_path.stat().st_size} 字节")
        print(f"   修改时间: {time.ctime(plugin_path.stat().st_mtime)}")
        return True
    else:
        print(f"❌ 插件文件不存在: {plugin_path}")
        return False

def main():
    """主函数"""
    print("🚀 Unity插件调试工具")
    print("=" * 40)
    
    # 检查游戏进程
    if not check_game_process():
        print("\n❌ 游戏未运行，请先启动游戏")
        return
    
    # 检查插件文件
    if not check_plugin_file():
        print("\n❌ 插件文件不存在，请检查编译")
        return
    
    # 检查信号文件
    check_signal_files()
    
    # 发送测试信号
    send_test_signal()
    
    # 等待响应
    if wait_for_response(15):
        print("\n🎉 Unity插件通信正常!")
    else:
        print("\n❌ Unity插件通信失败!")
        print("\n💡 可能的原因:")
        print("   1. Unity插件没有正确加载")
        print("   2. 游戏场景不是预期的场景")
        print("   3. 插件代码有编译错误")
        print("   4. 需要重新编译插件")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的自动化数据采集系统测试
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试模块导入"""
    try:
        sys.path.append('src/Python')
        from auto_data_collector_v2 import AutoDataCollectorV2
        print("✅ 模块导入测试通过")
        return True
    except Exception as e:
        print(f"❌ 模块导入测试失败: {e}")
        return False

def test_data_collector():
    """测试数据采集器初始化"""
    try:
        from auto_data_collector_v2 import AutoDataCollectorV2
        collector = AutoDataCollectorV2('test_output')
        print("✅ 数据采集器初始化测试通过")
        
        # 清理测试目录
        import shutil
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        return True
    except Exception as e:
        print(f"❌ 数据采集器测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    required_files = [
        'src/Python/auto_data_collector_v2.py',
        'src/Scripts/auto_data_collection.bat',
        'src/GameRuntime/Tracking/ContinuousTracker.cs'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    else:
        print("✅ 文件结构测试通过")
        return True

def main():
    """主测试函数"""
    print("🧪 自动化数据采集系统测试")
    print("=" * 40)
    
    tests = [
        ("文件结构", test_file_structure),
        ("模块导入", test_imports),
        ("数据采集器", test_data_collector)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 测试: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name}测试失败")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败!")
        return 1

if __name__ == "__main__":
    exit(main())

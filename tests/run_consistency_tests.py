#!/usr/bin/env python3
"""运行功能一致性测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试模块
from tests.test_functional_consistency import main

if __name__ == "__main__":
    print("开始执行功能一致性测试...")
    success = main()
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 存在测试失败，请查看报告！")
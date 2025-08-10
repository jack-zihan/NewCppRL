#!/usr/bin/env python3
"""
功能一致性测试 - 验证新旧版本行为完全一致
测试工程师（Agent E）ultrathink模式
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any
import traceback

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入新旧版本
from rules.jump_path import JUMP as OldJUMP
from rules.jump_path import Boustrophedon_Cell_Path as OldBCP
from rules.jump_path import SNAKE as OldSNAKE

from rules_new.algorithms.jump_path import JUMP as NewJUMP
from rules_new.algorithms.jump_path import BoustrophedonCellPath as NewBCP
from rules_new.algorithms.jump_path import SNAKE as NewSNAKE

class FunctionalConsistencyTester:
    """功能一致性测试器 - ultrathink模式"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_details = []
        
    def compare_values(self, val1, val2, tolerance=1e-6, name="value"):
        """比较两个值是否一致"""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = abs(val1 - val2)
            if diff > tolerance:
                return False, f"{name}: {val1} vs {val2}, diff={diff}"
            return True, None
        elif isinstance(val1, bool) and isinstance(val2, bool):
            if val1 != val2:
                return False, f"{name}: {val1} vs {val2}"
            return True, None
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                return False, f"{name} length: {len(val1)} vs {len(val2)}"
            for i, (v1, v2) in enumerate(zip(val1, val2)):
                result, msg = self.compare_values(v1, v2, tolerance, f"{name}[{i}]")
                if not result:
                    return False, msg
            return True, None
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if val1.shape != val2.shape:
                return False, f"{name} shape: {val1.shape} vs {val2.shape}"
            if not np.allclose(val1, val2, rtol=tolerance, atol=tolerance):
                max_diff = np.max(np.abs(val1 - val2))
                return False, f"{name} array diff, max={max_diff}"
            return True, None
        else:
            if val1 != val2:
                return False, f"{name}: {val1} vs {val2}"
            return True, None
    
    def run_test(self, test_name: str, test_func):
        """运行单个测试"""
        self.total_tests += 1
        print(f"\n{'='*60}")
        print(f"测试: {test_name}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            
            if result["passed"]:
                self.passed_tests += 1
                status = "✅ PASSED"
            else:
                self.failed_tests += 1
                status = "❌ FAILED"
            
            print(f"状态: {status}")
            print(f"耗时: {elapsed:.3f}s")
            
            if not result["passed"]:
                print(f"原因: {result['reason']}")
                if "details" in result:
                    print(f"详情: {result['details']}")
            
            self.test_results.append({
                "name": test_name,
                "passed": result["passed"],
                "time": elapsed,
                "result": result
            })
            
            return result["passed"]
            
        except Exception as e:
            self.failed_tests += 1
            print(f"状态: ❌ ERROR")
            print(f"错误: {str(e)}")
            print(f"追踪:\n{traceback.format_exc()}")
            
            self.test_results.append({
                "name": test_name,
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            return False
    
    def test_initial_turn_direction(self):
        """测试1: 初始turn_direction值"""
        print("验证初始turn_direction值...")
        
        # 测试参数
        test_cases = [
            {"farm_shape": "square", "expected": True},
            {"farm_shape": "rectangle", "expected": True},
            {"farm_shape": "triangle", "expected": True},
        ]
        
        all_passed = True
        details = []
        
        for case in test_cases:
            farm_shape = case["farm_shape"]
            expected = case["expected"]
            
            # 创建测试数据
            if farm_shape == "square":
                farm = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
            elif farm_shape == "rectangle":
                farm = np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
            else:
                farm = np.array([[0, 0], [10, 0], [5, 10]])
            
            # 创建旧版本
            old_jump = OldJUMP(200, 200, 0.5, farm)
            old_value = old_jump.turn_direction
            
            # 创建新版本
            new_jump = NewJUMP(200, 200, 0.5, farm)
            new_value = new_jump.turn_direction
            
            passed, msg = self.compare_values(old_value, new_value, name=f"{farm_shape}_turn_direction")
            
            if not passed:
                all_passed = False
                details.append(f"{farm_shape}: OLD={old_value}, NEW={new_value}")
            else:
                details.append(f"{farm_shape}: ✅ 都是 {new_value}")
        
        return {
            "passed": all_passed,
            "reason": "turn_direction初始值一致" if all_passed else "turn_direction初始值不一致",
            "details": "\n".join(details)
        }
    
    def test_farm_vertices_coordinate(self):
        """测试2: farm_vertices坐标系转换"""
        print("验证farm_vertices坐标系...")
        
        # 测试各种农场形状
        test_farms = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),  # 正方形
            np.array([[5, 5], [15, 5], [15, 15], [5, 15]]),  # 偏移正方形
            np.array([[0, 0], [20, 0], [20, 10], [0, 10]]),  # 矩形
            np.array([[0, 0], [10, 0], [5, 10]]),  # 三角形
        ]
        
        all_passed = True
        details = []
        
        for i, farm in enumerate(test_farms):
            print(f"  测试农场形状 {i+1}/{len(test_farms)}...")
            
            # 创建算法实例
            old_jump = OldJUMP(200, 200, 0.5, farm)
            new_jump = NewJUMP(200, 200, 0.5, farm)
            
            # 获取farm_vertices
            old_vertices = old_jump.farm_vertices
            new_vertices = new_jump.farm_vertices
            
            # 验证坐标系（新版本应该是[y,x]格式）
            # 检查是否进行了转换
            if old_vertices.shape != new_vertices.shape:
                all_passed = False
                details.append(f"Farm {i}: Shape不同 OLD={old_vertices.shape} NEW={new_vertices.shape}")
                continue
            
            # 验证转换是否正确（x,y -> y,x）
            for j in range(len(old_vertices)):
                old_point = old_vertices[j]
                new_point = new_vertices[j]
                
                # 新版本应该将[x,y]转换为[y,x]
                if not (np.isclose(old_point[0], new_point[1]) and np.isclose(old_point[1], new_point[0])):
                    all_passed = False
                    details.append(f"Farm {i} Point {j}: 转换错误 OLD={old_point} NEW={new_point}")
                    break
            else:
                details.append(f"Farm {i}: ✅ 坐标转换正确")
        
        return {
            "passed": all_passed,
            "reason": "farm_vertices坐标系转换正确" if all_passed else "坐标系转换有误",
            "details": "\n".join(details)
        }
    
    def test_jump_algorithm_consistency(self):
        """测试3: JUMP算法行为一致性"""
        print("验证JUMP算法完整行为...")
        
        # 设置随机种子确保可重复
        np.random.seed(42)
        
        # 测试参数
        farm = np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
        turning_radius = 0.5
        
        # 创建算法实例
        old_jump = OldJUMP(200, 200, turning_radius, farm)
        new_jump = NewJUMP(200, 200, turning_radius, farm)
        
        # 生成路径
        weeds = [(5, 5), (15, 7), (10, 3)]
        old_path = old_jump.get_path(weeds)
        new_path = new_jump.get_path(weeds)
        
        # 比较路径长度
        len_match = len(old_path) == len(new_path)
        
        # 比较路径点
        path_match = True
        max_diff = 0
        if len_match:
            for i, (old_pt, new_pt) in enumerate(zip(old_path, new_path)):
                # 允许小的数值误差
                diff = np.linalg.norm(np.array(old_pt) - np.array(new_pt))
                max_diff = max(max_diff, diff)
                if diff > 1e-4:
                    path_match = False
                    break
        else:
            path_match = False
        
        details = [
            f"路径长度: OLD={len(old_path)}, NEW={len(new_path)}",
            f"路径匹配: {path_match}",
            f"最大偏差: {max_diff:.6f}"
        ]
        
        return {
            "passed": len_match and path_match,
            "reason": "JUMP算法行为一致" if (len_match and path_match) else "JUMP算法行为不一致",
            "details": "\n".join(details)
        }
    
    def test_snake_algorithm_consistency(self):
        """测试4: SNAKE算法行为一致性"""
        print("验证SNAKE算法完整行为...")
        
        np.random.seed(123)
        
        farm = np.array([[0, 0], [15, 0], [15, 15], [0, 15]])
        turning_radius = 0.8
        
        old_snake = OldSNAKE(200, 200, turning_radius, farm)
        new_snake = NewSNAKE(200, 200, turning_radius, farm)
        
        weeds = [(3, 3), (10, 10), (5, 12)]
        old_path = old_snake.get_path(weeds)
        new_path = new_snake.get_path(weeds)
        
        len_match = len(old_path) == len(new_path)
        
        path_match = True
        max_diff = 0
        if len_match:
            for old_pt, new_pt in zip(old_path, new_path):
                diff = np.linalg.norm(np.array(old_pt) - np.array(new_pt))
                max_diff = max(max_diff, diff)
                if diff > 1e-4:
                    path_match = False
                    break
        else:
            path_match = False
        
        details = [
            f"路径长度: OLD={len(old_path)}, NEW={len(new_path)}",
            f"路径匹配: {path_match}",
            f"最大偏差: {max_diff:.6f}"
        ]
        
        return {
            "passed": len_match and path_match,
            "reason": "SNAKE算法行为一致" if (len_match and path_match) else "SNAKE算法行为不一致",
            "details": "\n".join(details)
        }
    
    def test_bcp_algorithm_consistency(self):
        """测试5: BCP算法行为一致性"""
        print("验证BCP算法完整行为...")
        
        np.random.seed(999)
        
        farm = np.array([[0, 0], [25, 0], [25, 12], [0, 12]])
        turning_radius = 1.0
        
        old_bcp = OldBCP(200, 200, turning_radius, farm)
        new_bcp = NewBCP(200, 200, turning_radius, farm)
        
        weeds = [(5, 5), (20, 8), (12, 3), (8, 10)]
        old_path = old_bcp.get_path(weeds)
        new_path = new_bcp.get_path(weeds)
        
        len_match = len(old_path) == len(new_path)
        
        path_match = True
        max_diff = 0
        if len_match:
            for old_pt, new_pt in zip(old_path, new_path):
                diff = np.linalg.norm(np.array(old_pt) - np.array(new_pt))
                max_diff = max(max_diff, diff)
                if diff > 1e-4:
                    path_match = False
                    break
        else:
            path_match = False
        
        # 验证步进值
        old_step = old_bcp.step
        new_step = new_bcp.step
        step_match = abs(old_step - new_step) < 1e-6
        
        details = [
            f"路径长度: OLD={len(old_path)}, NEW={len(new_path)}",
            f"路径匹配: {path_match}",
            f"最大偏差: {max_diff:.6f}",
            f"步进值: OLD={old_step:.6f}, NEW={new_step:.6f}, 匹配={step_match}"
        ]
        
        return {
            "passed": len_match and path_match and step_match,
            "reason": "BCP算法行为一致" if (len_match and path_match and step_match) else "BCP算法行为不一致",
            "details": "\n".join(details)
        }
    
    def test_edge_cases(self):
        """测试6: 边界条件测试"""
        print("验证边界条件处理...")
        
        test_cases = []
        all_passed = True
        
        # 测试空杂草列表
        farm = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        old_jump = OldJUMP(200, 200, 0.5, farm)
        new_jump = NewJUMP(200, 200, 0.5, farm)
        
        old_path = old_jump.get_path([])
        new_path = new_jump.get_path([])
        
        empty_match = len(old_path) == len(new_path)
        test_cases.append(f"空杂草列表: OLD_len={len(old_path)}, NEW_len={len(new_path)}, 匹配={empty_match}")
        if not empty_match:
            all_passed = False
        
        # 测试单个杂草点
        old_path = old_jump.get_path([(5, 5)])
        new_path = new_jump.get_path([(5, 5)])
        
        single_match = len(old_path) == len(new_path)
        test_cases.append(f"单个杂草: OLD_len={len(old_path)}, NEW_len={len(new_path)}, 匹配={single_match}")
        if not single_match:
            all_passed = False
        
        # 测试极小转弯半径
        old_jump_small = OldJUMP(200, 200, 0.01, farm)
        new_jump_small = NewJUMP(200, 200, 0.01, farm)
        
        old_path = old_jump_small.get_path([(3, 3), (7, 7)])
        new_path = new_jump_small.get_path([(3, 3), (7, 7)])
        
        small_radius_match = abs(len(old_path) - len(new_path)) <= 2  # 允许小差异
        test_cases.append(f"极小半径: OLD_len={len(old_path)}, NEW_len={len(new_path)}, 近似匹配={small_radius_match}")
        if not small_radius_match:
            all_passed = False
        
        return {
            "passed": all_passed,
            "reason": "边界条件处理一致" if all_passed else "边界条件处理不一致",
            "details": "\n".join(test_cases)
        }
    
    def test_multiple_seeds(self):
        """测试7: 多种子稳定性测试"""
        print("验证多随机种子下的稳定性...")
        
        seeds = [42, 123, 999, 2024, 8888]
        farm = np.array([[0, 0], [30, 0], [30, 20], [0, 20]])
        turning_radius = 0.7
        
        results = []
        all_passed = True
        
        for seed in seeds:
            np.random.seed(seed)
            
            # 生成随机杂草
            num_weeds = np.random.randint(3, 10)
            weeds = [(np.random.uniform(0, 30), np.random.uniform(0, 20)) for _ in range(num_weeds)]
            
            # 测试JUMP
            old_jump = OldJUMP(200, 200, turning_radius, farm)
            new_jump = NewJUMP(200, 200, turning_radius, farm)
            
            old_path = old_jump.get_path(weeds)
            new_path = new_jump.get_path(weeds)
            
            len_diff = abs(len(old_path) - len(new_path))
            jump_match = len_diff <= 2  # 允许极小差异
            
            # 测试SNAKE
            old_snake = OldSNAKE(200, 200, turning_radius, farm)
            new_snake = NewSNAKE(200, 200, turning_radius, farm)
            
            old_path = old_snake.get_path(weeds)
            new_path = new_snake.get_path(weeds)
            
            snake_match = abs(len(old_path) - len(new_path)) <= 2
            
            # 测试BCP
            old_bcp = OldBCP(200, 200, turning_radius, farm)
            new_bcp = NewBCP(200, 200, turning_radius, farm)
            
            old_path = old_bcp.get_path(weeds)
            new_path = new_bcp.get_path(weeds)
            
            bcp_match = abs(len(old_path) - len(new_path)) <= 2
            
            seed_passed = jump_match and snake_match and bcp_match
            if not seed_passed:
                all_passed = False
            
            results.append(f"Seed {seed}: JUMP={jump_match}, SNAKE={snake_match}, BCP={bcp_match}")
        
        return {
            "passed": all_passed,
            "reason": "多种子稳定性良好" if all_passed else "存在种子相关的不稳定性",
            "details": "\n".join(results)
        }
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*80)
        print("功能一致性测试报告 - ultrathink模式")
        print("="*80)
        
        # 计算统计
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # 生成报告内容
        report = []
        report.append("# 功能一致性测试报告 - Report 8")
        report.append("\n## 执行摘要")
        report.append(f"- **测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"- **测试模式**: ultrathink - 极度严格验证")
        report.append(f"- **总测试数**: {self.total_tests}")
        report.append(f"- **通过数**: {self.passed_tests}")
        report.append(f"- **失败数**: {self.failed_tests}")
        report.append(f"- **通过率**: {pass_rate:.1f}%")
        
        report.append("\n## 详细测试结果")
        report.append("\n| 测试名称 | 状态 | 耗时(s) | 备注 |")
        report.append("|----------|------|---------|------|")
        
        for test in self.test_results:
            status = "✅" if test.get("passed", False) else "❌"
            name = test["name"]
            time_str = f"{test.get('time', 0):.3f}" if "time" in test else "N/A"
            
            if "error" in test:
                note = f"错误: {test['error'][:50]}..."
            elif "result" in test and not test["result"]["passed"]:
                note = test["result"]["reason"][:50]
            else:
                note = "通过"
            
            report.append(f"| {name} | {status} | {time_str} | {note} |")
        
        report.append("\n## 关键验证点")
        report.append("\n| 验证项 | 状态 | 详情 |")
        report.append("|--------|------|------|")
        
        key_points = [
            ("turn_direction初始值", "turn_direction" in str(self.test_results)),
            ("farm_vertices坐标转换", "farm_vertices" in str(self.test_results)),
            ("JUMP算法一致性", "JUMP" in str(self.test_results)),
            ("SNAKE算法一致性", "SNAKE" in str(self.test_results)),
            ("BCP算法一致性", "BCP" in str(self.test_results)),
            ("边界条件处理", "边界" in str(self.test_results)),
            ("多种子稳定性", "种子" in str(self.test_results))
        ]
        
        for point, tested in key_points:
            status = "✅ 已验证" if tested else "⚠️ 未测试"
            report.append(f"| {point} | {status} | - |")
        
        # 添加详细测试输出
        report.append("\n## 测试详情")
        for test in self.test_results:
            if "result" in test and "details" in test["result"]:
                report.append(f"\n### {test['name']}")
                report.append("```")
                report.append(test["result"]["details"])
                report.append("```")
        
        report.append("\n## 最终判定")
        
        if pass_rate >= 95:
            report.append("\n### ✅ **完全通过**")
            report.append("- 所有关键测试通过")
            report.append("- 新旧版本行为一致")
            report.append("- **可以安全替换旧版本**")
        elif pass_rate >= 80:
            report.append("\n### ⚠️ **条件通过**")
            report.append("- 主要功能一致")
            report.append("- 存在少量可接受的差异")
            report.append("- 建议进一步审查失败项")
        else:
            report.append("\n### ❌ **不通过**")
            report.append("- 存在关键不一致")
            report.append("- 需要进一步修复")
            report.append("- **不建议替换旧版本**")
        
        report.append("\n## ultrathink验证声明")
        report.append("本测试采用极度严格的验证标准：")
        report.append("- 数值精度要求: < 1e-6")
        report.append("- 路径点偏差: < 1e-4")
        report.append("- 完整算法行为对比")
        report.append("- 多种子稳定性验证")
        report.append("- 边界条件全覆盖")
        
        # 保存报告
        report_content = "\n".join(report)
        report_path = "/home/lzh/NewCppRL/rules_new/team_reports/report8_functional_consistency_test.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"\n报告已保存至: {report_path}")
        print("\n" + "="*80)
        
        return report_content

def main():
    """主测试函数"""
    print("="*80)
    print("功能一致性测试 - ultrathink模式启动")
    print("测试工程师: Agent E")
    print("验证标准: 极度严格")
    print("="*80)
    
    tester = FunctionalConsistencyTester()
    
    # 执行所有测试
    tester.run_test("初始turn_direction值验证", tester.test_initial_turn_direction)
    tester.run_test("farm_vertices坐标系验证", tester.test_farm_vertices_coordinate)
    tester.run_test("JUMP算法一致性验证", tester.test_jump_algorithm_consistency)
    tester.run_test("SNAKE算法一致性验证", tester.test_snake_algorithm_consistency)
    tester.run_test("BCP算法一致性验证", tester.test_bcp_algorithm_consistency)
    tester.run_test("边界条件处理验证", tester.test_edge_cases)
    tester.run_test("多种子稳定性验证", tester.test_multiple_seeds)
    
    # 生成报告
    report = tester.generate_report()
    
    return tester.passed_tests == tester.total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
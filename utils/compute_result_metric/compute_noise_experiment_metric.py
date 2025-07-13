#!/usr/bin/env python3
"""
计算三种方法在不同场景和生成方式下的平均指标，并与OURS方法对比
"""

from typing import Dict, List, Tuple
from collections import defaultdict

# 定义数据结构
MethodData = Dict[str, Dict[str, Dict[str, float]]]  # {scenario: {generation: {metric: value}}}
FinalMetrics = Dict[str, float]  # {metric: average_value}


def parse_data() -> Dict[str, MethodData]:
    """解析原始数据，返回结构化的数据字典"""

    # 初始化数据结构
    methods_data = {
        "[0.05, 5, 0.2]": defaultdict(lambda: defaultdict(dict)),
        "[0.02, 2, 0.1]": defaultdict(lambda: defaultdict(dict)),
        "[0.01, 1, 0.05]": defaultdict(lambda: defaultdict(dict)),
        "OURS": defaultdict(lambda: defaultdict(dict))
    }

    # 原始数据
    raw_data = {
        "sac_our_model_con3_[0.05, 5, 0.2]_easy.csv": {
            "Gau": {"cover_90": 1774.2234, "cover_95": 1910.2381, "cover_98": 2212.8579, "collapse_rate": 0.0000},
            "Uni": {"cover_90": 1862.7442, "cover_95": 2005.9876, "cover_98": 2359.6782, "collapse_rate": 0.0000}
        },
        "sac_our_model_con3_[0.05, 5, 0.2]_medium.csv": {
            "Gau": {"cover_90": 2075.2368, "cover_95": 2925.0847, "cover_98": 2908.9444, "collapse_rate": 0.0000},
            "Uni": {"cover_90": 2198.4236, "cover_95": 2390.5162, "cover_98": 2985.2649, "collapse_rate": 0.0000}
        },
        "sac_our_model_con3_[0.05, 5, 0.2]_hard.csv": {
            "Gau": {"cover_90": 2483.7997, "cover_95": 2841.4937, "cover_98": 3870.2425, "collapse_rate": 0.2000},
            "Uni": {"cover_90": 2895.6045, "cover_95": 3090.8232, "cover_98": 5220.4571, "collapse_rate": 0.1600}
        },
        "sac_our_model_con3_[0.02, 2, 0.1]_easy.csv": {
            "Gau": {"cover_90": 1854.4129, "cover_95": 1908.2742, "cover_98": 2209.1279, "collapse_rate": 0.0000},
            "Uni": {"cover_90": 1951.2376, "cover_95": 2061.4687, "cover_98": 2370.0915, "collapse_rate": 0.0000}
        },
        "sac_our_model_con3_[0.02, 2, 0.1]_medium.csv": {
            "Gau": {"cover_90": 2048.2236, "cover_95": 2324.8327, "cover_98": 3230.3421, "collapse_rate": 0.0400},
            "Uni": {"cover_90": 2091.6589, "cover_95": 2298.6345, "cover_98": 3081.7645, "collapse_rate": 0.0000}
        },
        "sac_our_model_con3_[0.02, 2, 0.1]_hard.csv": {
            "Gau": {"cover_90": 2498.7124, "cover_95": 2776.5116, "cover_98": 3384.6578, "collapse_rate": 0.2200},
            "Uni": {"cover_90": 2768.8409, "cover_95": 2995.9327, "cover_98": 3649.8761, "collapse_rate": 0.2400}
        },
        "sac_our_model_con3_[0.01, 1, 0.05]_easy.csv": {
            "Gau": {"cover_90": 1703.2957, "cover_95": 1820.7013, "cover_98": 2143.9604, "collapse_rate": 0.0000},
            "Uni": {"cover_90": 1916.4527, "cover_95": 2070.8314, "cover_98": 2572.6411, "collapse_rate": 0.0000}
        },
        "sac_our_model_con3_[0.01, 1, 0.05]_medium.csv": {
            "Gau": {"cover_90": 1999.8742, "cover_95": 2258.2228, "cover_98": 2609.7439, "collapse_rate": 0.0000},
            "Uni": {"cover_90": 2270.3478, "cover_95": 2456.3324, "cover_98": 3245.2336, "collapse_rate": 0.0000}
        },
        "sac_our_model_con3_[0.01, 1, 0.05]_hard.csv": {
            "Gau": {"cover_90": 2345.6691, "cover_95": 2700.4256, "cover_98": 3057.6324, "collapse_rate": 0.2200},
            "Uni": {"cover_90": 2658.0440, "cover_95": 3002.8149, "cover_98": 3489.9741, "collapse_rate": 0.1600}
        }
    }

    # OURS方法的数据
    ours_data = {
        'Gaussian': {
            'OURS': {'Easy': [1827.7, 2249.0], 'Medium': [1962.7, 2656.9], 'Hard': [2352.5, 3051.9]}
        },
        'Uniform': {
            'OURS': {'Easy': [1135.5, 2569.4], 'Medium': [1819.8, 2342.2], 'Hard': [2646.0, 2845.2]}
        }
    }

    # 解析三种方法的数据到结构化格式
    for filename, data in raw_data.items():
        # 提取方法名和场景
        parts = filename.replace("sac_our_model_con3_", "").replace(".csv", "").rsplit("_", 1)
        method = parts[0]
        scenario = parts[1]

        # 存储数据
        for gen_type, metrics in data.items():
            methods_data[method][scenario][gen_type] = metrics

    # 添加OURS方法的数据
    scenarios_map = {'Easy': 'easy', 'Medium': 'medium', 'Hard': 'hard'}
    gen_map = {'Gaussian': 'Gau', 'Uniform': 'Uni'}

    for gen_orig, gen_data in ours_data.items():
        gen_type = gen_map[gen_orig]
        for scenario_orig, values in gen_data['OURS'].items():
            scenario = scenarios_map[scenario_orig]
            methods_data["OURS"][scenario][gen_type] = {
                "cover_90": values[0],
                "cover_98": values[1]
            }

    return dict(methods_data)


def calculate_scenario_average(scenario_data: Dict[str, Dict[str, float]], method_name: str = None) -> Dict[str, float]:
    """计算单个场景下Gau和Uni的平均值"""
    averages = {}

    # 根据方法确定需要计算的指标
    if method_name == "OURS":
        metrics = ["cover_90", "cover_98"]
    else:
        metrics = ["cover_90", "cover_95", "cover_98", "collapse_rate"]

    for metric in metrics:
        if metric in scenario_data.get("Gau", {}) and metric in scenario_data.get("Uni", {}):
            gau_value = scenario_data["Gau"][metric]
            uni_value = scenario_data["Uni"][metric]
            averages[metric] = (gau_value + uni_value) / 2

    return averages


def calculate_method_average(method_data: MethodData, method_name: str = None) -> FinalMetrics:
    """计算单个方法在所有场景下的最终平均值"""
    scenarios = ["easy", "medium", "hard"]

    # 根据方法确定需要计算的指标
    if method_name == "OURS":
        metrics = ["cover_90", "cover_98"]
    else:
        metrics = ["cover_90", "cover_95", "cover_98", "collapse_rate"]

    # 存储每个场景的平均值
    scenario_averages = []

    # 计算每个场景的平均值
    for scenario in scenarios:
        if scenario in method_data:
            avg = calculate_scenario_average(method_data[scenario], method_name)
            scenario_averages.append(avg)

    # 计算最终平均值
    final_averages = {}
    for metric in metrics:
        if scenario_averages and metric in scenario_averages[0]:
            values = [avg[metric] for avg in scenario_averages if metric in avg]
            final_averages[metric] = sum(values) / len(values)

    return final_averages


def calculate_improvement(method_value: float, ours_value: float) -> float:
    """计算相对于OURS的增长百分比"""
    return ((method_value - ours_value) / ours_value) * 100


def main():
    """主函数：计算并打印所有方法的最终平均指标"""

    # 解析数据
    all_data = parse_data()

    # 方法列表
    methods = ["[0.05, 5, 0.2]", "[0.02, 2, 0.1]", "[0.01, 1, 0.05]", "OURS"]

    print("=" * 80)
    print("四种方法的最终平均指标")
    print("=" * 80)
    print()

    # 存储计算结果
    results = {}

    # 计算并打印每个方法的结果
    for method in methods:
        if method in all_data:
            final_metrics = calculate_method_average(all_data[method], method)
            results[method] = final_metrics

            print(f"方法 {method}:")
            if "cover_90" in final_metrics:
                print(f"  cover_90:      {final_metrics['cover_90']:10.4f}")
            if "cover_95" in final_metrics:
                print(f"  cover_95:      {final_metrics['cover_95']:10.4f}")
            if "cover_98" in final_metrics:
                print(f"  cover_98:      {final_metrics['cover_98']:10.4f}")
            if "collapse_rate" in final_metrics:
                print(f"  collapse_rate: {final_metrics['collapse_rate']:10.4f}")
            print()

    # 打印汇总表格
    print("-" * 80)
    print("汇总表格：")
    print("-" * 80)
    print(f"{'方法':<20} {'cover_90':>12} {'cover_95':>12} {'cover_98':>12} {'collapse_rate':>14}")
    print("-" * 80)

    for method in methods:
        if method in all_data:
            metrics = results[method]
            line = f"{method:<20}"
            line += f" {metrics.get('cover_90', 0):>12.4f}" if 'cover_90' in metrics else " " * 13 + "-"
            line += f" {metrics.get('cover_95', 0):>12.4f}" if 'cover_95' in metrics else " " * 13 + "-"
            line += f" {metrics.get('cover_98', 0):>12.4f}" if 'cover_98' in metrics else " " * 13 + "-"
            line += f" {metrics.get('collapse_rate', 0):>14.4f}" if 'collapse_rate' in metrics else " " * 15 + "-"
            print(line)

    print("-" * 80)

    # 计算并打印相对于OURS的增长百分比
    if "OURS" in results:
        print("\n" + "=" * 80)
        print("相对于OURS方法的增长百分比")
        print("=" * 80)
        print()

        ours_metrics = results["OURS"]
        comparison_methods = ["[0.05, 5, 0.2]", "[0.02, 2, 0.1]", "[0.01, 1, 0.05]"]

        for method in comparison_methods:
            if method in results:
                print(f"方法 {method}:")
                method_metrics = results[method]

                # 计算cover_90的增长
                if "cover_90" in method_metrics and "cover_90" in ours_metrics:
                    improvement_90 = calculate_improvement(
                        method_metrics["cover_90"],
                        ours_metrics["cover_90"]
                    )
                    print(f"  cover_90 增长: {improvement_90:+.2f}%")

                # 计算cover_98的增长
                if "cover_98" in method_metrics and "cover_98" in ours_metrics:
                    improvement_98 = calculate_improvement(
                        method_metrics["cover_98"],
                        ours_metrics["cover_98"]
                    )
                    print(f"  cover_98 增长: {improvement_98:+.2f}%")

                print()

        # 打印增长百分比汇总表
        print("-" * 80)
        print("增长百分比汇总表：")
        print("-" * 80)
        print(f"{'方法':<20} {'cover_90增长':>15} {'cover_98增长':>15}")
        print("-" * 80)

        for method in comparison_methods:
            if method in results:
                method_metrics = results[method]
                line = f"{method:<20}"

                if "cover_90" in method_metrics and "cover_90" in ours_metrics:
                    improvement_90 = calculate_improvement(
                        method_metrics["cover_90"],
                        ours_metrics["cover_90"]
                    )
                    line += f" {improvement_90:>14.2f}%"
                else:
                    line += " " * 16 + "-"

                if "cover_98" in method_metrics and "cover_98" in ours_metrics:
                    improvement_98 = calculate_improvement(
                        method_metrics["cover_98"],
                        ours_metrics["cover_98"]
                    )
                    line += f" {improvement_98:>14.2f}%"
                else:
                    line += " " * 16 + "-"

                print(line)

        print("-" * 80)


if __name__ == "__main__":
    main()
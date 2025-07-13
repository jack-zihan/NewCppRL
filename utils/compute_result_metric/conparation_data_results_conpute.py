import pandas as pd
import numpy as np

# Define method names
methods = ['REACT', 'BCP', 'JUMP', 'SNAKE', 'R_SNAKE', 'E2ECO', 'OURS']
rule_based_methods = ['REACT', 'BCP', 'JUMP', 'SNAKE', 'R_SNAKE']

# Define the data structure: [Gaussian, Uniform], each with Easy, Medium, Hard for each method
# Format: [L90, L98, CR]
data = {
    'Gaussian': {
        'REACT': {'Easy': [4279.4, 5951.3], 'Medium': [3361.1, 7709.1]},
        'BCP': {'Easy': [13831.4, 15159.2], 'Medium': [12395.4, np.nan]},
        'JUMP': {'Easy': [8397.7, 9139.6], 'Medium': [10352.0, 12633.9]},
        'SNAKE': {'Easy': [4506.7, 5153.9], 'Medium': [4274.0, 5452.8]},
        'R_SNAKE': {'Easy': [4838.4, 5440.1], 'Medium': [4029.0, np.nan]},
        'E2ECO': {'Easy': [2790.0, 3904.9], 'Medium': [2208.4, 3273.8], 'Hard': [2578.6, 4542.5]},
        'OURS': {'Easy': [1827.7, 2249.0], 'Medium': [1962.7, 2656.9], 'Hard': [2352.5, 3051.9]}
    },
    'Uniform': {
        'REACT': {'Easy': [5273.2, 6112.3], 'Medium': [3564.5, 5505.5]},
        'BCP': {'Easy': [14414.9, 15765.2], 'Medium': [10479.0, 12845.5]},
        'JUMP': {'Easy': [8536.0, 9208.6], 'Medium': [9809.3, 10188.2]},
        'SNAKE': {'Easy': [4731.5, 5619.3], 'Medium': [5515.9, 7091.3]},
        'R_SNAKE': {'Easy': [5011.5, 6206.4], 'Medium': [np.nan, np.nan]},
        'E2ECO': {'Easy': [2376.1, 3240.2], 'Medium': [2492.6, 4229.0], 'Hard': [2960.4, 4891.3]},
        'OURS': {'Easy': [1135.5, 2569.4], 'Medium': [1819.8, 2342.2], 'Hard': [2646.0, 2845.2]}
    }
}

# ----------- Calculation 1: 51%-92% reduction vs rule-based in Easy/Medium only -----------
def reduction_vs_rule_based_fixed():
    reductions = []
    reductions_90 = []
    reductions_98 = []

    for method in rule_based_methods:
        l90_list, l98_list = [], []

        for dist in ['Gaussian', 'Uniform']:
            for difficulty in ['Easy', 'Medium']:
                # 该方法必须有该场景的结果才计入
                if difficulty in data[dist][method]:
                    l90, l98 = data[dist][method][difficulty]
                    if not np.isnan(l90): l90_list.append(l90)
                    if not np.isnan(l98): l98_list.append(l98)

        if l90_list and l98_list:
            avg_l90 = np.mean(l90_list)
            avg_l98 = np.mean(l98_list)

            # OURS 同样计算相同组合的平均值
            ours_l90_list, ours_l98_list = [], []
            for dist in ['Gaussian', 'Uniform']:
                for difficulty in ['Easy', 'Medium']:
                    l90, l98 = data[dist]['OURS'][difficulty]
                    ours_l90_list.append(l90)
                    ours_l98_list.append(l98)
            avg_ours_l90 = np.mean(ours_l90_list)
            avg_ours_l98 = np.mean(ours_l98_list)

            red_l90 = (avg_l90 - avg_ours_l90) / avg_l90 * 100
            red_l98 = (avg_l98 - avg_ours_l98) / avg_l98 * 100
            red_average = (red_l90 + red_l98) / 2
            reductions.append(red_average)
            reductions_90.append(red_l90)
            reductions_98.append(red_l98)
        else:
            raise Exception

    return [(round(min(reduc), 1), round(max(reduc), 1)) for reduc in [reductions, reductions_90, reductions_98]],


# ----------- Calculation 2: OURS path length increase from Medium to Hard -----------
def ours_path_increase_medium_to_hard():
    l90_medium_list, l98_medium_list = [], []
    l90_hard_list, l98_hard_list = [], []

    for dist in ['Gaussian', 'Uniform']:
        l90_m, l98_m = data[dist]['OURS']['Medium']
        l90_h, l98_h = data[dist]['OURS']['Hard']
        l90_medium_list.append(l90_m)
        l98_medium_list.append(l98_m)
        l90_hard_list.append(l90_h)
        l98_hard_list.append(l98_h)

    # 先算平均
    avg_l90_medium = np.mean(l90_medium_list)
    avg_l98_medium = np.mean(l98_medium_list)
    avg_l90_hard = np.mean(l90_hard_list)
    avg_l98_hard = np.mean(l98_hard_list)

    # 再算增长比例
    inc_l90 = (avg_l90_hard - avg_l90_medium) / avg_l90_medium * 100
    inc_l98 = (avg_l98_hard - avg_l98_medium) / avg_l98_medium * 100

    avg_increase = np.mean([inc_l90, inc_l98])
    return round(avg_increase, 1)


# ----------- Calculation 3: OURS vs E2ECO average L90/L98 reduction over all difficulties -----------
def reduction_ours_vs_e2eco():
    ours_l90_list, ours_l98_list = [], []
    e2eco_l90_list, e2eco_l98_list = [], []

    for dist in ['Gaussian', 'Uniform']:
        for difficulty in ['Easy', 'Medium', 'Hard']:
            if difficulty in data[dist]['OURS'] and difficulty in data[dist]['E2ECO']:
                ours_l90, ours_l98 = data[dist]['OURS'][difficulty]
                e2eco_l90, e2eco_l98 = data[dist]['E2ECO'][difficulty]
                ours_l90_list.append(ours_l90)
                ours_l98_list.append(ours_l98)
                e2eco_l90_list.append(e2eco_l90)
                e2eco_l98_list.append(e2eco_l98)

    avg_ours_l90 = np.mean(ours_l90_list)
    avg_ours_l98 = np.mean(ours_l98_list)
    avg_e2eco_l90 = np.mean(e2eco_l90_list)
    avg_e2eco_l98 = np.mean(e2eco_l98_list)

    red_l90 = (avg_e2eco_l90 - avg_ours_l90) / avg_e2eco_l90 * 100
    red_l98 = (avg_e2eco_l98 - avg_ours_l98) / avg_e2eco_l98 * 100

    return [round(np.mean([red_l90, red_l98]), 1), red_l90, red_l98]

def detailed_reduction_ours_vs_e2eco():
    results = []

    for dist in ['Gaussian', 'Uniform']:
        for difficulty in ['Easy', 'Medium', 'Hard']:
            if difficulty in data[dist]['OURS'] and difficulty in data[dist]['E2ECO']:
                ours_l90, ours_l98 = data[dist]['OURS'][difficulty]
                e2eco_l90, e2eco_l98 = data[dist]['E2ECO'][difficulty]

                red_l90 = (e2eco_l90 - ours_l90) / e2eco_l90 * 100
                red_l98 = (e2eco_l98 - ours_l98) / e2eco_l98 * 100
                avg_e2eco = (e2eco_l90 + e2eco_l98) / 2
                avg_ours = (ours_l90 + ours_l98) / 2
                red_avg = (avg_e2eco - avg_ours) / avg_e2eco * 100

                results.append({
                    'Distribution': dist,
                    'Difficulty': difficulty,
                    'E2ECO_L90': e2eco_l90,
                    'OURS_L90': ours_l90,
                    'Reduction_L90(%)': round(red_l90, 1),
                    'E2ECO_L98': e2eco_l98,
                    'OURS_L98': ours_l98,
                    'Reduction_L98(%)': round(red_l98, 1),
                    'E2ECO_Avg': round(avg_e2eco, 1),
                    'OURS_Avg': round(avg_ours, 1),
                    'Reduction_Avg(%)': round(red_avg, 1)
                })

    return results


print(reduction_vs_rule_based_fixed(), ours_path_increase_medium_to_hard(), reduction_ours_vs_e2eco())

# from pprint import pprint
# pprint(detailed_reduction_ours_vs_e2eco())

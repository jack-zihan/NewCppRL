"""调试障碍物渲染问题"""
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v2 import CppEnv
import matplotlib.pyplot as plt


def debug_obstacle_creation():
    """调试障碍物创建过程"""
    print("=" * 50)
    print("调试障碍物渲染问题")
    print("=" * 50)
    
    # 创建环境
    env = CppEnv()
    
    # 重置环境
    obs, info = env.reset(seed=42)
    
    # 检查maps_dict
    print("\n1. 检查地图字典内容:")
    if hasattr(env, 'maps_dict'):
        for key in env.maps_dict.keys():
            print(f"  - {key}: shape={env.maps_dict[key].shape}, "
                  f"sum={env.maps_dict[key].sum()}, "
                  f"dtype={env.maps_dict[key].dtype}")
    else:
        print("  ERROR: maps_dict 不存在！")
        return
    
    # 特别检查obstacle地图
    print("\n2. 障碍物地图详细信息:")
    obstacle_map = env.maps_dict.get('obstacle')
    if obstacle_map is not None:
        print(f"  - 形状: {obstacle_map.shape}")
        print(f"  - 数据类型: {obstacle_map.dtype}")
        print(f"  - 最小值: {obstacle_map.min()}")
        print(f"  - 最大值: {obstacle_map.max()}")
        print(f"  - 非零像素数: {np.sum(obstacle_map > 0)}")
        print(f"  - 唯一值: {np.unique(obstacle_map)}")
        
        # 检查是否有障碍物生成
        if np.sum(obstacle_map > 0) == 0:
            print("  ⚠️ 警告：障碍物地图为空！")
        
        # 保存障碍物地图为图像
        if np.sum(obstacle_map > 0) > 0:
            obstacle_visual = (obstacle_map * 255).astype(np.uint8)
            cv2.imwrite('debug_obstacle_map.png', obstacle_visual)
            print("  - 已保存障碍物地图到 debug_obstacle_map.png")
    else:
        print("  ERROR: 障碍物地图不存在！")
    
    # 检查配置
    print("\n3. 检查环境配置:")
    if hasattr(env, 'config'):
        config = env.config
        print(f"  - num_obstacles_range: {config.num_obstacles_range}")
        print(f"  - obstacle_size_range: {config.obstacle_size_range}")
        print(f"  - use_box_boundary: {config.use_box_boundary}")
        print(f"  - obstacle_min_distance_to_edge: {config.obstacle_min_distance_to_edge}")
        print(f"  - obstacle_min_distance_to_agent: {config.obstacle_min_distance_to_agent}")
    
    # 获取渲染图像
    print("\n4. 检查渲染输出:")
    try:
        # 尝试获取渲染图像
        if hasattr(env, 'render'):
            render_img = env.render()
            if render_img is not None:
                print(f"  - 渲染图像形状: {render_img.shape}")
                print(f"  - 渲染图像类型: {render_img.dtype}")
                cv2.imwrite('debug_render.png', render_img)
                print("  - 已保存渲染图像到 debug_render.png")
            else:
                print("  - render() 返回 None")
    except Exception as e:
        print(f"  - 渲染时出错: {e}")
    
    # 测试障碍物生成逻辑
    print("\n5. 直接测试障碍物生成:")
    from envs_new.components.map.map_components import ObstacleCreator
    
    creator = ObstacleCreator()
    
    # 模拟配置
    class MockConfig:
        obstacle_min_distance_to_edge = 10
        obstacle_size_range = (5, 15)
    
    config = MockConfig()
    dimensions = (100, 100)
    rng = np.random.default_rng(42)
    
    # 创建单个障碍物
    obstacle = creator._create_random_obstacle(dimensions, config, rng)
    print(f"  - 生成的障碍物形状: {obstacle.shape}")
    print(f"  - 障碍物数据类型: {obstacle.dtype}")
    print(f"  - 障碍物顶点:\n{obstacle.reshape(-1, 2)}")
    
    # 创建测试地图并绘制障碍物
    test_map = np.zeros((100, 100), dtype=np.uint8)
    cv2.fillPoly(test_map, [obstacle], color=(1,))
    filled_pixels = np.sum(test_map > 0)
    print(f"  - 填充的像素数: {filled_pixels}")
    
    if filled_pixels == 0:
        print("  ⚠️ 警告：障碍物没有被正确填充！")
        print("  - 可能是坐标超出边界或形状无效")
    
    env.close()
    
    print("\n" + "=" * 50)
    print("调试完成，请检查生成的图像文件")
    print("=" * 50)


def test_original_vs_optimized():
    """对比原始和优化后的代码"""
    print("\n测试原始API vs 优化后的API:")
    
    # 测试两种创建旋转矩形的方法
    center = (50.0, 50.0)
    size = (30.0, 20.0)
    angle = 45.0
    
    # 方法1: 使用cv2.RotatedRect (如果存在)
    try:
        rect_obj = cv2.RotatedRect(center=center, size=size, angle=angle)
        points1 = rect_obj.points()
        print(f"  方法1 (cv2.RotatedRect): 成功")
        print(f"    points shape: {points1.shape}")
    except Exception as e:
        print(f"  方法1 (cv2.RotatedRect): 失败 - {e}")
        points1 = None
    
    # 方法2: 使用元组
    try:
        points2 = cv2.boxPoints((center, size, angle))
        print(f"  方法2 (元组): 成功")
        print(f"    points shape: {points2.shape}")
    except Exception as e:
        print(f"  方法2 (元组): 失败 - {e}")
        points2 = None
    
    # 比较结果
    if points1 is not None and points2 is not None:
        if np.allclose(points1, points2):
            print("  ✅ 两种方法结果相同")
        else:
            print("  ❌ 两种方法结果不同！")
            print(f"    差异: {np.max(np.abs(points1 - points2))}")


if __name__ == "__main__":
    debug_obstacle_creation()
    test_original_vs_optimized()
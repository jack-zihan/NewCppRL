from pathlib import Path as FilePath
import sys

# 设置基础路径（项目根目录）
BASE_DIR = FilePath(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def to_absolute_path(path):
    """将路径转换为绝对路径"""
    p = FilePath(path)
    return str(p if p.is_absolute() else BASE_DIR / p)

import math
import random
import dubins
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import HumanRendering
from matplotlib.path import Path
from shapely.geometry import Point, Polygon, LineString
from rules.config import Config
import os
import csv
from rules.env_make import get_env
import time

'''
获取position需要把xy对调 长宽都是对调 并且返回的rad是y方向相反的  real_radians 是正常00在左下的坐标系下的。
'''
store = None

env, _ = get_env()

task_type = "R_SNAKE"

agent_width = Config.CAR_WIDTH
sight_width = Config.SIGHT_WIDTH
sight_length = Config.SIGHT_LENGTH
agent_position = [env.agent.y, env.agent.x] 
W = Config.W
H = Config.H
# turning_radius = Config.TURNING_RADIUS
w_max_rad = abs(env.w_range.max) * (math.pi / 180)
turning_radius = env.v_range.max / w_max_rad
farm_vertices = env.min_area_rect[0][:, 0, ::-1]
init_weed = env.map_weed.sum()

# farm_vertices = Config.FARM_VERTICES

discovered = []
rad = 0
cover_90,cover_95, cover_98,cover,dist_list = -1, -1, -1, [], []

difficulty = "easy"
save_path = BASE_DIR / 'rules' / 'logs' / f"coverage_results_{task_type}_{difficulty}.csv"
weed_dist = "gaussian"
random_seed = 25
map_id = 2
done = False
overall_length = 0




def is_point_in_polygon(point, vertices):
    path = Path(vertices)
    return path.contains_point(point)


def find_longest_edge(vertices):
    max_length = 0
    longest_edge = None
    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i + 1) % len(vertices)]
        length = np.linalg.norm(end - start)
        if length > max_length:
            max_length = length
            longest_edge = (start, end)
    return longest_edge


longest_edge = find_longest_edge(farm_vertices)

def save_data_to_csv(file_path, weed_dist, random_seed, map_id, collapse, cover_90, cover_95, cover_98,cover, dist_list):
    file_path = to_absolute_path(file_path)  # 确保路径是绝对路径
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["weed_dist","random_seed", "map_id", "collapse", "cover_90", "cover_95", "cover_98", "cover", "dist_list"])
        cover_str = ",".join(map(str, cover))
        dist_str = ",".join(map(str, dist_list))
        writer.writerow([weed_dist, random_seed, map_id, collapse, cover_90, cover_95, cover_98,cover_str, dist_str])
def go(p2):  # verified
    global done
    global discovered
    global rad
    global agent_position
    global agent_width
    global cover_98
    global cover_95
    global cover_90
    global overall_length
    global cover
    global dist_list
    prev_position = agent_position
    radian = math.atan2(p2[1] - agent_position[1], p2[0] - agent_position[0])
    length = math.sqrt((p2[0] - agent_position[0]) ** 2 + (p2[1] - agent_position[1]) ** 2)
    delta_angle = - (radian - rad) % (2 * math.pi)
    delta_angle = delta_angle - 2 * math.pi if delta_angle > math.pi else delta_angle
    delta_angle = math.degrees(delta_angle)
    
    env.set_action_type("continuous")

    obs, reward, done, time_out, _ = env.step([length, delta_angle])

    agent_position = [env.agent.y, env.agent.x]
    
    distance = np.linalg.norm(np.array(agent_position) - np.array(prev_position))
    overall_length += distance
    rad = np.pi / 2 - math.radians(env.agent.direction)
    discovered = np.argwhere(np.logical_and(env.map_weed, np.logical_not(env.map_frontier)) == 1)
    discovered = [point for point in discovered if is_point_in_polygon(point, farm_vertices)]
    cover_rate = (init_weed - env.map_weed.sum()) / init_weed
    
    # methods = [np.floor, round, np.ceil, int]
    # agent_position_int = next(((int(m(agent_position[0])), int(m(agent_position[1]))) for m in methods
    #     if 0 <= (p := (int(m(agent_position[0])), int(m(agent_position[1]))))[0] < env.map_obstacle.shape[0] 
    #     and 0 <= p[1] < env.map_obstacle.shape[1] 
    #     and env.map_obstacle[p[0], p[1]] == 1), None)

    if cover_rate >= 0.98:
        cover_98 = overall_length
    elif cover_rate >= 0.95:
        cover_95 = overall_length
    elif cover_rate >= 0.90:
        cover_90 = overall_length
    cover.append(cover_rate)
    dist_list.append(overall_length)
    if done:
        if env.check_collision():
            save_data_to_csv(save_path, weed_dist, random_seed, map_id, 1, cover_90, cover_95, cover_98, cover, dist_list)
            exit()
        else:
            save_data_to_csv(save_path, weed_dist, random_seed, map_id, 0, cover_90, cover_95, cover_98, cover, dist_list)
            exit()
        
    # global store
    # if store is None:
    #     store = env.map_frontier


def navigate(goal):
    global agent_position
    vector = np.array(goal) - np.array(agent_position)
    distance = np.linalg.norm(vector)
    num_steps = int(distance // 2)
    step_vector = vector / num_steps
    waypoints = [agent_position + step_vector * i for i in range(1, num_steps + 1)]
    waypoints.append(goal)
    for p2 in waypoints:
        while abs(p2[0] - agent_position[0]) > 1 or abs(p2[1] - agent_position[1]) > 1:
            go(p2)


def dubins_navigate(p2, r):
    path = dubins.shortest_path((agent_position[0], agent_position[1], rad), (p2[0], p2[1], p2[2]), r)
    configurations, _ = path.sample_many(0.5)

    for point in configurations[1:]:
        navigate(list(point[:2]))
    
def dubins_navigate_obstacle(p2, r, max_attempts=5):
    path = dubins.shortest_path((agent_position[0], agent_position[1], rad), (p2[0], p2[1], p2[2]), r)
    configurations, _ = path.sample_many(0.5)

    for attempt in range(max_attempts):
        path_clear = True
        new_path = []
        
        for point in configurations[1:]:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < W and 0 <= y < H:
                if env.map_obstacle[y, x] == 1:
                    path_clear = False
                    # 尝试绕过障碍物，通过微调当前点的方向
                    detour_point = local_adjustment(point, r)
                    if detour_point is not None:
                        new_path.append(detour_point)
                    else:
                        print("Detour failed, obstacle is too close.")
                        break
                else:
                    new_path.append(point)

        if path_clear or len(new_path) == len(configurations) - 1:
            # 如果路径清晰，或者已经成功绕开障碍物
            for point in new_path:
                navigate(list(point[:2]))
            print("Path successfully navigated.")
            return True

    print("Failed to find a clear path after multiple attempts.")
    return False

def local_adjustment(point, r):
    # 基于当前点，尝试绕过障碍物，返回调整后的点
    # 调整策略：尝试调整方向或者移动位置
    detour_x = point[0] + random.uniform(-r, r)  # 在允许的范围内调整
    detour_y = point[1] + random.uniform(-r, r)
    if 0 <= detour_x < W and 0 <= detour_y < H and env.map_obstacle[int(detour_y), int(detour_x)] == 0:
        return [detour_x, detour_y, point[2]]  # 保持相同的方向
    return None



def find_longest_edge(farm_vertices):
    # 计算多边形每条边的长度
    num_vertices = len(farm_vertices)
    max_length = 0
    longest_edge = None

    # 循环通过每个点和它的下一个点（包括最后一个点和第一个点）
    for i in range(num_vertices):
        # 下一个点的索引，考虑最后一个点连接到第一个点
        next_index = (i + 1) % num_vertices
        # 当前边的两个端点
        point1 = farm_vertices[i]
        point2 = farm_vertices[next_index]
        # 计算边的长度
        length = np.linalg.norm(point2 - point1)

        # 更新最长边
        if length > max_length:
            max_length = length
            longest_edge = [point1, point2]

    return longest_edge


def find_nearest_point_jump(radian, p, coordinates):
    radian = - radian  # 因为这个函数里面要使用的是矩阵的坐标系 不是标准坐标系
    radian = radian % (2 * np.pi)
    if len(coordinates) == 0:
        return None, -1
    rotation_matrix = np.array([
        [np.cos(radian), -np.sin(radian)],
        [np.sin(radian), np.cos(radian)]
    ])

    p_rotated = np.dot(rotation_matrix, np.array(p))
    rotated_coords = [np.dot(rotation_matrix, np.array(c)) for c in coordinates]
    nearest_index = min(range(len(rotated_coords)), key=lambda i: abs(rotated_coords[i][0] - p_rotated[0]))
    nearest_point = coordinates[nearest_index]

    return nearest_point, nearest_index


def transform_to_local(global_coord, start, end):
    direction_vector = np.array(end) - np.array(start)
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    perp_vector = np.array([-unit_vector[1], unit_vector[0]])
    relative_coord = np.array(global_coord) - np.array(start)
    local_x = np.dot(relative_coord, unit_vector)
    local_y = np.dot(relative_coord, perp_vector)
    return np.array([local_x, local_y])


def find_lowest_point(start, end, discovered):  # 返回的是新坐标系下的点坐标
    if len(discovered) == 0:
        return None
    offsets = [abs(find_offset(start, end, coord)) for coord in discovered]
    min_index = min((idx for idx, offset in enumerate(offsets)), key=lambda idx: offsets[idx], default=None)

    return discovered[min_index]


def get_forward_jump(discovered, point, rad, vertical_r):
    rad_vector = np.array([np.cos(rad), np.sin(rad)])
    vertical_vector = np.array([np.cos(vertical_r), np.sin(vertical_r)])
    rad_forward = [p for p in discovered if np.dot(p - point, rad_vector) > 0]
    final_points = [p for p in rad_forward if np.dot(p - point, vertical_vector) > 0]

    return final_points


def get_forward_snake(discovered, point, rad):
    rad_vector = np.array([np.cos(rad), np.sin(rad)])
    final_points = [p for p in discovered if np.dot(p - point, rad_vector) > 0]

    return final_points


def get_forward_rsnake(discovered, point, rad, real_radians):
    forward_vector = np.array([np.cos(rad), np.sin(rad)])
    upward_vector = np.array([np.cos(real_radians + np.pi / 2), np.sin(real_radians + np.pi / 2)])
    final_points = [
        p for p in discovered
        if np.dot(p - point, forward_vector) > 0 and np.dot(p - point, upward_vector) > - 3 / 2 * sight_width
    ]

    return final_points


def find_nearest_point(p, coordinates, r):
    if len(coordinates) == 0:
        return None
    p = np.array(p)
    coordinates = np.array(coordinates)
    distances = np.sqrt(np.sum((coordinates - p) ** 2, axis=1))
    valid_indices = np.where(distances >= 2 * r)[0]

    # 如果没有找到任何有效点，则返回None
    if len(valid_indices) == 0:
        return None
    nearest_index = valid_indices[np.argmin(distances[valid_indices])]
    return coordinates[nearest_index]


def find_offset(start, end, point, real_radians=None):
    # 将输入的点转换成NumPy数组
    start = np.array(start)
    end = np.array(end)
    point = np.array(point)

    if real_radians is None:
        # 如果没有提供角度，使用start到end的向量
        line_vec = end - start
    else:
        # 根据角度计算新的方向向量
        # 这里我们假设方向向量的长度为end和start之间的距离
        line_length = np.linalg.norm(end - start)
        line_vec = np.array([np.cos(real_radians) * line_length, np.sin(real_radians) * line_length])

    # 计算从start到point的向量
    point_vec = point - start

    # 计算叉积（这里不取模，保留符号信息）
    cross_product = np.cross(line_vec, point_vec)

    # 计算直线方向向量的长度
    norm_line_vec = np.linalg.norm(line_vec)

    # 计算垂直距离（使用叉积的z分量和直线长度）
    distance = cross_product / norm_line_vec

    return distance


max_edge_points = find_longest_edge(farm_vertices)
dx = max_edge_points[1][0] - max_edge_points[0][0]
dy = max_edge_points[1][1] - max_edge_points[0][1]
real_radians = np.arctan2(dy, dx)
radians = - real_radians
real_radians = real_radians % (2 * np.pi) if real_radians >= 0 else (real_radians + 2 * np.pi) % (2 * np.pi)
start = [0, 0]
end = np.array([100 * np.cos(real_radians), 100 * np.sin(real_radians)])
poly_path = Path(farm_vertices)
y, x = np.mgrid[:H, :W]
coor = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # 创建 (x, y) 坐标列表
mask = np.zeros((H, W))
mask[poly_path.contains_points(coor).reshape(H, W)] = 1

polygon = Polygon(farm_vertices)
min_x, min_y = farm_vertices.min(axis=0)
max_x, max_y = farm_vertices.max(axis=0)
diag_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
rad_n = radians

# 生成路径点
path_points = []
y_offset = -diag_length + agent_width / 2
turn = True

check = 5000000
empty = 0
init_start, init_end = [], []
starting = False
if task_type == 'REACT':
    
    times = 0
    start_time = time.time()
    while times < 50:
        if time.time() - start_time > 300: 
            save_data_to_csv(save_path, weed_dist, random_seed, map_id, 0, cover_90, cover_95, cover_98, cover, dist_list)
            print("运行时间超过5分钟，程序已退出。")
            sys.exit()

        rand_goal = [random.uniform(0, W), random.uniform(0, H)]
        start = agent_position
        end = rand_goal
        x_points = []
        line = LineString([start, end])
        for i in np.arange(0, line.length, 1):
            interpolated_point = np.array(line.interpolate(i).coords[0])
            x_points.append(interpolated_point)

        valid_points = [point for point in x_points if
                        0 <= int(point[1]) < H and 0 <= int(point[0]) < W and mask[int(point[1]), int(point[0])] == 1]
        p_i = 0
        found = False
        if valid_points:
            goal_rad = math.atan2(end[1] - start[1], end[0] - start[0])
            dubins_navigate([valid_points[0][0], valid_points[0][1], goal_rad], turning_radius)
        while p_i < len(valid_points):
            weed = find_nearest_point(agent_position, discovered, 0)
            while weed is not None:
                found = True
                dubins_navigate([weed[0], weed[1], rad_n], turning_radius)
                weed = find_nearest_point(agent_position, discovered, 0)
            if found:
                break

            navigate(valid_points[p_i])
            p_i += 1
        times += 1

else:
    start_time = time.time()
    while y_offset < diag_length:
        if time.time() - start_time > 300: 
            save_data_to_csv(save_path, weed_dist, random_seed, map_id, 0, cover_90, cover_95, cover_98, cover, dist_list)
            print("运行时间超过5分钟，程序已退出。")
            sys.exit()
        x_points = []
        new_start = [start[0] + y_offset * np.cos(real_radians + np.pi / 2) - diag_length * np.cos(real_radians),
                     start[1] + y_offset * np.sin(real_radians + np.pi / 2) - diag_length * np.sin(real_radians)]
        new_end = [end[0] + y_offset * np.cos(real_radians + np.pi / 2) + diag_length * np.cos(real_radians),
                   end[1] + y_offset * np.sin(real_radians + np.pi / 2) + diag_length * np.sin(real_radians)]
        line = LineString([new_start, new_end])
        # if polygon.intersects(line):
        #     for i in np.arange(0, line.length, 1):
        #         interpolated_point = np.array(line.interpolate(i).coords[0])
        #         x_points.append(interpolated_point)
        
        for i in np.arange(0, line.length, 1):
            interpolated_point = np.array(line.interpolate(i).coords[0])
            x_points.append(interpolated_point)


        # valid_points = [point for point in x_points if
        #                 0 <= int(point[1]) < H and 0 <= int(point[0]) < W and mask[int(point[1]), int(point[0])] == 1]
        valid_points = [point for point in x_points if
                        0 <= int(point[1]) < H and 0 <= int(point[0]) < W and mask[int(point[1]), int(point[0])] == 1]

        if valid_points:
            if len(init_start) == 0:
                init_start = new_start
                init_end = new_end
            turn = not turn

        if int(y_offset) == int(check):  # 这个结束方式不严谨
            if len(valid_points) == 0:
                empty += 1
        if empty >= 100:
            break
        check = y_offset

        valid_points = valid_points if not turn else valid_points[::-1]
        if turn:
            rad_n = real_radians + np.pi
        else:
            rad_n = real_radians
        rad_n = (rad_n + math.pi) % (2 * math.pi) - math.pi

        if valid_points:
            if not starting:
                env.agent.x = valid_points[0][1]
                env.agent.y = valid_points[0][0]
                env.agent.direction = (math.degrees(np.pi / 2 - turning_radius) % 360)
                agent_position = valid_points[0]
                rad = turning_radius
                starting = True
            else:
                dubins_navigate([valid_points[0][0], valid_points[0][1], rad_n], turning_radius)
                
        if task_type == 'JUMP':
            p_i = 0
            while p_i < len(valid_points):
                if len(discovered) > 0:
                    # 过滤点
                    vertical = real_radians + np.pi / 2
                    vertical = vertical - 2 * np.pi if vertical > np.pi else vertical
                    forward = get_forward_jump(discovered, agent_position, rad_n, vertical)
                    weed, _ = find_nearest_point_jump(rad_n, agent_position, forward)
                    if weed is not None:
                        # 先行进到合适的位置，再jump
                        point, i = find_nearest_point_jump(rad_n, weed, valid_points)
                        if i < p_i + (4 * turning_radius) + 4 or i - (4 * turning_radius) < 0 or i + 4 * turning_radius >= len(valid_points) or i + (
                                4 * turning_radius) + 1 >= len(valid_points):
                            navigate(valid_points[i + 2]) if i + 2 < len(valid_points) else navigate(valid_points[-1])
                            p_i = i + 3
                            continue
                        navigate(valid_points[int(i - (4 * turning_radius))])
                        dubins_navigate([weed[0], weed[1], rad_n], turning_radius)
                        dubins_navigate([valid_points[int(i + (4 * turning_radius))][0], valid_points[int(i + (4 * turning_radius))][1], rad_n],
                                        turning_radius)
                        p_i = int(i + (4 * turning_radius) + 1)
                navigate(valid_points[p_i])
                p_i += 1

        elif task_type == 'SNAKE' or task_type == 'R_SNAKE':
            p_i = 0
            while p_i < len(valid_points):
                if task_type == 'SNAKE':
                    forward = get_forward_snake(discovered, agent_position, rad_n)
                elif task_type == 'R_SNAKE':
                    forward = get_forward_rsnake(discovered, agent_position, rad_n, real_radians)
                weed = find_nearest_point(agent_position, forward, turning_radius)
                if weed is not None:
                    dubins_navigate([weed[0], weed[1], rad_n], turning_radius)
                    points = []
                    start_point = agent_position
                    while polygon.contains(Point(start_point[0] + len(points) * np.cos(rad),
                                                 start_point[1] + len(points) * np.sin(rad))):
                        points.append(
                            (start_point[0] + len(points) * np.cos(rad), start_point[1] + len(points) * np.sin(rad)))
                    valid_points = points
                    p_i = 0
                if len(valid_points) > 0:
                    navigate(valid_points[p_i])
                p_i += 1
        elif task_type == 'BCP':
            p_i = 0
            while p_i < len(valid_points):
                navigate(valid_points[p_i])
                p_i += 1

        if task_type == 'JUMP':
            weed = find_lowest_point(init_start, init_end, discovered)
            if weed is not None:
                y_offset = min(y_offset + find_offset(new_start, new_end, weed, real_radians) + agent_width / 2,
                               y_offset + sight_width / 2, diag_length - agent_width / 2)  # !!！ weed[1] + b/2不严谨
            else:
                y_offset += sight_width / 2
        elif task_type == 'SNAKE' or task_type == 'R_SNAKE':
            vertical = real_radians + np.pi / 2
            vertical = vertical - 2 * np.pi if vertical > np.pi else vertical
            vertical_vector = np.array([np.cos(vertical), np.sin(vertical)])
            possible_dots = [point for point in discovered if np.dot(point - agent_position, vertical_vector) > 0]
            weed = find_lowest_point(init_start, init_end, possible_dots)
            if weed is not None:
                y_offset = min(y_offset + find_offset(new_start, new_end, weed) + agent_width / 2 + sight_width / 2, diag_length - agent_width / 2)
            else:
                y_offset = y_offset + sight_width / 2 + agent_width / 2
        else:
            y_offset += agent_width
save_data_to_csv(save_path, weed_dist, random_seed, map_id, 0, cover_90, cover_95, cover_98, cover, dist_list)
print('verified')

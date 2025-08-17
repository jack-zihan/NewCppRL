"""
状态变量映射配置
定义新旧环境之间的状态变量对应关系
以新环境的命名为标准
"""

# 新环境 -> 旧环境的状态映射
STATE_MAPPING = {
    # =========================
    # Agent相关状态
    # =========================
    'agent_manager.agent.position': {
        'old_path': ['agent.x', 'agent.y'],
        'type': 'tuple',
        'description': 'Agent当前位置'
    },
    'agent_manager.agent.direction': {
        'old_path': 'agent.direction',
        'type': 'float',
        'description': 'Agent朝向角度(度)'
    },
    'agent_manager.agent.velocity': {
        'old_path': 'velocity',
        'type': 'float',
        'description': '线速度'
    },
    'agent_manager.agent.angular_velocity': {
        'old_path': 'w',
        'type': 'float',
        'description': '角速度'
    },
    'agent_manager.collision_count': {
        'old_path': 'collision_num',
        'type': 'int',
        'description': '碰撞次数'
    },
    
    # =========================
    # 地图相关状态
    # =========================
    'map_manager.weed_map': {
        'old_path': 'map_weed',
        'type': 'ndarray',
        'description': '杂草地图'
    },
    'map_manager.noisy_weed_map': {
        'old_path': 'map_weed_noisy',
        'type': 'ndarray',
        'description': '带噪声的杂草地图'
    },
    'map_manager.original_weed_map': {
        'old_path': 'map_weed_ori',
        'type': 'ndarray',
        'description': '原始杂草地图'
    },
    'map_manager.frontier_map': {
        'old_path': 'map_frontier',
        'type': 'ndarray',
        'description': '边界地图'
    },
    'map_manager.obstacle_map': {
        'old_path': 'map_obstacle',
        'type': 'ndarray',
        'description': '障碍物地图'
    },
    'map_manager.boundary_map': {
        'old_path': 'map_boundary',
        'type': 'ndarray',
        'description': '边界地图'
    },
    'map_manager.trajectory_map': {
        'old_path': 'map_trajectory',
        'type': 'ndarray',
        'description': '轨迹地图'
    },
    'map_manager.mist_map': {
        'old_path': 'map_mist',
        'type': 'ndarray',
        'description': '迷雾地图'
    },
    
    # =========================
    # 度量相关状态
    # =========================
    'metrics_manager.current_metrics.weed_count': {
        'old_path': 'weed_num_t',
        'type': 'int',
        'description': '当前剩余杂草数'
    },
    'metrics_manager.current_metrics.initial_weed_count': {
        'old_path': 'weed_num',
        'type': 'int',
        'description': '初始杂草总数'
    },
    'metrics_manager.current_metrics.frontier_count': {
        'old_path': 'frontier_num_t',
        'type': 'int',
        'description': '当前剩余边界数'
    },
    'metrics_manager.current_metrics.initial_frontier_count': {
        'old_path': 'frontier_num',
        'type': 'int',
        'description': '初始边界总数'
    },
    
    # =========================
    # 环境状态管理器中的状态信息
    # =========================
    'env_state.get_info("agent_position").current': {
        'old_path': ['agent.x', 'agent.y'],
        'type': 'tuple',
        'description': '当前位置（StateVariable格式）'
    },
    'env_state.get_info("agent_position").last': {
        'old_path': None,  # 旧环境可能在step中临时记录
        'type': 'tuple',
        'description': '上一帧位置'
    },
    'env_state.get_info("step_count").current': {
        'old_path': 'current_step',
        'type': 'int',
        'description': '当前步数'
    },
    
    # =========================
    # 环境配置和参数
    # =========================
    'config.dimensions': {
        'old_path': 'dimensions',
        'type': 'tuple',
        'description': '地图尺寸(width, height)'
    },
    'config.action_space_type': {
        'old_path': 'action_space_type',
        'type': 'str',
        'description': '动作空间类型'
    },
    'config.use_sgcnn': {
        'old_path': 'use_sgcnn',
        'type': 'bool',
        'description': '是否使用SGCNN'
    },
    'config.use_trajectory': {
        'old_path': 'use_traj',
        'type': 'bool',
        'description': '是否使用轨迹'
    },
    
    # =========================
    # 随机数生成器
    # =========================
    'rng': {
        'old_path': 'np_random',
        'type': 'Generator',
        'description': '随机数生成器'
    }
}

# 需要在旧环境中特殊处理的零散状态记录
SCATTERED_STATE_TRACKING = {
    'position_history': {
        'capture_points': ['step'],  # 在哪些方法中捕获
        'old_vars': ['x_last', 'y_last'],  # 旧环境中的变量名
        'description': '位置历史记录'
    },
    'direction_history': {
        'capture_points': ['step'],
        'old_vars': ['direction_last'],
        'description': '方向历史记录'
    },
    'action_history': {
        'capture_points': ['step'],
        'old_vars': ['last_action', 'last_velocity', 'last_angular_velocity'],
        'description': '动作历史记录'
    }
}

# 动力学更新相关的状态变化
DYNAMICS_STATE_CHANGES = {
    'position': {
        'new': 'dynamics_manager.update_agent_position',
        'old': 'agent.update',
        'validation': 'position_tolerance'
    },
    'maps': {
        'new': 'dynamics_manager.update_maps',
        'old': 'various_inline_updates',
        'validation': 'exact_match'
    },
    'metrics': {
        'new': 'metrics_manager.update',
        'old': 'inline_calculations',
        'validation': 'exact_match'
    }
}

# 容差配置
TOLERANCE_CONFIG = {
    'position_tolerance': 1e-6,
    'angle_tolerance': 1e-5,
    'reward_tolerance': 1e-4,
    'general_float_tolerance': 1e-7
}
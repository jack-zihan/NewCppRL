"""
算法常量定义模块

集中管理所有算法中使用的常量，避免魔法数字
"""


class PathConstants:
    """路径规划相关常量"""
    
    # JUMP算法常量
    JUMP_SAFETY_FACTOR = 4      # 跳跃安全系数
    JUMP_THRESHOLD = 4           # 默认跳跃阈值
    
    # 路径分解常量
    PATH_DECOMPOSE_STEP = 2      # 路径分解步长
    WAYPOINT_TOLERANCE = 1       # 路径点容差
    
    # 转弯相关
    DEFAULT_TURNING_RADIUS = 5.0 # 默认转弯半径
    MIN_TURNING_RADIUS = 2.0     # 最小转弯半径
    MAX_TURNING_RADIUS = 10.0    # 最大转弯半径
    
    # 安全边距
    SAFETY_MARGIN = 2            # 默认安全边距
    COLLISION_MARGIN = 1         # 碰撞检测边距
    
    # 覆盖相关
    COVERAGE_OVERLAP = 0.8       # 覆盖重叠系数
    MAX_COVERAGE_ITERATIONS = 10000  # 最大覆盖迭代次数
    
    # 性能相关
    DEFAULT_TIMEOUT = 300        # 默认超时时间（秒）
    MAX_PATH_LENGTH = 100000     # 最大路径长度
    
    
class AlgorithmDefaults:
    """算法默认参数"""
    
    # 初始方向设置
    INITIAL_TURN_DIRECTION = True  # 修正：旧版使用turn=True，不是False
    
    # SNAKE算法默认参数
    SNAKE_GREEDY_SEARCH = True
    SNAKE_FORWARD_ONLY = True
    
    # BCP算法默认参数
    BCP_MAX_ITERATIONS = 1000
    BCP_CONVERGENCE_THRESHOLD = 0.01
    
    # 共同参数
    DEFAULT_AGENT_WIDTH = 5
    DEFAULT_SIGHT_WIDTH = 24
    DEFAULT_SIGHT_LENGTH = 24
    
    
class PerformanceThresholds:
    """性能阈值定义"""
    
    # 覆盖率里程碑
    COVERAGE_MILESTONE_90 = 0.90
    COVERAGE_MILESTONE_95 = 0.95
    COVERAGE_MILESTONE_98 = 0.98
    
    # 效率指标
    MIN_EFFICIENCY_RATIO = 0.8   # 最小效率比
    TARGET_EFFICIENCY = 0.95      # 目标效率
    
    # 时间限制
    MAX_PLANNING_TIME = 0.1       # 单次规划最大时间（秒）
    MAX_TOTAL_TIME = 600          # 总运行时间限制（秒）
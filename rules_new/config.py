import os
import numpy as np


class Config:
    # 环境参数
    W = 600
    H = 600
    # ENV_NAME = "Pasture-v2"
    # RENDER_MODE = 'rgb_array'
    # ACTION_TYPE = "continuous"
    # WEED_COUNT = 50
    # GAUSSIAN_WEED = True
    RETURN_MAP = True
    NUM_OBSTACLE_MIN = 0
    NUM_OBSTACLE_MAX = 0


    # 小车参数
    CAR_WIDTH = 5
    SIGHT_WIDTH = 24
    SIGHT_LENGTH = 24
    # TURNING_RADIUS = 7

    # 路径设置
    DATA_DIR = 'path/to/data'
    MODEL_SAVE_PATH = ''
    LOG_DIR = 'rules/logs'
    
    SEED = 0

    # # JUMP SNAKE BCP R_SNAKE REACT # REACT不应该设置init_x y
    # TASK_TYPE = "SNAKE"


    SEED = 0
    DEBUG_MODE = True

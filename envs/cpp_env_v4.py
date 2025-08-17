from __future__ import annotations

import numpy as np
import cv2
import gymnasium as gym
from pathlib import Path
from envs.cpp_env_base_copy import CppEnvBase
from envs.utils import total_variation, MowerAgent


class CppEnvV4(CppEnvBase):
    """
    Pasture-v4: 无mist, 无weed奖励, 有frontier_hifs, 不使用多尺度
    """

    def __init__(self, *args, **kwargs):
        # 强制不使用多尺度
        kwargs['use_sgcnn'] = False
        kwargs['use_global_obs'] = False  # V4也不使用global obs
        super().__init__(*args, **kwargs)
        
        # 确保强制覆盖这些属性
        self.use_sgcnn = False
        self.use_global_obs = False
        
        self.map_frontier_hifs = None
        
        # 修正observation_space（V4返回4通道）
        obs_shape = (4, *self.state_downsize)
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=0., high=1., shape=obs_shape, dtype=np.float32
            ),
            'vector': gym.spaces.Box(
                low=-1., high=1., shape=(1,), dtype=np.float32
            ),
            'weed_ratio': gym.spaces.Box(
                low=0., high=1., shape=(), dtype=np.float32
            ),
        })

    def generate_frontier_maps(self, map_id: int):
        """加载frontier和frontier_hifs地图"""
        super().generate_frontier_maps(map_id)

        # 加载hifs方向场（.npy格式）
        hifs_dir = self.map_dir.parent / 'hifs'
        hifs_filename = f'orientation_map_{self.map_id+1}.npy'
        hifs_path = hifs_dir / hifs_filename
        self.map_frontier_hifs = np.load(str(hifs_path)).astype(np.float32)

    def get_maps_and_mask(self) -> tuple[np.ndarray, list[float]]:
        """生成观测，包含frontier_hifs方向场"""
        maps_list = [
            self.map_frontier,
            self.map_frontier_hifs,  # 保持原始弧度值（包括-1）
            self.map_obstacle,
            self.map_trajectory if self.use_traj else np.zeros_like(self.map_frontier),
        ]
        mask = [0., 0., 1., 0.]
        maps = np.stack(maps_list, axis=-1)
        return maps, mask

    def get_reward(self, steer_tp1, x_t, y_t, x_tp1, y_tp1) -> float:
        """完全重写奖励计算，移除weed奖励"""
        # 计算新的状态值
        weed_num_tp1 = self.map_weed.sum(dtype=np.int32)
        frontier_area_tp1 = self.map_frontier.sum(dtype=np.int32)
        frontier_tv_tp1 = total_variation(self.map_frontier.astype(np.int32))

        # 常数惩罚
        reward_const = -0.1

        # 转向惩罚
        reward_turn_gap = -0.5 * abs(steer_tp1 - self.steer_t) / self.w_range.max
        reward_turn_direction = -0.30 * (0. if (steer_tp1 * self.steer_t >= 0
                                                or (steer_tp1 == 0 and self.steer_t == 0))
                                         else 1.)
        reward_turn_self = 0.25 * (0.4 - abs(steer_tp1 / self.w_range.max) ** 0.5)
        reward_turn = 0.0 * (reward_turn_gap + reward_turn_direction + reward_turn_self)

        # Frontier覆盖奖励
        reward_frontier_coverage = (self.frontier_area_t - frontier_area_tp1) / (
                2 * MowerAgent.width * self.v_range.max)
        reward_frontier_tv = 0.5 * (self.frontier_tv_t - frontier_tv_tp1) / self.v_range.max
        reward_frontier = 0.125 * (reward_frontier_coverage + reward_frontier_tv)

        # 移除weed奖励！
        # reward_weed = 20.0 * (self.weed_num_t - weed_num_tp1)

        # Extra奖励
        reward_extra = self.get_extra_reward(steer_tp1, x_t, y_t, x_tp1, y_tp1)

        # 汇总（不包含weed）
        reward = reward_const + reward_frontier + reward_extra + reward_turn

        reward = np.where(np.abs(reward) < 1e-8, 0., reward)

        # 更新状态变量
        self.weed_num_t = weed_num_tp1  # 保留以避免其他地方报错
        self.frontier_area_t = frontier_area_tp1
        self.frontier_tv_t = frontier_tv_tp1
        self.steer_t = steer_tp1

        return reward

    def step(self, action):
        """重写step，修改结束条件"""
        # 执行父类的大部分逻辑
        x_t, y_t = self.agent.position_discrete
        acc, steer = self.get_action(action)
        self.agent.control(acc, steer)

        # 更新地图
        cv2.fillPoly(self.map_weed, [self.agent.convex_hull.round().astype(np.int32)], color=(0.,))
        cv2.ellipse(img=self.map_frontier,
                    center=self.agent.position_discrete,
                    axes=(self.vision_length, self.vision_length),
                    angle=self.agent.direction,
                    startAngle=-self.vision_angle / 2,
                    endAngle=self.vision_angle / 2,
                    color=(0.,),
                    thickness=-1,)
        cv2.ellipse(img=self.map_mist,
                    center=self.agent.position_discrete,
                    axes=(self.vision_length + 1, self.vision_length + 1),
                    angle=self.agent.direction,
                    startAngle=-self.vision_angle / 2,
                    endAngle=self.vision_angle / 2,
                    color=(1.,),
                    thickness=-1,)

        crashed = self.check_collision()
        x_tp1, y_tp1 = self.agent.position_discrete
        x_t = max(min(x_t, self.dimensions[0] - 1), 0)
        y_t = max(min(y_t, self.dimensions[1] - 1), 0)
        x_tp1 = max(min(x_tp1, self.dimensions[0] - 1), 0)
        y_tp1 = max(min(y_tp1, self.dimensions[1] - 1), 0)
        cv2.line(self.map_trajectory, pt1=(x_t, y_t), pt2=(x_tp1, y_tp1), color=(1.,))

        reward = self.get_reward(steer, x_t, y_t, x_tp1, y_tp1)
        if crashed:
            reward -= 399.

        self.t += 1
        time_out = self.t == 3000

        # 修改结束条件：frontier全部覆盖
        finish = self.map_frontier.sum() == 0  # 替代 self.weed_num_t == 0
        if finish:
            reward += 500

        done = crashed or finish
        obs = self.observation()

        # 添加覆盖率信息
        coverage_rate = 1 - (self.map_frontier.sum() / self.map_frontier_full.sum()) if self.map_frontier_full.sum() > 0 else 1.0
        info = {'coverage_rate': coverage_rate, 'crashed': crashed, 'finished': finish}

        return obs, reward, done, time_out, info

    def observation(self) -> dict[str, np.ndarray | float]:
        """修改observation，将weed_ratio改为coverage_rate"""
        # 调用父类方法获取基础观测
        obs_dict = super().observation()

        # 计算覆盖率
        coverage_rate = 1 - (self.map_frontier.sum() / self.map_frontier_full.sum()) if self.map_frontier_full.sum() > 0 else 1.0

        # 修改weed_ratio为coverage_rate（保持键名以兼容）
        obs_dict['weed_ratio'] = coverage_rate

        return obs_dict

    def reset(self, **kwargs):
        """重置时确保初始化frontier_hifs"""
        obs, info = super().reset(**kwargs)

        # 添加覆盖率信息
        coverage_rate = 1 - (self.map_frontier.sum() / self.map_frontier_full.sum()) if self.map_frontier_full.sum() > 0 else 0.0
        info['coverage_rate'] = coverage_rate

        return obs, info


class CppEnvV5(CppEnvV4):
    """
    Pasture-v5: 与v4相同，但使用多尺度观测和方向场奖励
    """

    def __init__(self, *args, **kwargs):
        # 方向场奖励参数
        self.direction_field_weight = kwargs.pop('direction_field_weight', 0.01)  # 每度差异的惩罚系数
        self.direction_current_weight = 0.6  # 当前位置权重
        self.direction_last_weight = 0.4     # 上一位置权重
        
        # 强制使用多尺度
        kwargs['use_sgcnn'] = True
        kwargs['use_global_obs'] = True  # V5使用global obs
        
        # 直接调用CppEnvBase的__init__，跳过V4的强制设置
        CppEnvBase.__init__(self, *args, **kwargs)
        
        # 确保强制覆盖这些属性
        self.use_sgcnn = True
        self.use_global_obs = True
        
        self.map_frontier_hifs = None
        
        # 修正observation_space（V5使用SGCNN返回20通道）
        # SGCNN: 4个尺度，每个尺度从4通道池化得到5个特征
        obs_shape = (20, 16, 16)  # 4 scales × 5 features per scale = 20 channels
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=0., high=1., shape=obs_shape, dtype=np.float32
            ),
            'vector': gym.spaces.Box(
                low=-1., high=1., shape=(1,), dtype=np.float32
            ),
            'weed_ratio': gym.spaces.Box(
                low=0., high=1., shape=(), dtype=np.float32
            ),
        })
    
    def _convert_agent_to_field_direction(self, agent_direction):
        """
        将小车朝向(0-360°)转换为方向场坐标系(0-π弧度)
        
        坐标系对应关系：
        小车系统            方向场系统
        0° (6点钟,向下)  →  π/2 (6点钟,向下)
        90° (3点钟,向右) →  π (3点钟,向右)
        180° (12点钟,向上) → π/2 (等价于向下)
        270° (9点钟,向左) → 0 (9点钟,向左)
        """
        # 转换公式
        field_degrees = (270 - agent_direction) % 360
        
        # 映射到[0, 180)范围（因为方向场是无向的）
        if field_degrees >= 180:
            field_degrees = field_degrees - 180
            
        # 转换为弧度
        return np.radians(field_degrees)
    
    def _compute_direction_difference_degrees(self, agent_dir, field_dir):
        """
        计算朝向差异，返回角度差（0-90度）
        
        Args:
            agent_dir: 小车朝向（0-360度）
            field_dir: 方向场值（弧度，-1表示无效）
        
        Returns:
            角度差异（0-90度），无效区域返回0
        """
        # -1表示无方向引导，不产生惩罚
        if field_dir < 0:
            return 0.0
        
        # 转换小车朝向到方向场坐标系
        agent_field_rad = self._convert_agent_to_field_direction(agent_dir)
        
        # 计算弧度差（考虑方向场的无向性）
        diff_rad = abs(agent_field_rad - field_dir)
        
        # 如果差异大于90度，取补角（因为方向场是无向的）
        if diff_rad > np.pi/2:
            diff_rad = np.pi - diff_rad
        
        # 转换为度
        diff_degrees = np.degrees(diff_rad)
        
        # 确保在[0, 90]范围内
        return np.clip(diff_degrees, 0, 90)
    
    def get_extra_reward(self, steer_tp1, x_t, y_t, x_tp1, y_tp1) -> float:
        """添加方向场奖励作为额外奖励（类似cpp_env_v2的APF奖励）"""
        reward_direction = 0.

        # 获取方向场值
        field_dir_current = self.map_frontier_hifs[y_tp1, x_tp1]
        field_dir_last = self.map_frontier_hifs[y_t, x_t]

        # 计算角度差异（度）
        # 如果是-1区域，返回0（不产生惩罚）
        angle_diff_current = self._compute_direction_difference_degrees(
            self.agent.direction, field_dir_current
        )
        angle_diff_last = self._compute_direction_difference_degrees(
            self.agent.direction, field_dir_last
        )

        # 加权平均角度差
        weighted_angle_diff = (
            self.direction_current_weight * angle_diff_current +
            self.direction_last_weight * angle_diff_last
        )

        # 方向场惩罚
        # 例如：weight=0.01时
        # 0度差异 → 0惩罚
        # 30度差异 → -0.3惩罚
        # 60度差异 → -0.6惩罚
        # 90度差异 → -0.9惩罚
        reward_direction = -self.direction_field_weight * weighted_angle_diff

        return reward_direction


if __name__ == "__main__":
    from gymnasium.wrappers import HumanRendering
    
    if_render = True
    episodes = 3
    
    # 测试 Pasture-v4 (无多尺度)
    # env = CppEnvV4(
    #     render_mode='rgb_array' if if_render else None,
    #     map_dir = "/home/lzh/NewCppRL/envs/maps/complex_field/farmland",
    #     # state_pixels=True,
    #     state_pixels=False,
    #     # use_sgcnn=False,  # V4会强制设为False
    #     # use_global_obs=False,
    #     # num_obstacles_range = [0, 0]
    # )
    
    # 测试 Pasture-v5 (有多尺度和方向场奖励) - 需要时取消注释
    env = CppEnvV5(
        render_mode='rgb_array' if if_render else None,
        # state_pixels=True,
        map_dir="/home/lzh/NewCppRL/envs/maps/complex_field/farmland",
        state_pixels=False,
        direction_field_weight=0.01,  # 可调节：每度差异的惩罚
        # use_sgcnn=True,  # V5会强制设为True
        # use_global_obs=True,
        # num_obstacles_range = [0, 0]
    )
    
    env = HumanRendering(env)  # 封装后，使得step和reset时展示渲染图像
    
    print(f"Testing {env.unwrapped.__class__.__name__}...")
    print(f"Observation shape will be: {env.observation_space['observation'].shape}")
    
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset(seed=120, options={
            'weed_dist': 'gaussian',
            # 'map_id': 80,
            "weed_num": 100
        })
        
        # 输出方向场信息（仅V5，在第一个episode的reset后）
        if episode == 0 and hasattr(env.unwrapped, 'direction_field_weight'):
            print(f"Direction field weight: {env.unwrapped.direction_field_weight}")
            if env.unwrapped.map_frontier_hifs is not None:
                valid_pixels = (env.unwrapped.map_frontier_hifs >= 0).sum()
                total_pixels = env.unwrapped.map_frontier_hifs.size
                if total_pixels > 0:
                    print(f"Direction field: {valid_pixels}/{total_pixels} valid pixels ({valid_pixels/total_pixels*100:.1f}%)")
        
        print(f"Initial coverage: {info.get('coverage_rate', 0):.2%}")
        
        env.action_space.seed(66)
        done = False
        step_count = 0
        
        while not done:
            action = env.action_space.sample()

            # action = 1 * 21 + 10  # 可以使用固定动作测试
            obs, reward, done, _, info = env.step(action)
            # obs, reward, done, _, info = env.step((0, 4))  # 连续动作空间
            
            step_count += 1
            print(f"  Step {step_count}: reward={reward:.4f}, coverage={info.get('coverage_rate', 0):.2%}")
            
            if if_render:
                env.render()
            
            if done:
                print(f"\nEpisode finished after {step_count} steps!")
                print(f"  Final coverage: {info.get('coverage_rate', 0):.2%}")
                if info.get('crashed'):
                    print("  Termination: Crashed")
                elif info.get('finished'):
                    print("  Termination: Frontier fully covered!")
                else:
                    print("  Termination: Unknown reason")
    
    env.close()
    print("\n✅ Test completed successfully!")
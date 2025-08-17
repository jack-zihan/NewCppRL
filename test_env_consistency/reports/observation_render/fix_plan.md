# 观测和渲染系统修复方案

## 修复方案概述

本方案提供具体的代码修复建议，确保新旧环境的观测生成和渲染系统功能一致。

## 1. APF参数统一修复

### 问题描述
不同地图类型使用了不同的APF参数，导致势场计算结果差异。

### 修复代码

```python
# 创建统一的APF配置文件：configs/apf_config.py
class APFConfig:
    """APF势场计算统一配置"""
    
    # 标准配置（与旧版保持一致）
    CONFIGS = {
        'frontier': {
            'max_step': 30,
            'eps': None,
            'pad': False,
            'gamma_auto': True  # 自动计算gamma
        },
        'obstacle': {
            'max_step': 10,
            'eps': None,
            'pad': True,  # 障碍物需要边界填充
            'gamma_auto': True
        },
        'weed': {
            'max_step': 40,
            'eps': 1e-2,  # 明确指定eps
            'pad': False,
            'gamma_auto': True
        },
        'trajectory': {
            'max_step': 4,
            'eps': None,
            'pad': False,
            'gamma_auto': True
        }
    }
    
    @staticmethod
    def get_config(map_type: str) -> dict:
        """获取指定类型的APF配置"""
        if map_type not in APFConfig.CONFIGS:
            raise ValueError(f"Unknown map type: {map_type}")
        return APFConfig.CONFIGS[map_type].copy()
    
    @staticmethod
    def calculate_gamma(max_step: int) -> float:
        """计算gamma值（与旧版一致）"""
        return (max_step - 1) / max_step
```

### 应用修复

```python
# 在envs/cpp_env_v2.py中修改get_maps_and_mask函数
from configs.apf_config import APFConfig

def get_maps_and_mask(self) -> tuple[np.ndarray, list[float]]:
    # ... 前面的代码保持不变 ...
    
    if self.use_apf:
        # 使用统一配置
        frontier_cfg = APFConfig.get_config('frontier')
        apf_frontier = self.get_discounted_apf(
            apf_frontier, 
            max_step=frontier_cfg['max_step'],
            eps=frontier_cfg['eps'],
            pad=frontier_cfg['pad']
        )
        
        obstacle_cfg = APFConfig.get_config('obstacle')
        apf_obstacle = self.get_discounted_apf(
            apf_obstacle,
            max_step=obstacle_cfg['max_step'],
            eps=obstacle_cfg['eps'],
            pad=obstacle_cfg['pad']
        )
        
        weed_cfg = APFConfig.get_config('weed')
        apf_weed = self.get_discounted_apf(
            apf_weed,
            max_step=weed_cfg['max_step'],
            eps=weed_cfg['eps'],
            pad=weed_cfg['pad']
        )
        
        trajectory_cfg = APFConfig.get_config('trajectory')
        apf_trajectory = self.get_discounted_apf(
            apf_trajectory,
            max_step=trajectory_cfg['max_step'],
            eps=trajectory_cfg['eps'],
            pad=trajectory_cfg['pad']
        )
    
    # ... 后续代码保持不变 ...
```

## 2. Mist语义修复

### 问题描述
Mist的0/1语义不一致，需要统一定义。

### 修复代码

```python
# 创建mist管理类：envs/mist_manager.py
class MistManager:
    """
    统一管理Mist（迷雾）语义
    
    约定：
    - mist = 0: 未探索区域（有迷雾）
    - mist = 1: 已探索区域（无迷雾）
    """
    
    @staticmethod
    def init_mist(dimensions: tuple) -> np.ndarray:
        """初始化mist地图（全部未探索）"""
        return np.zeros(dimensions, dtype=np.uint8)
    
    @staticmethod
    def update_mist(map_mist: np.ndarray, 
                   position: tuple,
                   vision_params: dict) -> np.ndarray:
        """
        更新mist地图（标记已探索区域）
        
        Args:
            map_mist: 当前mist地图
            position: 智能体位置
            vision_params: 视野参数
        """
        cv2.ellipse(
            img=map_mist,
            center=position,
            axes=(vision_params['length'] + 1, vision_params['length'] + 1),
            angle=vision_params['angle'],
            startAngle=-vision_params['fov'] / 2,
            endAngle=vision_params['fov'] / 2,
            color=(1,),  # 标记为已探索
            thickness=-1
        )
        return map_mist
    
    @staticmethod
    def get_explored_mask(map_mist: np.ndarray) -> np.ndarray:
        """获取已探索区域掩码"""
        return map_mist.astype(bool)
    
    @staticmethod
    def get_unexplored_mask(map_mist: np.ndarray) -> np.ndarray:
        """获取未探索区域掩码"""
        return np.logical_not(map_mist)
```

### 应用修复

```python
# 修改reset函数中的mist初始化
from envs.mist_manager import MistManager

def reset(self, ...):
    # ... 其他初始化代码 ...
    
    # 使用统一的mist初始化
    self.map_mist = MistManager.init_mist((self.dimensions[1], self.dimensions[0]))
    
    # 更新初始探索区域
    vision_params = {
        'length': self.vision_length,
        'angle': self.agent.direction,
        'fov': self.vision_angle
    }
    self.map_mist = MistManager.update_mist(
        self.map_mist,
        self.agent.position_discrete,
        vision_params
    )
    
    # ... 后续代码 ...

# 修改step函数中的mist更新
def step(self, action):
    # ... 前面的代码 ...
    
    # 使用统一的mist更新
    vision_params = {
        'length': self.vision_length,
        'angle': self.agent.direction,
        'fov': self.vision_angle
    }
    self.map_mist = MistManager.update_mist(
        self.map_mist,
        self.agent.position_discrete,
        vision_params
    )
    
    # ... 后续代码 ...

# 修改get_maps_and_mask中的mist使用
def get_maps_and_mask(self):
    # ... 前面的代码 ...
    
    # 使用统一的语义
    explored_mask = MistManager.get_explored_mask(self.map_mist)
    unexplored_mask = MistManager.get_unexplored_mask(self.map_mist)
    
    # 修正原来的逻辑
    apf_frontier = np.logical_and(
        total_variation_mat(self.map_frontier),
        explored_mask  # 已探索区域的边界
    )
    apf_obstacle = np.logical_and(
        total_variation_mat(self.map_obstacle),
        explored_mask  # 已探索区域的障碍物
    )
    
    # ... 后续代码 ...
    
    maps_list = [
        apf_frontier,
        unexplored_mask,  # 未探索区域
        apf_obstacle,
        apf_weed,
    ]
```

## 3. 噪声注入机制统一

### 问题描述
噪声注入的时机和方式需要统一。

### 修复代码

```python
# 创建噪声管理器：envs/noise_manager.py
class NoiseManager:
    """统一管理各种噪声的注入"""
    
    def __init__(self, np_random, config: dict):
        """
        初始化噪声管理器
        
        Args:
            np_random: numpy随机数生成器
            config: 噪声配置
        """
        self.np_random = np_random
        self.position_noise = config.get('position_noise', 0.0)
        self.direction_noise = config.get('direction_noise', 0.0)
        self.weed_noise_prob = config.get('weed_noise', 0.0)
    
    def apply_position_noise(self, position: tuple) -> tuple:
        """应用位置噪声"""
        if self.position_noise <= 0:
            return position
        
        x, y = position
        delta_x = np.clip(
            self.np_random.normal(0, self.position_noise),
            -self.position_noise,
            self.position_noise
        )
        delta_y = np.clip(
            self.np_random.normal(0, self.position_noise),
            -self.position_noise,
            self.position_noise
        )
        
        return (x + delta_x, y + delta_y)
    
    def apply_direction_noise(self, direction: float) -> float:
        """应用方向噪声"""
        if self.direction_noise <= 0:
            return direction
        
        delta_direction = np.clip(
            self.np_random.normal(0, self.direction_noise),
            -self.direction_noise,
            self.direction_noise
        )
        
        return (direction + delta_direction) % 360
    
    def should_use_noisy_weed(self) -> bool:
        """决定是否使用噪声杂草地图"""
        if self.weed_noise_prob <= 0:
            return False
        return self.np_random.uniform() < self.weed_noise_prob
    
    def apply_observation_noise(self, agent_pos: tuple, agent_dir: float) -> tuple:
        """
        统一应用观测噪声
        
        Returns:
            (noisy_position, noisy_direction)
        """
        noisy_pos = self.apply_position_noise(agent_pos)
        noisy_dir = self.apply_direction_noise(agent_dir)
        return noisy_pos, noisy_dir
```

### 应用修复

```python
# 修改get_rotated_obs_函数
def get_rotated_obs_(self, maps, mask: Sequence[float]):
    # 统一使用噪声管理器
    noisy_pos, noisy_dir = self.noise_manager.apply_observation_noise(
        (self.agent.x, self.agent.y),
        self.agent.direction
    )
    
    agent_x, agent_y = noisy_pos
    agent_direction = noisy_dir
    
    # ... 后续的旋转变换代码保持不变 ...
    
# 修改get_maps_and_mask函数
def get_maps_and_mask(self):
    # 使用噪声管理器决定是否使用噪声杂草
    if self.noise_manager.should_use_noisy_weed():
        map_weed_ = self.map_weed_noisy
    else:
        map_weed_ = self.map_weed
    
    # ... 后续代码 ...
```

## 4. 渲染颜色配置统一

### 问题描述
渲染颜色需要统一配置管理。

### 修复代码

```python
# 创建渲染配置：configs/render_config.py
class RenderConfig:
    """统一的渲染配置"""
    
    # 颜色定义（RGB）
    COLORS = {
        'background': (255, 255, 255),
        'field_frontier': (76, 187, 23),
        'covered_farmland_base': (112, 173, 7),
        'weed_undiscovered': (0, 0, 0),
        'weed_discovered': (255, 0, 0),
        'obstacle': (30, 75, 130),
        'obstacle_edge': (47, 82, 143),
        'agent': (255, 0, 0),
        'trajectory': (255, 38, 255),
        'vision_ellipse': (192, 192, 192),
        'covered_weed_base': (0, 0, 0),
    }
    
    # 透明度配置
    ALPHA = {
        'covered_farmland': 0.25,  # 基础色的权重
        'covered_weed': 0.9,       # 基础色的权重
        'vision_ellipse': 0.5,     # 半透明
    }
    
    # 渲染层次顺序（从底到顶）
    RENDER_ORDER = [
        'background',
        'field_frontier',
        'covered_farmland',
        'vision_ellipse',
        'weed',
        'obstacle',
        'agent',
        'trajectory',
        'covered_weed',
        'mist_overlay'  # 如果启用mist渲染
    ]
    
    @classmethod
    def blend_color(cls, base_color, overlay_color, alpha):
        """混合两种颜色"""
        base = np.array(base_color)
        overlay = np.array(overlay_color)
        blended = alpha * overlay + (1 - alpha) * base
        return tuple(blended.astype(np.uint8))
    
    @classmethod
    def get_covered_farmland_color(cls, base_color):
        """获取已覆盖农田的颜色"""
        return cls.blend_color(
            base_color,
            cls.COLORS['covered_farmland_base'],
            cls.ALPHA['covered_farmland']
        )
    
    @classmethod
    def get_covered_weed_color(cls, base_color):
        """获取已清除杂草的颜色"""
        return cls.blend_color(
            base_color,
            cls.COLORS['covered_weed_base'],
            cls.ALPHA['covered_weed']
        )
```

### 应用修复

```python
# 修改render_map函数
from configs.render_config import RenderConfig

def render_map(self) -> np.ndarray:
    # 初始化背景
    rendered_map = np.ones(
        (self.dimensions[1], self.dimensions[0], 3),
        dtype=np.float32
    ) * np.array(RenderConfig.COLORS['background'])
    
    # 1. 渲染农田
    rendered_map = np.where(
        np.expand_dims(self.map_frontier, axis=-1),
        np.array(RenderConfig.COLORS['field_frontier']),
        rendered_map
    )
    
    # 2. 渲染已覆盖农田
    if self.render_covered_farmland:
        covered_mask = np.logical_and(
            self.map_frontier_full,
            np.logical_not(self.map_frontier)
        )
        covered_color = RenderConfig.get_covered_farmland_color(
            RenderConfig.COLORS['background']
        )
        rendered_map = np.where(
            np.expand_dims(covered_mask, axis=-1),
            np.array(covered_color),
            rendered_map
        )
    
    # 3. 渲染视野椭圆
    cv2.ellipse(
        img=rendered_map,
        center=self.agent.position_discrete,
        axes=(self.vision_length, self.vision_length),
        angle=self.agent.direction,
        startAngle=-self.vision_angle / 2,
        endAngle=self.vision_angle / 2,
        color=RenderConfig.COLORS['vision_ellipse'],
        thickness=-1
    )
    
    # 4. 渲染杂草
    weed_undiscovered = get_map_pasture_larger(
        np.logical_and(self.map_weed, self.map_frontier)
    )
    weed_discovered = get_map_pasture_larger(
        np.logical_and(self.map_weed, np.logical_not(self.map_frontier))
    )
    
    rendered_map = np.where(
        np.expand_dims(weed_undiscovered, axis=-1),
        np.array(RenderConfig.COLORS['weed_undiscovered']),
        rendered_map
    )
    
    rendered_map = np.where(
        np.expand_dims(weed_discovered, axis=-1),
        np.array(RenderConfig.COLORS['weed_discovered']),
        rendered_map
    )
    
    # 5. 渲染障碍物
    rendered_map = np.where(
        np.expand_dims(self.map_obstacle, axis=-1),
        np.array(RenderConfig.COLORS['obstacle']),
        rendered_map
    )
    
    # 6. 渲染障碍物边缘
    mask_tv = total_variation_mat(self.map_obstacle)
    rendered_map = np.where(
        np.expand_dims(mask_tv, axis=-1),
        np.array(RenderConfig.COLORS['obstacle_edge']),
        rendered_map
    )
    
    # 7. 渲染智能体
    cv2.fillPoly(
        rendered_map,
        [self.agent.convex_hull.round().astype(np.int32)],
        color=RenderConfig.COLORS['agent']
    )
    
    # 8. 渲染轨迹
    rendered_map = np.where(
        np.expand_dims(self.map_trajectory, axis=-1) != 0,
        np.array(RenderConfig.COLORS['trajectory']),
        rendered_map
    )
    
    # 9. 渲染已清除杂草
    if self.render_covered_weed:
        weed_covered = get_map_pasture_larger(
            np.logical_and(self.map_weed_ori, np.logical_not(self.map_weed))
        )
        covered_color = RenderConfig.get_covered_weed_color(rendered_map)
        rendered_map = np.where(
            np.expand_dims(weed_covered, axis=-1),
            covered_color,
            rendered_map
        )
    
    return rendered_map.astype(np.uint8)
```

## 5. 测试验证

### 创建一致性测试

```python
# tests/test_consistency.py
import numpy as np
import pytest
from envs.cpp_env_v2 import CppEnv as OldEnv
from rules_new.environment import Environment as NewEnv  # 假设的新环境

class TestConsistency:
    """环境一致性测试"""
    
    def test_apf_consistency(self):
        """测试APF计算一致性"""
        test_map = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        
        # 测试各种配置
        configs = ['frontier', 'obstacle', 'weed', 'trajectory']
        for config_name in configs:
            old_result = OldEnv.compute_apf(test_map, config_name)
            new_result = NewEnv.compute_apf(test_map, config_name)
            
            np.testing.assert_allclose(
                old_result, new_result,
                rtol=1e-5, atol=1e-8,
                err_msg=f"APF mismatch for {config_name}"
            )
    
    def test_mist_semantics(self):
        """测试Mist语义一致性"""
        old_env = OldEnv()
        new_env = NewEnv()
        
        # 重置后检查
        old_env.reset(seed=42)
        new_env.reset(seed=42)
        
        # 初始状态应该大部分是未探索（0）
        assert np.mean(old_env.map_mist) < 0.1
        assert np.mean(new_env.map_mist) < 0.1
        
        # 执行动作后，探索区域应该增加
        for _ in range(10):
            action = 0
            old_env.step(action)
            new_env.step(action)
        
        # 探索区域应该增加
        assert np.sum(old_env.map_mist) > 100
        assert np.sum(new_env.map_mist) > 100
    
    def test_rendering_colors(self):
        """测试渲染颜色一致性"""
        old_env = OldEnv()
        new_env = NewEnv()
        
        old_env.reset(seed=42)
        new_env.reset(seed=42)
        
        old_render = old_env.render_map()
        new_render = new_env.render_map()
        
        # 提取主要颜色
        old_colors = np.unique(old_render.reshape(-1, 3), axis=0)
        new_colors = np.unique(new_render.reshape(-1, 3), axis=0)
        
        # 检查关键颜色是否存在
        key_colors = [
            (76, 187, 23),   # field_frontier
            (30, 75, 130),   # obstacle
            (255, 0, 0),     # agent/weed
        ]
        
        for color in key_colors:
            assert any(np.all(old_colors == color, axis=1))
            assert any(np.all(new_colors == color, axis=1))
    
    def test_noise_injection(self):
        """测试噪声注入一致性"""
        # 创建带噪声的环境
        old_env = OldEnv(noise_position=5.0, noise_direction=10.0)
        new_env = NewEnv(noise_position=5.0, noise_direction=10.0)
        
        # 使用相同种子
        old_env.reset(seed=42)
        new_env.reset(seed=42)
        
        # 获取多次观测
        old_obs_list = []
        new_obs_list = []
        
        for i in range(5):
            np.random.seed(42 + i)
            old_obs = old_env.observation()
            
            np.random.seed(42 + i)
            new_obs = new_env.observation()
            
            old_obs_list.append(old_obs['observation'])
            new_obs_list.append(new_obs['observation'])
        
        # 检查噪声效果的统计特性
        old_std = np.std([np.mean(obs) for obs in old_obs_list])
        new_std = np.std([np.mean(obs) for obs in new_obs_list])
        
        # 标准差应该相近（说明噪声效果类似）
        assert abs(old_std - new_std) / max(old_std, new_std) < 0.2
```

## 6. 实施计划

### 第一阶段：基础修复（2小时）
1. 创建配置文件（APFConfig, RenderConfig）
2. 实现MistManager
3. 实现NoiseManager

### 第二阶段：代码集成（3小时）
1. 修改get_maps_and_mask函数
2. 修改render_map函数
3. 修改噪声注入逻辑

### 第三阶段：测试验证（2小时）
1. 运行一致性测试
2. 对比输出结果
3. 调试不一致的地方

### 第四阶段：文档更新（1小时）
1. 更新环境文档
2. 添加迁移指南
3. 记录关键参数

## 7. 注意事项

1. **保持向后兼容**：修改时确保不破坏现有功能
2. **充分测试**：每个修改都要有对应的测试用例
3. **性能监控**：确保修复不会降低性能
4. **版本控制**：使用git分支管理修改
5. **代码审查**：修改完成后进行代码审查

## 8. 验证清单

- [ ] APF参数完全一致
- [ ] Mist语义统一
- [ ] 噪声注入机制一致
- [ ] 渲染颜色匹配
- [ ] 坐标变换精度相同
- [ ] 性能没有下降
- [ ] 所有测试通过
- [ ] 文档已更新

---
**文档版本**: 1.0
**最后更新**: 2025-08-14
**负责人**: Bug侦探
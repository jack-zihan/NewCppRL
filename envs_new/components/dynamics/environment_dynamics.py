"""
割草机器人环境动力学系统。

核心设计：
1. 组件化更新器模式：每个Updater负责特定的状态更新逻辑
2. 依赖解析与排序：自动处理组件间依赖关系，确保正确的更新顺序
3. 统一状态管理：通过StateVariable系统跟踪历史状态变化
4. 碰撞回滚机制：检测到碰撞时自动回滚位置

更新器执行顺序（由依赖关系决定）：
1. 基础组件：agent、frontier、weed、mist（无依赖）
2. 依赖组件：trajectory（依赖agent）、flags（依赖weed）
3. 独立组件：step（无依赖，但通常最后执行）
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Union, List, Any

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.entity.agent import Agent
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.dynamics.collision_detector import CollisionDetector
from envs_new.components.dynamics.action_processor import ActionProcessor
from envs_new.utils.dependency_sorter import sort_components_by_dependencies
from envs_new.utils.math_utils import total_variation


class FrontierUpdater:
    """
    前沿区域探索状态更新器 - 跟踪未探索区域的变化。
    
    更新逻辑：使用椭圆扇形表示机器人视野，将探索过的区域标记为已知。
    计算前沿变化度(TV)用于评估前沿复杂度。
    """

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return []  # 无依赖，基础组件

    def setup_state(self, env_state: EnvironmentState, history_length: int = 2) -> None:
        """初始化状态变量"""
        env_state.add_state_info('frontier_area', history_length, 0)
        env_state.add_state_info('frontier_variation', history_length, 0)

    def update(self, state: Dict[str, Any]) -> None:
        """更新前沿地图并记录状态变化"""
        maps_dict = state['maps_dict']
        agent = state['agent']
        env_state = state['env_state']
        
        # 使用椭圆扇形模拟机器人视野，将探索过的区域设为0（已知）
        if 'field_frontier' in maps_dict:
            cv2.ellipse(
                img=maps_dict['field_frontier'],
                center=agent.position_discrete,  # 椭圆中心：机器人位置
                axes=(int(agent.vision_length), int(agent.vision_length)),  # 长短轴：视野范围
                angle=float(agent.direction),  # 旋转角度：机器人朝向
                startAngle=float(-agent.vision_angle / 2),  # 扇形起始角
                endAngle=float(agent.vision_angle / 2),  # 扇形结束角
                color=(0,),  # 填充为0（已探索）
                thickness=-1  # -1表示实心填充
            )
            
            # 记录状态变化
            new_frontier_area = int(maps_dict['field_frontier'].sum())
            new_frontier_variation = total_variation(maps_dict['field_frontier'].astype(np.int32))
            env_state.update_state(
                frontier_area=new_frontier_area,
                frontier_variation=new_frontier_variation
            )


class WeedUpdater:
    """
    杂草清除状态更新器 - 处理机器人清除杂草的逻辑。
    
    更新逻辑：将机器人凸包覆盖区域内的杂草标记为已清除。
    """

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return []  # 无依赖，基础组件

    def setup_state(self, env_state: EnvironmentState, history_length: int = 2) -> None:
        """初始化状态变量"""
        env_state.add_state_info('weed_count', history_length, env_state.total_weed_count)

    def update(self, state: Dict[str, Any]) -> None:
        """更新杂草地图并记录状态变化"""
        maps_dict = state['maps_dict']
        agent = state['agent']
        env_state = state['env_state']
        
        # 使用凸包填充算法清除机器人覆盖区域内的杂草
        if 'weed' in maps_dict:
            convex_hull = agent.convex_hull.round().astype(np.int32)
            cv2.fillPoly(maps_dict['weed'], [convex_hull], color=(0,))  # 0表示清除
            
            # 记录状态变化
            new_weed_count = int(maps_dict['weed'].sum())
            env_state.update_state(weed_count=new_weed_count)


class AgentUpdater:
    """智能体状态更新器"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return []  # 无依赖

    def setup_state(self, env_state: EnvironmentState, history_length: int = 2) -> None:
        """初始化agent相关状态变量"""
        env_state.add_state_info('agent_position', history_length, (0.0, 0.0))
        env_state.add_state_info('agent_speed', history_length, 0.0)
        env_state.add_state_info('agent_steer', history_length, 0.0)
        env_state.add_state_info('trajectory_length', history_length, 0.0)

    def update(self, state: Dict[str, Any]) -> None:
        """更新agent状态信息"""
        agent = state['agent']
        env_state = state['env_state']
        
        # 记录agent状态变化
        env_state.update_state(
            agent_position=agent.position,
            agent_speed=agent.last_speed,
            agent_steer=agent.last_steer
        )
        
        # 更新轨迹长度
        self._update_trajectory_length(env_state)
        
        # 设置当前转向供奖励计算使用
        env_state.agent_steer = agent.last_steer

    def _update_trajectory_length(self, env_state: EnvironmentState) -> None:
        """计算累积轨迹长度"""
        if len(env_state.get_info('agent_position')) >= 2:
            agent_pos_info = env_state.get_info('agent_position')
            distance = np.linalg.norm(np.array(agent_pos_info.current) - np.array(agent_pos_info.last))
            env_state.update_state(trajectory_length=env_state.trajectory_length + distance)


class MistUpdater:
    """雾效地图更新器"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return []  # 无依赖

    def setup_state(self, env_state: EnvironmentState, history_length: int = 2) -> None:
        """无需初始化状态变量"""
        pass

    def update(self, state: Dict[str, Any]) -> None:
        """更新雾效地图可见性"""
        maps_dict = state['maps_dict']
        agent = state['agent']
        
        if 'mist' in maps_dict:
            # 在agent当前视野范围内清除雾效
            cv2.ellipse(
                img=maps_dict['mist'],
                center=agent.position_discrete,
                axes=(int(agent.vision_length + 1), int(agent.vision_length + 1)),
                angle=float(agent.direction),
                startAngle=float(-agent.vision_angle / 2),
                endAngle=float(agent.vision_angle / 2),
                color=(0,),
                thickness=-1
            )


class TrajectoryUpdater:
    """轨迹记录更新器"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['agent']  # 依赖agent位置历史

    def setup_state(self, env_state: EnvironmentState, history_length: int = 2) -> None:
        """无需初始化状态变量"""
        pass

    def update(self, state: Dict[str, Any]) -> None:
        """记录agent轨迹到地图"""
        maps_dict = state['maps_dict']
        agent = state['agent']
        env_state = state['env_state']
        
        if 'trajectory' in maps_dict and len(env_state.get_info('agent_position')) >= 2:
            agent_pos_info = env_state.get_info('agent_position')
            cv2.line(
                maps_dict['trajectory'], 
                tuple(map(int, agent_pos_info.last)), 
                agent.position_discrete, 
                color=(1.,)
            )


class FlagsUpdater:
    """环境标志更新器"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['weed']  # 依赖杂草计数来判断任务完成

    def setup_state(self, env_state: EnvironmentState, history_length: int = 2) -> None:
        """初始化标志状态"""
        env_state.add_state_info('crashed', history_length, False)
        env_state.add_state_info('finished', history_length, False)
        env_state.add_state_info('timeout', history_length, False)

    def update(self, state: Dict[str, Any]) -> None:
        """更新环境标志状态"""
        maps_dict = state['maps_dict']
        env_state = state['env_state']
        context = state.get('context', {})
        
        # 检查任务完成条件
        finished = env_state.finished or self._is_task_finished(maps_dict)
        
        env_state.update_state(
            crashed=context.get('crashed', False),
            finished=finished
        )

    def _is_task_finished(self, maps_dict: Dict[str, np.ndarray]) -> bool:
        """判断任务是否完成"""
        return int(maps_dict['weed'].sum()) == 0 if 'weed' in maps_dict else False


class StepUpdater:
    """步数计数更新器"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return []  # 无依赖

    def setup_state(self, env_state: EnvironmentState, history_length: int = 2) -> None:
        """初始化步数状态"""
        env_state.add_state_info('current_step', history_length, -1)
        env_state.add_state_info('timeout', history_length, False)

    def update(self, state: Dict[str, Any]) -> None:
        """更新步数计数"""
        env_state = state['env_state']
        
        # 增加步数
        current_step = env_state.current_step + 1
        env_state.current_step = current_step
        
        env_state.update_state(
            current_step=current_step,
            timeout=current_step >= env_state.max_steps
        )


class EnvironmentDynamics:
    """
    环境动力学管理器 - 协调所有更新器组件的核心类。
    
    设计模式：
    1. 组合模式：将复杂的更新逻辑分解为独立组件
    2. 依赖注入：动态注入所需的更新器组件
    3. 模板方法：step()定义更新流程，具体逻辑由组件实现
    """

    # 所有可用的updater组件
    AVAILABLE_UPDATERS = {
        'frontier': FrontierUpdater,
        'weed': WeedUpdater,
        'agent': AgentUpdater,
        'mist': MistUpdater,
        'trajectory': TrajectoryUpdater,
        'flags': FlagsUpdater,
        'step': StepUpdater
    }

    def __init__(self, config: EnvironmentConfig, action_processor: ActionProcessor,
                 enabled_updaters: Union[List[str], Dict[str, bool], None] = None,
                 history_length: int = 2):
        """
        初始化环境动力学系统
        
        Args:
            config: 环境配置
            action_processor: 动作处理器
            enabled_updaters: 启用的updater组件
            history_length: 状态历史长度
        """
        self.config = config
        self.action_processor = action_processor
        self.collision_detector = CollisionDetector()
        self.history_length = history_length

        # 确定启用的updaters
        if enabled_updaters is None:
            # 默认启用所有
            components_to_create = list(self.AVAILABLE_UPDATERS.keys())
        elif isinstance(enabled_updaters, list):
            components_to_create = enabled_updaters
        elif isinstance(enabled_updaters, dict):
            components_to_create = [name for name, enabled in enabled_updaters.items() if enabled]
        else:
            raise ValueError(f"enabled_updaters must be None, list, or dict, got {type(enabled_updaters)}")

        # 验证组件
        invalid_components = set(components_to_create) - set(self.AVAILABLE_UPDATERS.keys())
        if invalid_components:
            raise ValueError(f"Unknown updater components: {invalid_components}")

        # 使用拓扑排序算法处理组件依赖，确保依赖项先于依赖者执行
        sorted_components = sort_components_by_dependencies(
            self.AVAILABLE_UPDATERS, components_to_create
        )
        
        self._updaters = {}
        for name in sorted_components:
            self._updaters[name] = self.AVAILABLE_UPDATERS[name]()

    def step(self, agent: Agent, maps_dict: Dict[str, np.ndarray],
             env_state: EnvironmentState, action: Union[int, Tuple],
             action_type: str) -> Tuple[Agent, Dict[str, np.ndarray], EnvironmentState]:
        """
        执行一步环境更新 - 核心更新流程。
        
        更新顺序：
        1. 动作解析与机器人控制
        2. 碰撞检测与位置回滚
        3. 按依赖顺序执行所有updater
        """
        # 解析动作并控制agent
        linear_velocity, angular_velocity = self.action_processor.parse_action(action, action_type)
        agent.control(linear_velocity, angular_velocity)
        
        # 碰撞检测与处理：若发生碰撞，回滚到上一步的安全位置
        crashed = self.collision_detector.check_collision(agent, maps_dict)
        if crashed:
            agent.rollback_position()

        # 构建共享状态字典，传递给所有updater
        state = {
            'maps_dict': maps_dict,  # 地图数据
            'agent': agent,          # 机器人实体
            'env_state': env_state,  # 环境状态
            'context': {'crashed': crashed}  # 上下文信息
        }
        
        # 按依赖顺序执行所有updater
        for name in self._updaters:
            self._updaters[name].update(state)
        
        return agent, maps_dict, env_state

    def reset(self, agent: Agent, maps_dict: Dict[str, np.ndarray], env_state: EnvironmentState) -> None:
        """重置环境动力学"""
        # 初始化所有updater的状态
        for name in self._updaters:
            self._updaters[name].setup_state(env_state, self.history_length)
        
        # 执行初始更新
        state = {
            'maps_dict': maps_dict,
            'agent': agent,
            'env_state': env_state,
            'context': {'crashed': False}
        }
        
        for name in self._updaters:
            self._updaters[name].update(state)

    def add_updater(self, name: str, updater: Any) -> None:
        """添加自定义updater"""
        self._updaters[name] = updater

    def remove_updater(self, name: str) -> None:
        """移除updater"""
        if name in self._updaters:
            del self._updaters[name]

    def get_updaters(self) -> Dict[str, Any]:
        """获取所有updater（拷贝）"""
        return self._updaters.copy()

    def get_collision_info(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """获取碰撞信息"""
        return self.collision_detector.get_collision_details(agent, maps_dict)

    def is_valid_action(self, action: Union[int, Tuple], action_type: str) -> bool:
        """验证动作有效性"""
        try:
            self.action_processor.parse_action(action, action_type)
            return True
        except (ValueError, TypeError):
            return False
from __future__ import annotations
import math
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from envs.components.utils import MowerAgent, total_variation


class SceneDynamic:
    """
    场景动力学类，用于处理场景的动态更新和状态信息统计。
    """

    def __init__(self):
        # 注册各地图键的更新逻辑，便于扩展更多地图类型。
        self._map_update_handlers = {
            "weed": self._cut_weeds,
            "field_frontier": self._update_frontier_map,
            "mist": self._update_mist_map,
            "trajectory": self._update_trajectory_map,
        }

    def reset(self, maps_dict: Dict[str, np.ndarray], agent: MowerAgent, state_info: Dict[str, Any]):
        """
        在场景重置时初始化状态信息，包括前沿面积、边缘差分、杂草数量和其他标记。
        """
        frontier_area = self._get_frontier_area(maps_dict)
        frontier_variation = self._get_frontier_variation(maps_dict)
        weed_count = self._get_weed_count(maps_dict)

        state_info.update({
            "frontier_area": frontier_area, "prev_frontier_area": frontier_area,
            "frontier_variation": frontier_variation, "prev_frontier_variation": frontier_variation,
            "weed_count": weed_count, "prev_weed_count": weed_count,
            "prev_discrete_pos": agent.position_discrete,
            "crashed": False,
            "finished": False,
            "prev_steer": 0.0, "current_steer": 0.0,
        })
        self._apply_map_update(maps_dict, agent, state_info)

    def dynamic(self,
                maps_dict: Dict[str, np.ndarray], agent: MowerAgent, state_info: Dict[str, Any],
                linear_velocity: float, angular_velocity: float
                ) -> Tuple[
        Dict[str, np.ndarray], MowerAgent, Dict[str, Any]]:
        """
        更新场景状态信息，包括机器人控制、地图更新、碰撞检测和状态统计。

        :param maps_dict: 当前场景地图的字典。
        :param agent: 场景中的机器人对象。
        :param state_info: 状态信息字典。
        :param linear_velocity: 机器人的线速度。
        :param angular_velocity: 机器人的角速度。
        :return: 更新后的地图字典、机器人对象、状态信息、是否碰撞和是否完成标记。
        """
        state_info["prev_steer"] = state_info["current_steer"]
        state_info["prev_discrete_pos"] = agent.position_discrete

        agent.control(linear_velocity, angular_velocity)
        self._apply_map_update(maps_dict, agent, state_info)

        state_info.update({
            "current_steer": agent.last_steer,
            "discrete_pos": agent.position_discrete,
            "crashed": self._check_collision(maps_dict, agent),
            "finished": state_info["weed_count"] == 0,
        })

        return maps_dict, agent, state_info

    # -------------------- 私有方法 --------------------

    def _apply_map_update(self, maps_dict: Dict[str, np.ndarray], agent: MowerAgent, state_info: Dict[str, Any]):
        for map_key in maps_dict:
            if map_key in self._map_update_handlers:
                self._map_update_handlers[map_key](maps_dict, agent, state_info)

        state_info["prev_frontier_area"] = state_info["frontier_area"]
        state_info["prev_frontier_variation"] = state_info["frontier_variation"]
        state_info["prev_weed_count"] = state_info["weed_count"]

        state_info["frontier_area"] = self._get_frontier_area(maps_dict)
        state_info["frontier_variation"] = self._get_frontier_variation(maps_dict)
        state_info["weed_count"] = self._get_weed_count(maps_dict)

    @staticmethod
    def _cut_weeds(maps_dict: Dict[str, np.ndarray], agent: MowerAgent,
                   state_info: Optional[Dict[str, Any]] = None):
        """
        将机器人占据区域内的杂草移除。
        """
        cv2.fillPoly(maps_dict["weed"], [agent.convex_hull.round().astype(np.int32)], color=(0.,))


    @staticmethod
    def _update_frontier_map(maps_dict: Dict[str, np.ndarray], agent: MowerAgent,
                             state_info: Optional[Dict[str, Any]] = None):
        """
        更新前沿地图，移除机器人视野内的区域。
        """
        cv2.ellipse(img=maps_dict["field_frontier"],
                    center=agent.position_discrete,
                    axes=(int(agent.vision_length), int(agent.vision_length)),
                    angle=agent.direction,
                    startAngle=-int(agent.vision_angle / 2),
                    endAngle=int(agent.vision_angle / 2),
                    color=(0.,),
                    thickness=-1,
                    )
    # 全部换成1->0的过程，要不覆盖任务会增加agent的理解难度 TODO：所以之后的逻辑要看着修改mist相关
    @staticmethod
    def _update_mist_map(maps_dict: Dict[str, np.ndarray], agent: MowerAgent,
                         state_info: Optional[Dict[str, Any]] = None):
        """
        更新迷雾地图，机器人视野范围内的区域设置为可见。
        """
        cv2.ellipse(
            img=maps_dict["mist"],
            center=agent.position_discrete,
            axes=(int(agent.vision_length + 1), int(agent.vision_length + 1)),
            angle=agent.direction,
            startAngle=-int(agent.vision_angle / 2),
            endAngle=int(agent.vision_angle / 2),
            color=(0.,),
            thickness=-1
        )

    @staticmethod
    def _update_trajectory_map(
            maps_dict: Dict[str, np.ndarray], agent: MowerAgent, state_info: Optional[Dict[str, Any]] = None
    ):
        """
        在轨迹地图中记录机器人的运动路径。
        """
        cv2.line(maps_dict["trajectory"], state_info.get("prev_discrete_pos"), agent.position_discrete, (0.,), 1)

    @staticmethod
    def _check_collision(maps_dict: Dict[str, np.ndarray], agent: MowerAgent) -> bool:
        """
        检测机器人是否与障碍物碰撞或越界。
        """
        assert "obstacle" in maps_dict  # "Obstacle map is required for collision detection."
        obstacle_map = maps_dict["obstacle"]
        dimensions = obstacle_map.shape

        map_agent = np.zeros_like(obstacle_map, dtype=np.uint8)
        cv2.fillPoly(map_agent, [agent.convex_hull.round().astype(np.int32)], 1)

        # 检测是否越界
        crashed_bounds = not (
                (0 <= agent.convex_hull[:, 0]).all() and (agent.convex_hull[:, 0] < dimensions[0]).all() and
                (0 <= agent.convex_hull[:, 1]).all() and (agent.convex_hull[:, 1] < dimensions[1]).all()
        )

        crashed_obstacles = np.any(map_agent & obstacle_map)
        return crashed_bounds or crashed_obstacles

    @staticmethod
    def _get_frontier_area(maps_dict: Dict[str, np.ndarray]) -> int:
        return int(maps_dict["field_frontier"].sum())

    @staticmethod
    def _get_frontier_variation(maps_dict: Dict[str, np.ndarray]) -> int:
        return int(total_variation(maps_dict["field_frontier"].astype(np.int32)))

    @staticmethod
    def _get_weed_count(maps_dict: Dict[str, np.ndarray]) -> int:
        return int(maps_dict["weed"].sum())

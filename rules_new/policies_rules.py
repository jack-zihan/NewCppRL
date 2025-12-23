"""
Rule-based controllers adapted to envs_new.

Each controller exposes:
    reset(env)
    act(obs, env) -> (linear_velocity, angular_velocity)

All metric/stat handling is done by the caller; controllers只负责出动作。
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
import cv2
from shapely.geometry import LineString
import dubins


# ---------- 通用工具 ----------

def _normalize_deg(angle: float) -> float:
    """Wrap angle to [-180, 180)."""
    angle = (angle + 180.0) % 360.0 - 180.0
    return angle


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    """Return coords (x,y) where mask > 0."""
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1)


def _coverage_mask(env) -> np.ndarray | None:
    base = getattr(env, "unwrapped", env)
    maps = base.maps_dict
    # 优先使用 field；weed 版本在未来单独扩展
    if "field" in maps:
        return maps["field"]
    if "weed" in maps:
        return maps["weed"]
    return None


def _compute_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    coords = _mask_coords(mask)
    if coords.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    minx, miny = coords.min(axis=0)
    maxx, maxy = coords.max(axis=0)
    return float(minx), float(miny), float(maxx), float(maxy)


def _fov_strip_spacing(config) -> float:
    """根据视野参数计算 BCP 扫线间距（FOV 版本）。

    论文中的“搜索版 BCP”使用 Sw（FOV 最远处宽度）作为相邻 pass 的间距。
    这里按照三角形视野近似：Sw = 2 * Sd * sin(angle/2)。
    """
    length = float(config.agent_vision_length)
    angle_rad = math.radians(float(config.agent_vision_angle))
    spacing = 2.0 * length * math.sin(angle_rad / 2.0)
    # 理论上一定 >0，做一个极小保护避免数值问题
    return max(spacing, 1.0)


def _env_deg_to_dubins(theta_deg: float) -> float:
    """将env角度（度）转换为Dubins弧度。

    注：env和Dubins都使用逆时针为正的坐标系，只需度数→弧度转换。
    """
    return math.radians(float(theta_deg))


def _dubins_to_env_deg(theta_dubins_rad: float) -> float:
    """将Dubins弧度转换为env角度（度）。

    注：env和Dubins都使用逆时针为正的坐标系，只需弧度→度数转换。
    """
    return math.degrees(float(theta_dubins_rad))


def _plan_dubins_path(env, start_pose: Tuple[float, float, float],
                      end_pose: Tuple[float, float, float],
                      step_size: float = 1.0) -> List[Tuple[float, float, float]]:
    """在env动力学约束下，从start到end规划一条Dubins路径，返回(x, y, theta_env_deg)三元组。

    start_pose / end_pose: (x, y, theta_env_deg)，其中theta_env_deg与agent.direction一致：
        0° 沿 +x，顺时针为正。

    返回: List[(x, y, theta_env_deg)]，theta是曲线在该点的切线方向（env坐标系）。
    """
    base = getattr(env, "unwrapped", env)
    v_max = float(base.config.v_max)
    w_max_deg = float(base.config.w_max)
    w_max = abs(w_max_deg) * math.pi / 180.0
    if w_max <= 0.0:
        raise RuntimeError(f"Invalid w_max in config: {w_max_deg}")
    turning_radius = v_max / w_max

    sx, sy, sth = start_pose
    ex, ey, eth = end_pose
    q0 = (float(sx), float(sy), _env_deg_to_dubins(sth))
    q1 = (float(ex), float(ey), _env_deg_to_dubins(eth))

    path = dubins.shortest_path(q0, q1, turning_radius)
    configs, _ = path.sample_many(step_size)
    # 去掉首尾端点，避免与直线部分重复
    inner = configs[1:-1] if len(configs) > 2 else configs[1:-1]
    # 保留theta并转换为env坐标系
    return [(float(x), float(y), _dubins_to_env_deg(theta)) for (x, y, theta) in inner]


class BaseRuleController:
    def __init__(self, name: str):
        self.name = name
        self.path: List[Tuple[float, float]] = []
        self.path_theta: List[float] = []  # 每个路径点的切线方向（env坐标系度数）
        self.idx: int = 0
        self.reach_eps = 1.0

    def reset(self, env) -> None:
        self.path = []
        self.path_theta = []
        self.idx = 0

    def _heading_deg(self, env) -> float:
        base = getattr(env, "unwrapped", env)
        dir_state = base.env_state.get_info("agent_direction")
        if dir_state and dir_state.current is not None:
            return float(dir_state.current)
        return float(base.agent.direction)

    def _pose(self, env) -> Tuple[float, float]:
        base = getattr(env, "unwrapped", env)
        pos_state = base.env_state.get_info("agent_position")
        if pos_state and pos_state.current is not None:
            return tuple(pos_state.current)
        return base.agent.position

    def _compute_action_to_point(self, env, target: Tuple[float, float]) -> Tuple[float, float]:
        """模拟旧版 go(p2) 瞬间转向行为的多步等价实现。

        旧版行为分析（rules/jump_path.py go()函数）：
        - 旧版 MowerAgent.control(speed, steer) 允许一步内瞬间转任意角度
        - go(p2) 计算 delta_angle 并一步完成所有转向后直线前进

        新版等价实现：
        - 新版每步最多转 max_w 度（约28.6°）
        - 当角度差 > max_w 时：v=0 纯转向（分解旧版的瞬转）
        - 当角度差 <= max_w 时：同时转向+前进（等价于旧版对齐后的前进）

        这样多步的累积效果等价于旧版的单步瞬转+前进。
        """
        x, y = self._pose(env)
        dir_deg = self._heading_deg(env)
        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            return 0.0, 0.0
        goal_dir = math.degrees(math.atan2(dy, dx))
        delta = _normalize_deg(goal_dir - dir_deg)

        base = getattr(env, "unwrapped", env)
        v_bounds, w_bounds = base.action_processor.get_action_bounds()
        max_w = float(w_bounds[1])
        speed_max = float(v_bounds[1])

        # 裁剪转向角度到允许范围
        steer = max(w_bounds[0], min(delta, max_w))

        # 关键修复：角度差大于max_w时，先原地转向（v=0），等价于旧版瞬转的分解
        if abs(delta) > max_w:
            return 0.0, steer  # 纯转向，不前进

        # 角度差小于等于max_w，可以同时转向+前进
        speed = min(speed_max, max(float(v_bounds[0]), dist))
        return speed, steer

    def _compute_action_with_theta(self, env, target: Tuple[float, float],
                                    target_theta: float) -> Tuple[float, float]:
        """使用Dubins提供的切线方向进行路径跟踪。

        与_compute_action_to_point的区别：
        - 旧方法：用atan2(target - current)计算目标方向（弦方向）
        - 新方法：直接使用Dubins曲线的切线方向theta

        弧线上弦方向≠切线方向，使用切线方向可以：
        1. 保证路径跟踪的平滑性
        2. 避免不必要的v=0纯转向
        3. 符合物理动力学约束
        """
        x, y = self._pose(env)
        dir_deg = self._heading_deg(env)
        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            return 0.0, 0.0

        # 关键区别：使用传入的切线方向，而非atan2计算的弦方向
        delta = _normalize_deg(target_theta - dir_deg)

        base = getattr(env, "unwrapped", env)
        v_bounds, w_bounds = base.action_processor.get_action_bounds()
        max_w = float(w_bounds[1])
        speed_max = float(v_bounds[1])

        # 裁剪转向角度到允许范围
        steer = max(w_bounds[0], min(delta, max_w))

        # 使用正确的theta后，角度差通常很小，无需v=0纯转向
        # 但保留一个安全阈值处理异常情况（如采样间隔过大导致的急转弯）
        if abs(delta) > 2.0 * max_w:
            # 极端情况：减速转向（而非完全停止）
            speed = speed_max * 0.3
        else:
            speed = min(speed_max, max(float(v_bounds[0]), dist))

        return speed, steer

    def act(self, obs, env) -> Tuple[float, float]:
        if not self.path:
            return 0.0, 0.0
        # advance if reached
        x, y = self._pose(env)
        tx, ty = self.path[self.idx]
        if math.hypot(tx - x, ty - y) <= self.reach_eps and self.idx + 1 < len(self.path):
            self.idx += 1
            tx, ty = self.path[self.idx]
        return self._compute_action_to_point(env, (tx, ty))


# ---------- 具体规则策略 ----------

class BCPController(BaseRuleController):
    """基于 envs_new 几何的 Boustrophedon 覆盖（BCP）。

    几何核心：
    - 使用 field 的最小外接矩形 bounding_box 作为"牧场"；
    - 主方向沿长边方向 u，垂直方向 v；
    - 在 v 方向上按 FOV 宽度 Sw 等距生成多条平行扫线；
    - 每条扫线在像素网格中用 field 掩码裁剪出有效段，并按照蛇形顺序排列。

    动力学核心：
    - act() 使用简单的"先对齐航向、再前进"的 (v, w) 控制，遵守 env 的 v/w 上界。
    """

    def __init__(self, name: str = "BCP"):
        super().__init__(name)

    def reset(self, env) -> None:
        """在线版 BCP：按 bounding_box + 车宽生成平行线，走完一条再规划下一条 + Dubins 换行。"""
        super().reset(env)
        base = getattr(env, "unwrapped", env)

        maps = base.maps_dict
        if "field" in maps:
            field = maps["field"]
        else:
            raise RuntimeError("BCPController.reset: field map missing in maps_dict")
        if field.ndim != 2:
            raise RuntimeError(f"BCPController.reset: field map must be 2D, got shape={field.shape}")

        # 1) 解析 bounding_box，构造局部坐标系 (u, v)
        bbox_list = base.env_state.get_static_info("bounding_box")
        if not bbox_list:
            raise RuntimeError("BCPController.reset: bounding_box missing in env_state")
        box = np.array(bbox_list[0]).reshape(-1, 2).astype(float)  # (4,2)
        if box.shape[0] != 4 or box.shape[1] != 2:
            raise RuntimeError(f"BCPController.reset: unexpected bounding_box shape={box.shape}")

        max_len = -1.0
        best_pair: Tuple[np.ndarray, np.ndarray] | None = None
        for i in range(4):
            p1, p2 = box[i], box[(i + 1) % 4]
            l = float(np.linalg.norm(p2 - p1))
            if l > max_len:
                max_len, best_pair = l, (p1, p2)
        assert best_pair is not None and max_len > 0
        p1, p2 = best_pair

        u = (p2 - p1) / max_len
        v = np.array([-u[1], u[0]], dtype=float)
        center = box.mean(axis=0)
        self._rect_center = center
        self._u = u
        self._v = v
        self.main_angle = math.atan2(u[1], u[0])

        # v 方向范围
        ts = (box - center) @ v
        t_min, t_max = float(ts.min()), float(ts.max())
        if t_max <= t_min:
            raise RuntimeError("BCPController.reset: degenerate bounding_box along v-axis")
        self._t_min, self._t_max = t_min, t_max

        # 2) strip 间距：车宽（割草宽度）
        spacing = float(getattr(base.config, "agent_width", 1.0))
        self._strip_spacing = max(spacing, 1.0)

        # 3) 预计算掩码和对角线长度（用于生成直线）
        H, W = field.shape
        # 使用实际field mask而非bbox，避免路径点落在bbox内但field外导致碰撞
        self._field_mask = (field > 0).astype(np.uint8)
        # 同时保留bbox_mask用于weed过滤（保持兼容性）
        bbox_mask = np.zeros_like(field, dtype=np.uint8)
        cv2.fillPoly(bbox_mask, [box.astype(np.int32)], color=(1,))
        self._bbox_mask = bbox_mask
        min_x, min_y = box.min(axis=0)
        max_x, max_y = box.max(axis=0)
        self._diag_length = math.hypot(max_x - min_x, max_y - min_y)

        # 4) 初始化第一条 pass
        full_span = t_max - t_min
        # 使得 strips 居中覆盖整个 v 方向
        # 这里只计算第一条的 t 值，之后按 strip_spacing 递增
        self._pass_index = 0
        self._pass_v = t_min + self._strip_spacing / 2.0
        # 若 span 很大，可通过 ceil 调整；这里简单 clip 在 [t_min, t_max]
        self._pass_v = max(self._t_min, min(self._pass_v, self._t_max))
        self._pass_dir = 1  # +1: 沿 +u，-1: 沿 -u

        self._mode = "line"  # 当前段类型：'line' 或 'turn'
        self._next_line_waypoints: List[Tuple[float, float]] | None = None
        self._next_line_thetas: List[float] | None = None  # 下一条直线的theta
        self._pose_initialized = False

        # 生成首条直线 waypoints（包含theta）
        self.path, self.path_theta = self._generate_pass_line(self._pass_v, self._pass_dir)
        self.idx = 0

    # ---------- BCP 辅助几何 ----------

    def _rect_to_world(self, x_r: float, y_r: float) -> Tuple[float, float]:
        c = self._rect_center
        u, v = self._u, self._v
        p = c + x_r * u + y_r * v
        return float(p[0]), float(p[1])

    def _generate_pass_line(self, v_coord: float, direction: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """在局部坐标系中生成一条平行于 u 的直线，并映射回世界坐标。

        关键修正：使用field_mask而非bbox_mask进行边界检查，避免生成bbox内但field外的点。

        Returns:
            (path, path_theta): path是坐标列表，path_theta是每个点的方向（度数）
        """
        H, W = self._field_mask.shape
        # 以对角线长度为界，在 rect 中构造一条足够长的线段
        length = self._diag_length
        xs = np.arange(-length, length, 1.0)
        if direction < 0:
            xs = xs[::-1]

        points: List[Tuple[float, float]] = []
        for x_r in xs:
            x, y = self._rect_to_world(x_r, v_coord)
            ix, iy = int(round(x)), int(round(y))
            # 使用field_mask检查是否在实际田地内
            if 0 <= ix < W and 0 <= iy < H and self._field_mask[iy, ix] > 0:
                points.append((x, y))

        # 计算直线方向（所有点theta相同）
        line_theta = math.degrees(self.main_angle) % 360.0
        if direction < 0:
            line_theta = (line_theta + 180.0) % 360.0
        thetas = [line_theta] * len(points)

        return points, thetas

    def _has_next_pass(self) -> bool:
        """是否还存在下一条 pass（基于 v 方向投影判断）。"""
        return (self._pass_v + self._strip_spacing) <= (self._t_max + 1e-6)

    def _prepare_turn_to_next_pass(self, env) -> None:
        """从当前 pose 出发，为切换到下一条 pass 生成 Dubins 路径并缓存下一条直线。"""
        if not self._has_next_pass():
            self.path = []
            self.path_theta = []
            self.idx = 0
            self._mode = "done"
            return

        base = getattr(env, "unwrapped", env)
        # 计算下一条 pass 的参数
        next_v = self._pass_v + self._strip_spacing
        next_dir = -self._pass_dir
        # 先在几何上生成下一条 pass 的直线（包含theta）
        next_line_path, next_line_thetas = self._generate_pass_line(next_v, next_dir)
        if not next_line_path:
            # 若生成失败，直接标记为结束
            self.path = []
            self.path_theta = []
            self.idx = 0
            self._mode = "done"
            return

        self._next_line_waypoints = next_line_path
        self._next_line_thetas = next_line_thetas

        # 当前 pose
        x, y = self._pose(env)
        heading_deg = self._heading_deg(env)
        start_pose = (x, y, heading_deg)

        # 下一条 pass 的起点 pose：位置为 next_line[0]，方向沿 new pass_dir
        nx, ny = next_line_path[0]
        main_deg = math.degrees(self.main_angle) % 360.0
        if next_dir < 0:
            main_deg = (main_deg + 180.0) % 360.0
        end_pose = (nx, ny, main_deg)

        turn_waypoints_with_theta = _plan_dubins_path(env, start_pose, end_pose, step_size=1.0)
        # 保留Dubins的theta信息用于平滑跟踪
        self.path = [(px, py) for (px, py, _) in turn_waypoints_with_theta]
        self.path_theta = [theta for (_, _, theta) in turn_waypoints_with_theta]
        self.idx = 0
        self._mode = "turn"

        # 预先更新 pass 参数（几何上已确定）
        self._pass_v = next_v
        self._pass_dir = next_dir
        self._pass_index += 1

    def act(self, obs, env):
        base = getattr(env, "unwrapped", env)

        # 首次调用时，将 agent 放在首条线起点并对齐方向
        if not getattr(self, "_pose_initialized", False):
            if not self.path:
                return 0.0, 0.0
            sx, sy = self.path[0]
            base.agent.set_position(sx, sy)
            base.agent.set_direction(math.degrees(self.main_angle) % 360.0)
            pos_state = base.env_state.get_info("agent_position")
            if pos_state is not None:
                pos_state.reset((sx, sy))
            dir_state = base.env_state.get_info("agent_direction")
            if dir_state is not None:
                dir_state.reset(base.agent.direction)
            self._pose_initialized = True
            self._mode = "line"
            self.idx = 0

        # 所有pass都结束
        if self._mode == "done":
            return 0.0, 0.0

        # TURN 模式：跟随 Dubins 弧段（使用theta追踪）
        if self._mode == "turn":
            if self.idx >= len(self.path):
                # Dubins 段结束，切入下一条直线
                if not self._next_line_waypoints:
                    self._mode = "done"
                    return 0.0, 0.0
                self.path = self._next_line_waypoints
                self.path_theta = self._next_line_thetas if self._next_line_thetas else []
                self.idx = 0
                self._next_line_waypoints = None
                self._next_line_thetas = None
                self._mode = "line"
            else:
                target = self.path[self.idx]
                x, y = self._pose(env)
                if math.hypot(target[0] - x, target[1] - y) <= self.reach_eps:
                    self.idx += 1
                    if self.idx >= len(self.path):
                        # 下一步切入 line 模式
                        if not self._next_line_waypoints:
                            self._mode = "done"
                            return 0.0, 0.0
                        self.path = self._next_line_waypoints
                        self.path_theta = self._next_line_thetas if self._next_line_thetas else []
                        self.idx = 0
                        self._next_line_waypoints = None
                        self._next_line_thetas = None
                        self._mode = "line"
                        target = self.path[self.idx]
                    else:
                        target = self.path[self.idx]
                # 使用theta追踪Dubins弧段
                target_theta = self.path_theta[self.idx] if self.idx < len(self.path_theta) else self._heading_deg(env)
                return self._compute_action_with_theta(env, target, target_theta)

        # LINE 模式：沿当前 pass 扫描（使用theta追踪）
        if self._mode == "line":
            if not self.path:
                # 当前 pass 为空，尝试切到下一条或结束
                if self._has_next_pass():
                    self._prepare_turn_to_next_pass(env)
                    return 0.0, 0.0
                else:
                    self._mode = "done"
                    return 0.0, 0.0

            if self.idx >= len(self.path):
                # 一条 pass 走完，准备 Dubins 换行
                if self._has_next_pass():
                    self._prepare_turn_to_next_pass(env)
                    return 0.0, 0.0
                else:
                    self._mode = "done"
                    return 0.0, 0.0

            target = self.path[self.idx]
            x, y = self._pose(env)
            if math.hypot(target[0] - x, target[1] - y) <= self.reach_eps:
                self.idx += 1
                if self.idx >= len(self.path):
                    if self._has_next_pass():
                        self._prepare_turn_to_next_pass(env)
                        return 0.0, 0.0
                    else:
                        self._mode = "done"
                        return 0.0, 0.0
                else:
                    target = self.path[self.idx]

            # 使用theta追踪直线段
            target_theta = self.path_theta[self.idx] if self.idx < len(self.path_theta) else self._heading_deg(env)
            return self._compute_action_with_theta(env, target, target_theta)

        return 0.0, 0.0


class SnakeController(BCPController):
    """SNAKE 规则算法：在 BCP 扫线基础上，遇到可行杂草则 Dubins 跳上去并沿该方向继续覆盖。"""

    def __init__(self, name: str = "SNAKE", restrict_band: bool = False):
        super().__init__(name)
        self._restrict_band = restrict_band

    def reset(self, env) -> None:
        """继承 BCP 几何，附加 Snake 所需参数。"""
        super().reset(env)
        base = getattr(env, "unwrapped", env)
        self._snake_sw = _fov_strip_spacing(base.config)
        self._snake_B = float(getattr(base.config, "agent_width", 1.0))

        w_max = abs(float(base.config.w_max)) * math.pi / 180.0
        v_max = float(base.config.v_max)
        if w_max <= 0.0:
            raise RuntimeError("SnakeController.reset: invalid w_max in config")
        self._turning_radius = v_max / w_max

        # detour状态追踪：防止在执行detour期间重复规划
        self._detour_active = False
        self._detour_end_idx = 0

    # ---------- 辅助：已发现杂草集合 ----------

    def _get_discovered_weeds_world_rect(self, env) -> tuple[np.ndarray, np.ndarray]:
        """与 Jump 相同：返回 (world_coords, rect_coords) 的已发现杂草集合。"""
        base = getattr(env, "unwrapped", env)
        maps = base.maps_dict
        weed = maps.get("weed")
        mist = maps.get("mist")
        if weed is None or mist is None:
            raise RuntimeError("SnakeController requires both 'weed' and 'mist' maps in maps_dict")
        if weed.shape != mist.shape:
            raise RuntimeError("SnakeController: weed and mist maps must have the same shape")

        discovered = (weed > 0) & (mist > 0)
        if hasattr(self, "_bbox_mask") and self._bbox_mask.shape == weed.shape:
            discovered &= self._bbox_mask.astype(bool)

        ys, xs = np.nonzero(discovered)
        if xs.size == 0:
            empty = np.zeros((0, 2), dtype=float)
            return empty, empty

        pts_world = np.stack([xs.astype(float), ys.astype(float)], axis=1)
        dx = pts_world - self._rect_center
        x_r = dx @ self._u
        y_r = dx @ self._v
        pts_rect = np.stack([x_r, y_r], axis=1)
        return pts_world, pts_rect

    # ---------- pass 间距（基于 Snake 规则） ----------

    def _compute_next_pass_v(self, env) -> float | None:
        """论文公式(3): yp(i+1) = min{yp(i) + Sw/2 + B/2, W - B/2}

        SNAKE与JUMP的区别：跳跃后不返回当前pass，继续向前覆盖。
        因此换行公式不考虑杂草位置——agent从杂草位置自然向前扫描覆盖。
        """
        cur_v = self._pass_v
        B = self._snake_B
        Sw = self._snake_sw

        # 论文公式(3)：只有两项
        c1 = cur_v + B / 2.0 + Sw / 2.0  # 固定间距
        top_center = self._t_max - B / 2.0  # 顶部边界 W - B/2

        v_next = min(c1, top_center)
        if v_next <= cur_v + 1e-6:
            return None
        return v_next

    def _prepare_turn_to_next_pass(self, env) -> None:
        """Snake 版换行：使用上面的 v_next 规则。"""
        # 重置detour状态：进入新的pass时清除detour标志
        self._detour_active = False
        self._detour_end_idx = 0

        v_next = self._compute_next_pass_v(env)
        if v_next is None:
            self.path = []
            self.path_theta = []
            self.idx = 0
            self._mode = "done"
            return

        next_dir = -self._pass_dir
        next_line_path, next_line_thetas = self._generate_pass_line(v_next, next_dir)
        if not next_line_path:
            self.path = []
            self.path_theta = []
            self.idx = 0
            self._mode = "done"
            return

        x, y = self._pose(env)
        heading_deg = self._heading_deg(env)
        start_pose = (x, y, heading_deg)

        # SNAKE 关键修复：选择离当前位置最近的端点作为换行目标
        # BCP 假设 Agent 在 pass 末端，但 SNAKE 可能在任意位置结束（因为 detour）
        d0 = math.hypot(next_line_path[0][0] - x, next_line_path[0][1] - y)
        d_last = math.hypot(next_line_path[-1][0] - x, next_line_path[-1][1] - y)

        if d_last < d0:
            # 反转 next_line，从更近的端点开始（同时反转theta并调整方向）
            next_line_path = next_line_path[::-1]
            # theta也要反转顺序，但方向值需要+180°（因为行进方向反了）
            next_line_thetas = [(theta + 180.0) % 360.0 for theta in reversed(next_line_thetas)]
            next_dir = -next_dir  # 方向也反转

        self._next_line_waypoints = next_line_path
        self._next_line_thetas = next_line_thetas

        nx, ny = next_line_path[0]
        main_deg = math.degrees(self.main_angle) % 360.0
        if next_dir < 0:
            main_deg = (main_deg + 180.0) % 360.0
        end_pose = (nx, ny, main_deg)

        turn_waypoints_with_theta = _plan_dubins_path(env, start_pose, end_pose, step_size=1.0)
        # 保留Dubins的theta信息用于平滑跟踪
        self.path = [(px, py) for (px, py, _) in turn_waypoints_with_theta]
        self.path_theta = [theta for (_, _, theta) in turn_waypoints_with_theta]
        self.idx = 0
        self._mode = "turn"

        self._pass_v = v_next
        self._pass_dir = next_dir
        self._pass_index += 1

    # ---------- 局部 Dubins "蛇行" ----------

    def _maybe_plan_snake_detour(self, env) -> bool:
        """按 SNAKE / R-SNAKE 规则：跳到前方杂草并从该位置继续直线覆盖。"""
        # 状态检查：如果正在执行detour，不要重复规划
        if self._detour_active and self.idx < self._detour_end_idx:
            return False

        if not self.path or self.idx >= len(self.path):
            return False

        base = getattr(env, "unwrapped", env)
        weeds_world, _ = self._get_discovered_weeds_world_rect(env)
        if weeds_world.shape[0] == 0:
            return False

        x, y = self._pose(env)
        pos = np.array([x, y], dtype=float)

        rad_vec = self._u if self._pass_dir >= 0 else -self._u
        delta = weeds_world - pos
        forward_mask = (delta @ rad_vec) > 0

        if not np.any(forward_mask):
            return False

        weeds_world = weeds_world[forward_mask]
        delta = delta[forward_mask]

        # R-SNAKE：额外限制“垂直偏移”不超过若干倍视野宽度
        if self._restrict_band:
            Sw = self._snake_sw
            v_vec = self._v
            side = delta @ v_vec
            band_mask = side > -1.5 * Sw
            weeds_world = weeds_world[band_mask]
            delta = delta[band_mask]
            if weeds_world.shape[0] == 0:
                return False

        # Dubins 可行性：距离至少 2R
        dists = np.linalg.norm(delta, axis=1)
        feasible = dists >= 2.0 * self._turning_radius
        if not np.any(feasible):
            return False

        idx_best = int(np.argmin(dists[feasible]))
        weed_world = weeds_world[feasible][idx_best]

        # 规划：当前姿态 -> weed（方向沿 pass），再从 weed 沿 pass 方向一直走到 bbox 边界
        main_deg = math.degrees(self.main_angle) % 360.0
        if self._pass_dir < 0:
            main_deg = (main_deg + 180.0) % 360.0

        start_pose = (pos[0], pos[1], self._heading_deg(env))
        weed_pose = (float(weed_world[0]), float(weed_world[1]), main_deg)

        try:
            to_weed_with_theta = _plan_dubins_path(env, start_pose, weed_pose, step_size=1.0)
            # 保留Dubins的theta信息用于平滑跟踪
            to_weed = [(px, py) for (px, py, _) in to_weed_with_theta]
            to_weed_thetas = [theta for (_, _, theta) in to_weed_with_theta]
        except Exception:
            return False

        # 从 weed 开始生成一条"向前"的直线，直到离开 field（不是bbox）
        H, W = self._field_mask.shape
        rad_vec_norm = rad_vec / np.linalg.norm(rad_vec)
        forward_line: List[Tuple[float, float]] = []
        max_len = self._diag_length * 2.0
        start = np.array([weed_world[0], weed_world[1]], dtype=float)
        for s in np.arange(0.0, max_len, 1.0):
            pt = start + s * rad_vec_norm
            ix, iy = int(round(pt[0])), int(round(pt[1]))
            # 使用field_mask检查是否在实际田地内，避免生成bbox内但field外的点
            if 0 <= ix < W and 0 <= iy < H and self._field_mask[iy, ix] > 0:
                forward_line.append((float(pt[0]), float(pt[1])))
            else:
                break

        if not forward_line:
            return False

        # 直线段的theta都相同（沿main_deg方向）
        forward_line_thetas = [main_deg] * len(forward_line)

        self.path = to_weed + forward_line
        self.path_theta = to_weed_thetas + forward_line_thetas
        self.idx = 0

        # 设置detour状态：标记正在执行detour，记录detour结束位置
        self._detour_active = True
        self._detour_end_idx = len(to_weed) + len(forward_line)

        # 当前 pass 的几何高度改为该直线所在的 v 值
        dx = np.array([weed_world[0], weed_world[1]]) - self._rect_center
        self._pass_v = float(dx @ self._v)
        return True

    # ---------- 主逻辑 ----------

    def act(self, obs, env):
        base = getattr(env, "unwrapped", env)

        if not getattr(self, "_pose_initialized", False):
            if not self.path:
                return 0.0, 0.0
            sx, sy = self.path[0]
            base.agent.set_position(sx, sy)
            base.agent.set_direction(math.degrees(self.main_angle) % 360.0)
            pos_state = base.env_state.get_info("agent_position")
            if pos_state is not None:
                pos_state.reset((sx, sy))
            dir_state = base.env_state.get_info("agent_direction")
            if dir_state is not None:
                dir_state.reset(base.agent.direction)
            self._pose_initialized = True
            self._mode = "line"
            self.idx = 0

        if self._mode == "done":
            return 0.0, 0.0

        # TURN：沿 Dubins 换行（使用theta追踪）
        if self._mode == "turn":
            if self.idx >= len(self.path):
                if not self._next_line_waypoints:
                    self._mode = "done"
                    return 0.0, 0.0
                self.path = self._next_line_waypoints
                self.path_theta = self._next_line_thetas if self._next_line_thetas else []
                self.idx = 0
                self._next_line_waypoints = None
                self._next_line_thetas = None
                self._mode = "line"
            else:
                target = self.path[self.idx]
                x, y = self._pose(env)
                if math.hypot(target[0] - x, target[1] - y) <= self.reach_eps:
                    self.idx += 1
                    if self.idx >= len(self.path):
                        if not self._next_line_waypoints:
                            self._mode = "done"
                            return 0.0, 0.0
                        self.path = self._next_line_waypoints
                        self.path_theta = self._next_line_thetas if self._next_line_thetas else []
                        self.idx = 0
                        self._next_line_waypoints = None
                        self._next_line_thetas = None
                        self._mode = "line"
                        target = self.path[self.idx]
                    else:
                        target = self.path[self.idx]
                # 使用theta追踪Dubins弧段
                target_theta = self.path_theta[self.idx] if self.idx < len(self.path_theta) else self._heading_deg(env)
                return self._compute_action_with_theta(env, target, target_theta)

        # LINE：扫线 + 可能触发"蛇形" detour（使用theta追踪）
        if self._mode == "line":
            if not self.path:
                self._prepare_turn_to_next_pass(env)
                return 0.0, 0.0

            if self.idx >= len(self.path):
                self._prepare_turn_to_next_pass(env)
                return 0.0, 0.0

            # 在当前 pass 上尝试一次 detour
            self._maybe_plan_snake_detour(env)

            target = self.path[self.idx]
            x, y = self._pose(env)
            if math.hypot(target[0] - x, target[1] - y) <= self.reach_eps:
                self.idx += 1
                if self.idx >= len(self.path):
                    self._prepare_turn_to_next_pass(env)
                    return 0.0, 0.0
                else:
                    target = self.path[self.idx]

            # 使用theta追踪
            target_theta = self.path_theta[self.idx] if self.idx < len(self.path_theta) else self._heading_deg(env)
            return self._compute_action_with_theta(env, target, target_theta)

        return 0.0, 0.0


class RSnakeController(SnakeController):
    """R-SNAKE：在 SNAKE 基础上，限制可跳杂草的纵向带宽。"""

    def __init__(self):
        super().__init__(name="R_SNAKE", restrict_band=True)


class JumpController(BCPController):
    """JUMP 规则算法：在 BCP 扫线基础上插入 Dubins“跳跃”。

    本实现直接对应论文与旧 rules 中的在线版本：
    - pass 间距由 Eq.(1) 的三项最小值控制；
    - detour 仅对已“发现”的杂草（weed & mist）做 Dubins 跳跃，并返回同一条 pass；
    - 仅在几何上有足够空间时才插入 detour，保持 Dubins 曲线可行。
    """

    def __init__(self):
        super().__init__("JUMP")

    def reset(self, env) -> None:
        """继承 BCP 的几何初始化，并设置 JUMP 相关参数。"""
        super().reset(env)
        base = getattr(env, "unwrapped", env)
        self._jump_sw = _fov_strip_spacing(base.config)  # 视野宽度 Sw
        self._jump_B = float(getattr(base.config, "agent_width", 1.0))  # 割草宽度 B

        # Dubins 最小转弯半径
        w_max = abs(float(base.config.w_max)) * math.pi / 180.0
        v_max = float(base.config.v_max)
        if w_max <= 0.0:
            raise RuntimeError("JumpController.reset: invalid w_max in config")
        self._turning_radius = v_max / w_max

        # detour状态追踪：防止在执行detour期间重复规划
        self._detour_active = False
        self._detour_end_idx = 0

    # ---------- JUMP 辅助：weed 集合与 pass spacing ----------

    def _get_discovered_weeds_world_rect(self, env) -> tuple[np.ndarray, np.ndarray]:
        """返回 (world_coords, rect_coords) 形式的“已发现且未割杂草”集合。

        - world_coords: 形状 (N, 2)，在全局像素坐标系下；
        - rect_coords:  形状 (N, 2)，在 (u, v) 局部坐标系下。

        “已发现”定义为：weed==1 且 mist==1；进一步用 bbox_mask 剔除场外像素。
        """
        base = getattr(env, "unwrapped", env)
        maps = base.maps_dict
        weed = maps.get("weed")
        mist = maps.get("mist")
        if weed is None or mist is None:
            raise RuntimeError("JumpController requires both 'weed' and 'mist' maps in maps_dict")
        if weed.shape != mist.shape:
            raise RuntimeError("JumpController: weed and mist maps must have the same shape")

        discovered = (weed > 0) & (mist > 0)
        # 仅保留 bounding box 内部的杂草
        if hasattr(self, "_bbox_mask") and self._bbox_mask.shape == weed.shape:
            discovered &= self._bbox_mask.astype(bool)

        ys, xs = np.nonzero(discovered)
        if xs.size == 0:
            empty = np.zeros((0, 2), dtype=float)
            return empty, empty

        pts_world = np.stack([xs.astype(float), ys.astype(float)], axis=1)
        dx = pts_world - self._rect_center
        x_r = dx @ self._u
        y_r = dx @ self._v
        pts_rect = np.stack([x_r, y_r], axis=1)
        return pts_world, pts_rect

    def _compute_next_pass_v(self, env) -> float | None:
        """Eq.(1)：v_next = min(yp + Sw/2, yp + offset_to_nearest_above_weed + B/2, W-B/2)。

        关键修正：使用相对偏移而非绝对坐标。
        旧代码：y_offset + find_offset(weed) + B/2 (相对偏移)
        """
        cur_v = self._pass_v
        B = self._jump_B
        Sw = self._jump_sw

        # 第一项：基于 FOV 的最大探索步长
        c1 = cur_v + Sw / 2.0

        # 第二项：当前pass上方最近杂草的相对偏移 + B/2
        _, weeds_rect = self._get_discovered_weeds_world_rect(env)
        if weeds_rect.shape[0] > 0:
            # 计算每个杂草相对于当前pass的偏移
            offsets_from_cur = weeds_rect[:, 1] - cur_v
            # 只考虑在当前pass上方的杂草 (offset > 0)
            above_mask = offsets_from_cur > 0
            if np.any(above_mask):
                min_offset = float(offsets_from_cur[above_mask].min())
                c2 = cur_v + min_offset + B / 2.0
            else:
                c2 = float("inf")
        else:
            c2 = float("inf")

        # 第三项：顶部 pass（中心距离上边 B/2）
        top_center = self._t_max - B / 2.0
        c3 = top_center

        v_next = min(c1, c2, c3)
        if v_next <= cur_v + 1e-6:
            return None
        return v_next

    def _prepare_turn_to_next_pass(self, env) -> None:
        """JUMP 版：使用动态 v_next 而非固定 strip_spacing。"""
        # 重置detour状态：进入新的pass时清除detour标志
        self._detour_active = False
        self._detour_end_idx = 0

        v_next = self._compute_next_pass_v(env)
        if v_next is None:
            self.path = []
            self.path_theta = []
            self.idx = 0
            self._mode = "done"
            return

        # 方向翻转（蛇形）
        next_dir = -self._pass_dir
        next_line_path, next_line_thetas = self._generate_pass_line(v_next, next_dir)
        if not next_line_path:
            self.path = []
            self.path_theta = []
            self.idx = 0
            self._mode = "done"
            return

        self._next_line_waypoints = next_line_path
        self._next_line_thetas = next_line_thetas

        # 当前 pose
        x, y = self._pose(env)
        heading_deg = self._heading_deg(env)
        start_pose = (x, y, heading_deg)

        # 下一条 pass 起点 pose
        nx, ny = next_line_path[0]
        main_deg = math.degrees(self.main_angle) % 360.0
        if next_dir < 0:
            main_deg = (main_deg + 180.0) % 360.0
        end_pose = (nx, ny, main_deg)

        turn_waypoints_with_theta = _plan_dubins_path(env, start_pose, end_pose, step_size=1.0)
        # 保留Dubins的theta信息用于平滑跟踪
        self.path = [(px, py) for (px, py, _) in turn_waypoints_with_theta]
        self.path_theta = [theta for (_, _, theta) in turn_waypoints_with_theta]
        self.idx = 0
        self._mode = "turn"

        # 更新 pass 参数
        self._pass_v = v_next
        self._pass_dir = next_dir
        self._pass_index += 1

    # ---------- JUMP 辅助：局部 Dubins 跳跃 ----------

    def _maybe_plan_jump_detour(self, env) -> bool:
        """按照旧 rules 逻辑，在当前 pass 上插入一次局部 Dubins detour。

        - 仅考虑"前方 + 上方"的已发现杂草；
        - 选择沿扫描方向最近的一株；
        - 若在当前 pass 上有足够空间（i±4R）则插入两段 Dubins：
            pass[start_i] -> weed -> pass[end_i]。
        """
        # 状态检查：如果正在执行detour，不要重复规划
        if self._detour_active and self.idx < self._detour_end_idx:
            return False

        if not self.path or self.idx >= len(self.path):
            return False

        base = getattr(env, "unwrapped", env)
        weeds_world, weeds_rect = self._get_discovered_weeds_world_rect(env)
        if weeds_world.shape[0] == 0:
            return False

        # 当前姿态在 world / rect 坐标中的表示
        x, y = self._pose(env)
        pos = np.array([x, y], dtype=float)
        dx_cur = pos - self._rect_center
        x_rc = float(dx_cur @ self._u)
        y_rc = float(dx_cur @ self._v)

        # 当前 pass 上所有路径点在 rect 中的 x_r，用于查找 i
        xs_line = []
        for px, py in self.path:
            dlt = np.array([px, py]) - self._rect_center
            xs_line.append(float(dlt @ self._u))
        xs_line = np.asarray(xs_line)

        # 1) 过滤：仅保留前方 + 上方的杂草
        rad_vec = self._u if self._pass_dir >= 0 else -self._u
        up_vec = self._v
        delta = weeds_world - pos
        forward_mask = (delta @ rad_vec) > 0
        upward_mask = (delta @ up_vec) > 0
        mask = forward_mask & upward_mask
        if not np.any(mask):
            return False

        weeds_world = weeds_world[mask]
        weeds_rect = weeds_rect[mask]

        # 2) 选择沿 u 方向最近的一株
        x_rw_all = weeds_rect[:, 0]
        idx_best = int(np.argmin(np.abs(x_rw_all - x_rc)))
        weed_world = weeds_world[idx_best]
        x_rw = float(x_rw_all[idx_best])

        # 3) 在当前 pass 上找到与该 weed 对齐的索引 i
        i = int(np.argmin(np.abs(xs_line - x_rw)))
        R = self._turning_radius
        k = max(4, int(math.ceil(4.0 * R)))
        p_i = self.idx

        # 可行性检查（对应旧代码里的 i±4R 条件）
        if i < p_i + k + 4 or i - k < 0 or i + k >= len(self.path) or i + k + 1 >= len(self.path):
            # 跳跃窗口过小：略过该区域，避免在末端或太近处强行 jump
            self.idx = min(len(self.path) - 1, max(self.idx, i + 2))
            return False

        start_i = int(i - k)
        end_i = int(i + k)

        # 4) 组装新路径：prefix + Dubins(to weed) + Dubins(back) + suffix
        main_deg = math.degrees(self.main_angle) % 360.0
        if self._pass_dir < 0:
            main_deg = (main_deg + 180.0) % 360.0

        prefix = self.path[self.idx:start_i + 1]
        # 同步提取prefix对应的theta
        prefix_theta = self.path_theta[self.idx:start_i + 1] if self.idx < len(self.path_theta) else [main_deg] * len(prefix)

        start_pt = self.path[start_i]
        end_pt = self.path[end_i]

        start_pose = (start_pt[0], start_pt[1], main_deg)
        weed_pose = (float(weed_world[0]), float(weed_world[1]), main_deg)
        end_pose = (end_pt[0], end_pt[1], main_deg)

        try:
            to_weed_with_theta = _plan_dubins_path(env, start_pose, weed_pose, step_size=1.0)
            to_weed = [(px, py) for (px, py, _) in to_weed_with_theta]
            to_weed_thetas = [theta for (_, _, theta) in to_weed_with_theta]

            back_to_line_with_theta = _plan_dubins_path(env, weed_pose, end_pose, step_size=1.0)
            back_to_line = [(px, py) for (px, py, _) in back_to_line_with_theta]
            back_to_line_thetas = [theta for (_, _, theta) in back_to_line_with_theta]
        except Exception:
            return False

        suffix = self.path[end_i:]
        # 同步提取suffix对应的theta
        suffix_theta = self.path_theta[end_i:] if end_i < len(self.path_theta) else [main_deg] * len(suffix)

        self.path = prefix + to_weed + back_to_line + suffix
        self.path_theta = prefix_theta + to_weed_thetas + back_to_line_thetas + suffix_theta
        self.idx = 0

        # 设置detour状态：标记正在执行detour，记录detour结束位置
        self._detour_active = True
        self._detour_end_idx = len(prefix) + len(to_weed) + len(back_to_line)

        return True

    # ---------- 主逻辑 ----------

    def act(self, obs, env):
        base = getattr(env, "unwrapped", env)

        # 首次初始化姿态
        if not getattr(self, "_pose_initialized", False):
            if not self.path:
                return 0.0, 0.0
            sx, sy = self.path[0]
            base.agent.set_position(sx, sy)
            base.agent.set_direction(math.degrees(self.main_angle) % 360.0)
            pos_state = base.env_state.get_info("agent_position")
            if pos_state is not None:
                pos_state.reset((sx, sy))
            dir_state = base.env_state.get_info("agent_direction")
            if dir_state is not None:
                dir_state.reset(base.agent.direction)
            self._pose_initialized = True
            self._mode = "line"
            self.idx = 0

        if self._mode == "done":
            return 0.0, 0.0

        # TURN 模式：复用 BCP 的 turn 逻辑，只是换行高度由 JUMP 规则决定
        if self._mode == "turn":
            if self.idx >= len(self.path):
                if not self._next_line_waypoints:
                    self._mode = "done"
                    return 0.0, 0.0
                self.path = self._next_line_waypoints
                self.path_theta = self._next_line_thetas if self._next_line_thetas else []
                self.idx = 0
                self._next_line_waypoints = None
                self._next_line_thetas = None
                self._mode = "line"
            else:
                target = self.path[self.idx]
                x, y = self._pose(env)
                if math.hypot(target[0] - x, target[1] - y) <= self.reach_eps:
                    self.idx += 1
                    if self.idx >= len(self.path):
                        if not self._next_line_waypoints:
                            self._mode = "done"
                            return 0.0, 0.0
                        self.path = self._next_line_waypoints
                        self.path_theta = self._next_line_thetas if self._next_line_thetas else []
                        self.idx = 0
                        self._next_line_waypoints = None
                        self._next_line_thetas = None
                        self._mode = "line"
                        target = self.path[self.idx]
                    else:
                        target = self.path[self.idx]
                # 使用theta追踪：从path_theta获取目标方向
                target_theta = self.path_theta[self.idx] if self.idx < len(self.path_theta) else self._heading_deg(env)
                return self._compute_action_with_theta(env, target, target_theta)

        # LINE 模式：沿当前 pass 直线，同时可能触发多次 jump
        if self._mode == "line":
            if not self.path:
                # 当前 pass 为空，尝试换行或结束
                self._prepare_turn_to_next_pass(env)
                return 0.0, 0.0

            if self.idx >= len(self.path):
                self._prepare_turn_to_next_pass(env)
                return 0.0, 0.0

            # 尝试在当前 pass 上插入一次 detour（如果几何可行）
            self._maybe_plan_jump_detour(env)

            # 跟踪当前 path（使用theta追踪）
            target = self.path[self.idx]
            x, y = self._pose(env)
            if math.hypot(target[0] - x, target[1] - y) <= self.reach_eps:
                self.idx += 1
                if self.idx >= len(self.path):
                    self._prepare_turn_to_next_pass(env)
                    return 0.0, 0.0
                else:
                    target = self.path[self.idx]

            # 使用theta追踪：从path_theta获取目标方向
            target_theta = self.path_theta[self.idx] if self.idx < len(self.path_theta) else self._heading_deg(env)
            return self._compute_action_with_theta(env, target, target_theta)

        return 0.0, 0.0


class ReactController(BaseRuleController):
    """基于杂草的 REACT 策略（Dubins 版本）。

    忠实于旧 rules 与论文的核心思路：
    - 若存在“已发现杂草”（weed & 已探索），优先用 Dubins 路径依次吃掉最近的杂草；
    - 若暂时没有杂草，则随机挑选一个“尚未探索的田地点”作为目标，用 Dubins 轨迹前往；
    - 整个过程是在线的、反应式的，而不是预先规划整条路径。
    """

    def __init__(self, name: str = "REACT"):
        super().__init__(name)
        self.mode: str = "explore"  # 'explore' | 'weed'
        self.target: Tuple[float, float] | None = None
        self._weed_clearing_active: bool = False  # 是否正在批量清除杂草

    def _get_discovered_weeds_world(self, env) -> np.ndarray:
        """weed>0 且已被视野探开的像素集合。"""
        base = getattr(env, "unwrapped", env)
        maps = base.maps_dict
        weed = maps.get("weed")
        if weed is None:
            return np.zeros((0, 2), dtype=float)

        mist = maps.get("mist")
        discovered = weed > 0
        if mist is not None:
            if mist.shape != weed.shape:
                raise RuntimeError("ReactController: weed and mist maps must have same shape")
            discovered &= mist > 0

        ys, xs = np.nonzero(discovered)
        if xs.size == 0:
            return np.zeros((0, 2), dtype=float)
        return np.stack([xs.astype(float), ys.astype(float)], axis=1)

    def reset(self, env) -> None:
        super().reset(env)
        self.mode = "explore"
        self.target = None
        self.path = []
        self.idx = 0
        self._weed_clearing_active = False

    # ---------- 规划子程序 ----------

    def _plan_dubins_to_point(self, env, goal: Tuple[float, float]) -> None:
        """从当前姿态到 goal 规划一条 Dubins 路径，并写入 self.path 和 self.path_theta。"""
        gx, gy = goal
        x, y = self._pose(env)
        heading_deg = self._heading_deg(env)

        # 目标朝向取"从当前位置指向目标"的方向，保证 Dubins 路径合理
        goal_dir_deg = math.degrees(math.atan2(gy - y, gx - x))
        start_pose = (x, y, heading_deg)
        end_pose = (gx, gy, goal_dir_deg)

        # _plan_dubins_path现在返回(x, y, theta)三元组
        inner_pts_with_theta = _plan_dubins_path(env, start_pose, end_pose, step_size=1.0)

        # 解包坐标和theta
        self.path = [(px, py) for (px, py, _) in inner_pts_with_theta]
        self.path_theta = [theta for (_, _, theta) in inner_pts_with_theta]

        # 加上终点以保证真正到达
        self.path.append((gx, gy))
        self.path_theta.append(goal_dir_deg)  # 终点方向
        self.idx = 0

    def _sample_explore_goal(self, env) -> Tuple[float, float] | None:
        """在field内随机采样一个"未探索"的目标点。

        论文："moves towards a randomly selected unexplored area"
        关键修正：使用field mask而非bbox，避免选择bbox内但field外的点导致碰撞。
        """
        base = getattr(env, "unwrapped", env)
        maps = base.maps_dict
        mist = maps.get("mist")
        field = maps.get("field")

        if mist is None or field is None:
            return None

        # 使用实际field掩码而非bbox
        field_mask = (field > 0).astype(np.uint8)

        # 选择：mist未探索 & 在实际田地内
        candidate = (mist == 0) & (field_mask > 0)
        ys, xs = np.nonzero(candidate)

        if xs.size == 0:
            # 全部探开时退化为field内任意点
            ys, xs = np.nonzero(field_mask > 0)

        if xs.size == 0:
            return None

        k = random.randint(0, xs.size - 1)
        return float(xs[k]), float(ys[k])

    def _plan_straight_to_point(self, env, goal: Tuple[float, float]) -> None:
        """从当前位置到goal规划直线路径，只保留田地内的点。

        这是旧版REACT的行为：
        1. 画直线从当前位置到目标
        2. 只保留落在field mask内的点作为valid_points
        3. 沿valid_points导航
        """
        gx, gy = goal
        x, y = self._pose(env)

        # 获取田地mask
        base = getattr(env, "unwrapped", env)
        field = base.maps_dict.get("field")
        if field is None:
            self.path = [(gx, gy)]
            # 直线路径的theta就是指向目标的方向
            self.path_theta = [math.degrees(math.atan2(gy - y, gx - x))]
            self.idx = 0
            return

        # 生成直线上的采样点
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            self.path = [(gx, gy)]
            self.path_theta = [math.degrees(math.atan2(dy, dx)) if dist > 1e-6 else 0.0]
            self.idx = 0
            return

        num_pts = int(dist)
        step_x, step_y = dx / num_pts, dy / num_pts
        raw_points = [(x + step_x * i, y + step_y * i) for i in range(1, num_pts + 1)]
        raw_points.append((gx, gy))

        # 只保留在田地内的点（关键过滤！）
        H, W = field.shape
        valid_points = []
        for px, py in raw_points:
            ix, iy = int(px), int(py)
            if 0 <= ix < W and 0 <= iy < H and field[iy, ix] > 0:
                valid_points.append((px, py))

        self.path = valid_points if valid_points else [(gx, gy)]
        # 直线路径上所有点的theta都是相同的（指向目标方向）
        line_theta = math.degrees(math.atan2(dy, dx))
        self.path_theta = [line_theta] * len(self.path)
        self.idx = 0

    # ---------- 主逻辑 ----------

    def act(self, obs, env):
        """REACT核心逻辑（修复版 - 等价于旧版rules while循环）

        旧版行为：
        1. 随机选点 → 沿探索路径行进
        2. 边走边检查杂草（每步都检查）
        3. 发现杂草 → 用while循环清除**所有**可见杂草
        4. 清完后break → 重新随机选点

        新版等价实现（状态机模拟while循环）：
        - weed模式 = "正在清除杂草"
        - weed路径完成后：检查是否还有杂草
          - 有 → 继续追下一个杂草（保持weed模式，等价于while循环）
          - 无 → 切explore模式，重新随机选点
        """

        # 1) weed模式路径完成：检查是否需要继续清除（等价于while weed is not None）
        if self.mode == "weed" and not self.path:
            weeds = self._get_discovered_weeds_world(env)
            if weeds.shape[0] > 0:
                # ★关键修复：还有杂草，继续追！（等价于旧版while循环内的下一次迭代）
                x, y = self._pose(env)
                rel = weeds - np.array([x, y], dtype=float)
                dists = np.linalg.norm(rel, axis=1)
                idx_min = int(np.argmin(dists))
                target = (float(weeds[idx_min, 0]), float(weeds[idx_min, 1]))
                self.target = target
                self._plan_dubins_to_point(env, target)
            else:
                # ★清完所有杂草后才切explore（等价于旧版while循环退出后break）
                self.mode = "explore"

        # 2) explore模式路径完成：检查杂草或随机选新目标
        if self.mode == "explore" and not self.path:
            # 先检查是否有新发现的杂草（边走边检查）
            weeds = self._get_discovered_weeds_world(env)
            if weeds.shape[0] > 0:
                # 发现杂草，切weed模式
                self.mode = "weed"
                x, y = self._pose(env)
                rel = weeds - np.array([x, y], dtype=float)
                dists = np.linalg.norm(rel, axis=1)
                idx_min = int(np.argmin(dists))
                target = (float(weeds[idx_min, 0]), float(weeds[idx_min, 1]))
                self.target = target
                self._plan_dubins_to_point(env, target)
            else:
                # 无杂草，随机选点探索（用直线路径，只保留田地内的点）
                goal = self._sample_explore_goal(env)
                if goal is None:
                    return 0.0, 0.0
                self.target = goal
                self._plan_straight_to_point(env, goal)  # 关键修复：用直线而非Dubins

        # 3) 跟随当前路径（使用Dubins的theta信息进行平滑跟踪）
        if not self.path:
            return 0.0, 0.0

        x, y = self._pose(env)
        tx, ty = self.path[self.idx]
        if math.hypot(tx - x, ty - y) <= self.reach_eps:
            self.idx += 1
            if self.idx >= len(self.path):
                # 路径完成，清空状态让下一tick重新评估
                self.path = []
                self.path_theta = []
                self.idx = 0
                return 0.0, 0.0
            tx, ty = self.path[self.idx]

        # 关键修复：使用theta进行跟踪，避免弦方向≠切线方向导致的"月牙形"轨迹
        if self.path_theta and self.idx < len(self.path_theta):
            return self._compute_action_with_theta(env, (tx, ty), self.path_theta[self.idx])
        else:
            # 兜底：无theta信息时使用旧方法
            return self._compute_action_to_point(env, (tx, ty))


# ---------- 工厂 ----------

def build_rule_policy(name: str):
    name = name.upper()
    if name == "BCP":
        ctrl = BCPController("BCP")
    elif name == "SNAKE":
        ctrl = SnakeController()
    elif name == "R_SNAKE":
        ctrl = RSnakeController()
    elif name == "JUMP":
        ctrl = JumpController()
    elif name == "REACT":
        ctrl = ReactController("REACT")
    else:
        raise ValueError(f"Unknown rule method: {name}")

    def policy_fn(obs, env):
        # controller may need to init once
        if not getattr(ctrl, "_inited", False):
            ctrl.reset(env)
            ctrl._inited = True
        return ctrl.act(obs, env)

    return policy_fn

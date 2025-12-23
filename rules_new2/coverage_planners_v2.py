from __future__ import annotations

"""
Geometry-faithful coverage planners for CppEnv v2 (rules_new2).

This module intentionally keeps the core concepts minimal:

1. RectifiedFrame:  rectified (X, Y) coordinates aligned with the
   bounding-box long edge.
2. Pose / PathExecutor:  a path is just a list of (x, y, theta_deg)
   poses in world space; a single executor turns them into (v, w).
3. BCPPlannerV2:  boustrophedon coverage using evenly spaced strips.

The goal is to reproduce the baseline BCP behavior from the RAL 2022
paper and the legacy rules implementation, while keeping the code as
simple and transparent as possible.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import dubins

from envs_new.cpp_env_v2 import CppEnv
from envs_new.components.dynamics.environment_dynamics import cpu_fov_bool


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------


@dataclass
class RectifiedFrame:
    """Rectified pasture frame aligned with the bounding-box long edge.

    - X axis runs along the long edge of the minimum-area rectangle.
    - Y axis is perpendicular to X.
    - origin is chosen as the rectangle center; x/y extents are measured
      relative to this origin.
    """

    origin: np.ndarray  # world (x, y) of rectified origin
    ex: np.ndarray  # unit vector along +X
    ey: np.ndarray  # unit vector along +Y
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    B: float  # coverage width (agent_width)
    R: float  # turning radius for Dubins
    margin_x: float  # safety margin along X for pass generation / turns
    S_w: float  # FOV chord length (for JUMP/SNAKE spacing)

    @classmethod
    def from_env(cls, env: CppEnv) -> "RectifiedFrame":
        """Build rectified frame from environment bounding box and config."""
        bbox_list = env.env_state.get_static_info("bounding_box")
        if not bbox_list:
            raise RuntimeError("RectifiedFrame.from_env: bounding_box missing in env_state")

        box = np.asarray(bbox_list[0], dtype=np.float64).reshape(-1, 2)
        if box.shape != (4, 2):
            raise RuntimeError(f"RectifiedFrame.from_env: unexpected box shape={box.shape}")

        # Find the longest edge to define X direction.
        max_len = -1.0
        best_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
        for i in range(4):
            p1, p2 = box[i], box[(i + 1) % 4]
            length = float(np.linalg.norm(p2 - p1))
            if length > max_len:
                max_len = length
                best_pair = (p1, p2)
        if best_pair is None or max_len <= 0.0:
            raise RuntimeError("RectifiedFrame.from_env: degenerate bounding_box")
        p1, p2 = best_pair

        ex = (p2 - p1) / max_len
        ey = np.array([-ex[1], ex[0]], dtype=np.float64)

        # Use rectangle center as origin for symmetric coordinates.
        center = box.mean(axis=0)
        rel = box - center
        x_coords = rel @ ex
        y_coords = rel @ ey
        x_min = float(x_coords.min())
        x_max = float(x_coords.max())
        y_min = float(y_coords.min())
        y_max = float(y_coords.max())

        cfg = env.config
        B = float(cfg.agent_width)

        v_max = float(cfg.v_max)
        w_max_rad = abs(float(cfg.w_max)) * math.pi / 180.0
        if w_max_rad <= 0.0:
            raise RuntimeError("RectifiedFrame.from_env: invalid w_max in config")
        R = v_max / w_max_rad

        vision_length = float(cfg.agent_vision_length)
        vision_angle_rad = math.radians(float(cfg.agent_vision_angle))
        S_w = 2.0 * vision_length * math.sin(vision_angle_rad / 2.0)

        # Use a conservative safety margin along X to keep turns away from edges.
        # Choose margin as one turning radius plus half vehicle width.
        margin_x = R + 0.5 * B

        return cls(
            origin=center,
            ex=ex,
            ey=ey,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            B=B,
            R=R,
            margin_x=margin_x,
            S_w=S_w,
        )

    # ---- coordinate and heading transforms ---------------------------------

    def world_to_rect(self, x: float, y: float) -> Tuple[float, float]:
        p = np.array([x, y], dtype=np.float64) - self.origin
        X = float(p @ self.ex)
        Y = float(p @ self.ey)
        return X, Y

    def rect_to_world(self, X: float, Y: float) -> Tuple[float, float]:
        p = self.origin + X * self.ex + Y * self.ey
        return float(p[0]), float(p[1])

    def world_heading_to_rect(self, direction_deg: float) -> float:
        theta = math.radians(direction_deg)
        heading_vec = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        X_comp = float(heading_vec @ self.ex)
        Y_comp = float(heading_vec @ self.ey)
        return math.atan2(Y_comp, X_comp)

    def rect_heading_to_world(self, theta_local: float) -> float:
        heading_vec = math.cos(theta_local) * self.ex + math.sin(theta_local) * self.ey
        theta_world = math.atan2(heading_vec[1], heading_vec[0])
        return math.degrees(theta_world) % 360.0


@dataclass
class Pose:
    """World-space pose with heading in degrees."""

    x: float
    y: float
    theta_deg: float


def _normalize_deg(angle_deg: float) -> float:
    """Normalize angle to (-180, 180]."""
    angle = (angle_deg + 180.0) % 360.0 - 180.0
    return angle


# ---------------------------------------------------------------------------
# Weed tracker (W set)
# ---------------------------------------------------------------------------


class WeedTracker:
    """Track weeds detected within FOV and not yet mowed (W set)."""

    def __init__(self, env: CppEnv, frame: RectifiedFrame) -> None:
        self.env = env
        self.frame = frame

        weed_map = env.maps_dict["weed"]
        ys, xs = np.nonzero(weed_map)
        self._all_indices = list(zip(ys.tolist(), xs.tolist()))
        if self._all_indices:
            world_pts = np.stack(
                [np.array([x + 0.5, y + 0.5], dtype=np.float64) for y, x in self._all_indices],
                axis=0,
            )
        else:
            world_pts = np.zeros((0, 2), dtype=np.float64)

        self._all_world = world_pts
        self._all_rect = np.zeros_like(world_pts)
        for i, (x, y) in enumerate(world_pts):
            X, Y = self.frame.world_to_rect(x, y)
            self._all_rect[i] = (X, Y)

        self._detected: List[int] = []

    def tick(self) -> None:
        """Update internal W 集合（已检测且未割的杂草）。

        设计上应在 *每个* planner 的 `act()` 中每步调用一次，从而保证：
        - 只要某个杂草点进入过 FOV，就会被加入 `_detected`；
        - 当该点在 `weed_map` 中被割掉（变为 0）时，会在 `get_active_weeds_rect`
          中被自动过滤掉。
        """
        env = self.env
        maps_dict = env.maps_dict
        agent = env.agent

        obstacle = maps_dict["obstacle"]
        fov = cpu_fov_bool(
            obstacle,
            agent.x,
            agent.y,
            agent.direction,
            agent.vision_length,
            agent.vision_angle,
            180,
        )
        weed_map = maps_dict["weed"]
        visible_mask = (weed_map == 1) & (fov.astype(bool))
        ys, xs = np.nonzero(visible_mask)
        visible_indices = set(zip(ys.tolist(), xs.tolist()))

        existing = set(self._detected)
        for idx, (y, x) in enumerate(self._all_indices):
            if (y, x) in visible_indices and idx not in existing:
                self._detected.append(idx)
                existing.add(idx)

    def get_active_weeds_rect(self) -> np.ndarray:
        # 调用方可能已经在当前步显式调用过 `tick()`；这里再次调用是幂等且安全的，
        # 代价只是一次额外的 FOV 计算，因此保持防御性更新以避免遗漏。
        self.tick()
        weed_map = self.env.maps_dict["weed"]
        active = []
        for idx in self._detected:
            y, x = self._all_indices[idx]
            if weed_map[y, x] == 1:
                active.append(self._all_rect[idx])
        if not active:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack(active, axis=0)


# ---------------------------------------------------------------------------
# Unified path executor
# ---------------------------------------------------------------------------


class PathExecutor:
    """Pure-pursuit style path tracking for a sequence of Pose."""

    def __init__(self, env: CppEnv, lookahead: float, reach_eps: float = 1.0) -> None:
        self.env = env
        self.lookahead = lookahead
        self.reach_eps = reach_eps
        self._path: List[Pose] = []
        self._idx: int = 0
        # Whether to "re-anchor" idx to the globally nearest waypoint each step.
        #
        # - For long, non-self-intersecting paths (e.g., pass strips), re-anchoring
        #   improves robustness against drift/overshoot of dense waypoints.
        # - For Dubins paths that may contain loops (e.g., in-place 180° turn),
        #   global re-anchoring is *incorrect* and can skip essential segments.
        self._allow_reanchor: bool = True

    def set_path(self, path: List[Pose], *, allow_reanchor: bool = True) -> None:
        self._path = list(path)
        self._idx = 0
        self._allow_reanchor = bool(allow_reanchor)

    @property
    def active(self) -> bool:
        return self._idx < len(self._path)

    def _advance_if_close(self, agent_x: float, agent_y: float) -> None:
        """Advance index while current waypoint is within reach_eps."""
        while self._idx + 1 < len(self._path):
            tx, ty = self._path[self._idx].x, self._path[self._idx].y
            if math.hypot(tx - agent_x, ty - agent_y) <= self.reach_eps:
                self._idx += 1
            else:
                break
        # If already at last point and距离很小，标记完成
        if self._idx == len(self._path) - 1:
            tx, ty = self._path[self._idx].x, self._path[self._idx].y
            if math.hypot(tx - agent_x, ty - agent_y) <= self.reach_eps:
                self._idx = len(self._path)

    def _select_lookahead_target(self, agent_x: float, agent_y: float) -> Optional[Pose]:
        """Find the first point ahead whose cumulative distance from current idx exceeds lookahead."""
        if self._idx >= len(self._path):
            return None

        # Start from current idx
        cumulative = 0.0
        prev_x, prev_y = agent_x, agent_y
        for j in range(self._idx, len(self._path)):
            cur = self._path[j]
            seg = math.hypot(cur.x - prev_x, cur.y - prev_y)
            cumulative += seg
            if cumulative >= self.lookahead:
                return cur
            prev_x, prev_y = cur.x, cur.y
        return self._path[-1]

    def step(self) -> Tuple[float, float]:
        """Return (v, w_deg) to follow the current path one step using pure pursuit."""
        if not self.active:
            return 0.0, 0.0

        agent = self.env.agent
        v_bounds, w_bounds = self.env.action_processor.get_action_bounds()
        v_min, v_max = float(v_bounds[0]), float(v_bounds[1])
        w_min, w_max = float(w_bounds[0]), float(w_bounds[1])

        if self._allow_reanchor:
            # Re-anchor idx to the nearest waypoint ahead (robust against drift).
            # NOTE: This is intentionally disabled for Dubins paths that may loop.
            distances = [math.hypot(p.x - agent.x, p.y - agent.y) for p in self._path[self._idx:]]
            if distances:
                nearest_offset = int(np.argmin(distances))  # type: ignore[attr-defined]
                self._idx += nearest_offset
                if self._idx >= len(self._path):
                    self._idx = len(self._path)
                    return 0.0, 0.0
        else:
            # Monotone local progress: never jump across segments (important for
            # Dubins loops). We still advance if the next waypoint is closer than
            # the current one, which handles overshoot without global re-anchoring.
            while self._idx + 1 < len(self._path):
                cur = self._path[self._idx]
                nxt = self._path[self._idx + 1]
                d_cur = math.hypot(cur.x - agent.x, cur.y - agent.y)
                d_nxt = math.hypot(nxt.x - agent.x, nxt.y - agent.y)
                if d_cur <= self.reach_eps or d_nxt + 1e-6 < d_cur:
                    self._idx += 1
                else:
                    break

        # If already at final point and very close, finish
        if self._idx == len(self._path) - 1:
            tx, ty = self._path[self._idx].x, self._path[self._idx].y
            # If we've passed the final point along path direction, finish.
            if len(self._path) >= 2:
                prev = self._path[self._idx - 1]
                path_dir = (tx - prev.x, ty - prev.y)
                to_agent = (agent.x - tx, agent.y - ty)
                if path_dir[0] * to_agent[0] + path_dir[1] * to_agent[1] > 0:
                    self._idx = len(self._path)
                    return 0.0, 0.0
            if math.hypot(tx - agent.x, ty - agent.y) <= self.reach_eps:
                self._idx = len(self._path)
                return 0.0, 0.0

        target = self._select_lookahead_target(agent.x, agent.y)
        if target is None:
            return 0.0, 0.0

        dx = target.x - agent.x
        dy = target.y - agent.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            self._advance_if_close(agent.x, agent.y)
            return 0.0, 0.0

        heading_rad = math.radians(agent.direction)
        alpha = math.atan2(dy, dx) - heading_rad
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        Ld = max(self.lookahead, 1e-3)
        kappa = 2.0 * math.sin(alpha) / Ld
        v = 0.8 * v_max
        w_rad = kappa * v
        w_deg = w_rad * 180.0 / math.pi

        # Clip to bounds
        w_deg = max(w_min, min(w_deg, w_max))
        v = max(v_min, min(v, v_max))

        # Safety: slow down when approaching the final waypoint
        if self._idx >= len(self._path) - 1:
            v = max(v_min, min(v, dist))

        v, w_deg = self.env.action_processor.clip_action(v, w_deg)
        return float(v), float(w_deg)


# ---------------------------------------------------------------------------
# Dubins helper
# ---------------------------------------------------------------------------


def plan_dubins_path(
    frame: RectifiedFrame,
    start: Pose,
    end: Pose,
    step_size: float,
) -> List[Pose]:
    """Plan a Dubins path between two world-space poses."""
    sX, sY = frame.world_to_rect(start.x, start.y)
    eX, eY = frame.world_to_rect(end.x, end.y)

    s_th = frame.world_heading_to_rect(start.theta_deg)
    e_th = frame.world_heading_to_rect(end.theta_deg)

    path = dubins.shortest_path((sX, sY, s_th), (eX, eY, e_th), frame.R)
    configs, _ = path.sample_many(step_size)

    poses: List[Pose] = []
    for X, Y, theta_local in configs[1:]:  # skip the start pose
        x_w, y_w = frame.rect_to_world(float(X), float(Y))
        theta_deg = frame.rect_heading_to_world(theta_local)
        poses.append(Pose(x_w, y_w, theta_deg))
    return poses


# ---------------------------------------------------------------------------
# BCP planner
# ---------------------------------------------------------------------------


class BCPPlannerV2:
    """Boustrophedon coverage with fixed spacing B in rectified frame."""

    def __init__(self, env: CppEnv) -> None:
        self.env = env
        self.frame = RectifiedFrame.from_env(env)
        self.executor = PathExecutor(env, lookahead=1.0 * self.frame.B)

        self._passes: List[List[Pose]] = []
        self._current_pass_idx: int = 0
        self._phase: str = "idle"  # "idle" | "pass" | "turn"
        self.total_path_length: float = 0.0

    # ---- public API --------------------------------------------------------

    def reset(self, place_agent: bool = True) -> None:
        """Construct all passes and (optionally) place agent at the first pass start.

        For two-stage environment resets (using env.reset(initial_position/...)),
        pass place_agent=False to avoid a large teleport that would pollute
        trajectory/mist maps.
        """
        self.total_path_length = 0.0
        self._build_passes()
        self._current_pass_idx = 0

        if not self._passes:
            self._phase = "idle"
            return

        first_pass = self._passes[0]
        start_pose = first_pass[0]
        if place_agent:
            # Place agent exactly on the first pass start.
            self.env.agent.reset((start_pose.x, start_pose.y), start_pose.theta_deg)

        self.executor.set_path(first_pass)
        self._phase = "pass"

    def act(self) -> Tuple[float, float]:
        """Return next action (v, w_deg) for the BCP planner."""
        if self._phase == "idle":
            return 0.0, 0.0

        # If we are currently executing a path (pass or turn), advance it.
        if self.executor.active:
            action = self.executor.step()
            self._update_path_length()
            return action

        # Current path finished: decide what to do next based on phase.
        if self._phase == "turn":
            # Finished a Dubins turn; start the next pass.
            next_idx = self._current_pass_idx
            if next_idx >= len(self._passes):
                self._phase = "idle"
                return 0.0, 0.0
            next_pass = self._passes[next_idx]
            self.executor.set_path(next_pass)
            self._phase = "pass"
            action = self.executor.step()
            self._update_path_length()
            return action

        if self._phase == "pass":
            # Finished a pass; plan Dubins to the start of the next pass.
            next_idx = self._current_pass_idx + 1
            if next_idx >= len(self._passes):
                self._phase = "idle"
                return 0.0, 0.0

            self._current_pass_idx = next_idx
            next_pass = self._passes[next_idx]
            next_start = next_pass[0]

            agent = self.env.agent
            start_pose = Pose(agent.x, agent.y, agent.direction)
            turn_path = plan_dubins_path(self.frame, start_pose, next_start, step_size=1.0)

            if turn_path:
                self.executor.set_path(turn_path, allow_reanchor=False)
                self._phase = "turn"
                action = self.executor.step()
                self._update_path_length()
                return action

            # Fallback: if Dubins path is empty (degenerate), start pass directly.
            self.executor.set_path(next_pass)
            self._phase = "pass"
            action = self.executor.step()
            self._update_path_length()
            return action

        # Unknown phase: be safe and stop.
        return 0.0, 0.0

    # ---- internal helpers --------------------------------------------------

    def _build_passes(self, step: float = 1.0) -> None:
        """Precompute all passes as Pose sequences."""
        f = self.frame
        safe_x_min = f.x_min + f.margin_x
        safe_x_max = f.x_max - f.margin_x
        length = safe_x_max - safe_x_min
        if length <= 0.0:
            xs = np.array([safe_x_min, safe_x_min + 1.0], dtype=np.float64)
        else:
            num = max(2, int(math.ceil(length / step)) + 1)
            xs = np.linspace(safe_x_min, safe_x_max, num=num, dtype=np.float64)

        y_bottom = f.y_min + f.B / 2.0
        y_top = f.y_max - f.B / 2.0

        ys: List[float] = []
        y = y_bottom
        while y <= y_top + 1e-6:
            ys.append(float(min(y, y_top)))
            y += f.B

        passes: List[List[Pose]] = []
        direction_sign = 1  # first pass along +X

        for idx, y_p in enumerate(ys):
            xs_dir = xs if direction_sign > 0 else xs[::-1]
            theta_local = 0.0 if direction_sign > 0 else math.pi
            theta_world = f.rect_heading_to_world(theta_local)

            path: List[Pose] = []
            for X in xs_dir:
                x_w, y_w = f.rect_to_world(float(X), y_p)
                path.append(Pose(x_w, y_w, theta_world))
            if path:
                passes.append(path)

            direction_sign *= -1  # alternate directions

        self._passes = passes

    def _update_path_length(self) -> None:
        """Accumulate path length based on env_state.agent_position."""
        pos_info = self.env.env_state.get_info("agent_position")
        if pos_info and len(pos_info) >= 2:
            last = np.array(pos_info.last, dtype=np.float64)
            cur = np.array(pos_info.current, dtype=np.float64)
            self.total_path_length += float(np.linalg.norm(cur - last))


# ---------------------------------------------------------------------------
# JUMP planner
# ---------------------------------------------------------------------------


@dataclass
class JumpPlan:
    weed_rect: np.ndarray  # (X_w, Y_w) in rect frame
    x_start: float         # rectified X of jump start (on current pass)
    x_end: float           # rectified X of jump end (back on current pass)
    i_start: int           # index on current pass closest to x_start
    i_end: int             # index on current pass closest to x_end


class JumpPlannerV2:
    """JUMP: BCP backbone with jump-return detours (L_jump = 4R)."""

    L_JUMP_FACTOR: float = 4.0

    def __init__(self, env: CppEnv) -> None:
        self.env = env
        self.frame = RectifiedFrame.from_env(env)
        self.executor = PathExecutor(env, lookahead=1.0 * self.frame.B)
        self.weeds = WeedTracker(env, self.frame)

        self.total_path_length: float = 0.0

        # pass geometry storage
        self._xs: np.ndarray = np.array([])
        self._y_p: float = 0.0
        self._direction_sign: int = 1  # +X or -X
        self._theta_p_world: float = 0.0
        self._pass_points: List[Pose] = []

        # jump state
        self._mode: str = "idle"  # pass | goto_start | jump_to | jump_back | turn | idle
        self._current_jump: Optional[JumpPlan] = None

    # ---- public API --------------------------------------------------------

    def reset(self, place_agent: bool = True) -> None:
        """Reset to first pass and (optionally) place agent at its start.

        When the environment is reset with initial_position/initial_direction,
        set place_agent=False to avoid a teleport jump.
        """
        self.total_path_length = 0.0
        self._build_xs()
        self._y_p = self.frame.y_min + self.frame.B / 2.0
        self._direction_sign = 1
        self._theta_p_world = self.frame.rect_heading_to_world(0.0)
        self._pass_points = self._build_pass(self._y_p, self._direction_sign)

        start_pose = self._pass_points[0]
        if place_agent:
            self.env.agent.reset((start_pose.x, start_pose.y), start_pose.theta_deg)
        self.executor.set_path(self._pass_points)
        self._mode = "pass"
        self._current_jump = None

    def act(self) -> Tuple[float, float]:
        # 无论当前处于何种子阶段，都应更新一次 W 集合，保证“曾进入 FOV
        # 的未割杂草”不会被遗漏。
        self.weeds.tick()
        if self._mode == "idle":
            return 0.0, 0.0

        # If we are in pass mode and currently following, we may preempt with a jump
        if self._mode == "pass" and self.executor.active:
            jp = self._find_jump_plan()
            if jp is not None:
                # segment along pass to jump start
                seg = self._pass_points[self.executor._idx : jp.i_start + 1]
                if len(seg) > 0:
                    self._current_jump = jp
                    self.executor.set_path(seg)
                    self._mode = "goto_start"

        # Execute active path if any
        if self.executor.active:
            action = self.executor.step()
            self._update_path_length()
            return action

        # Path finished -> advance state machine
        if self._mode == "goto_start":
            return self._start_jump_to()
        if self._mode == "jump_to":
            return self._start_jump_back()
        if self._mode == "jump_back":
            self._finish_jump_back()
            return self._step_or_switch_pass()
        if self._mode == "turn":
            # finished turn, start pass
            self.executor.set_path(self._pass_points)
            self._mode = "pass"
            action = self.executor.step()
            self._update_path_length()
            return action
        if self._mode == "pass":
            # pass ended naturally
            return self._switch_to_next_pass()

        return 0.0, 0.0

    # ---- internal helpers --------------------------------------------------

    def _build_xs(self, step: float = 1.0) -> None:
        f = self.frame
        safe_x_min = f.x_min + f.margin_x
        safe_x_max = f.x_max - f.margin_x
        length = safe_x_max - safe_x_min
        num = max(2, int(math.ceil(length / step)) + 1)
        self._xs = np.linspace(safe_x_min, safe_x_max, num=num, dtype=np.float64)

    def _build_pass(self, y_p: float, direction_sign: int) -> List[Pose]:
        f = self.frame
        xs_dir = self._xs if direction_sign > 0 else self._xs[::-1]
        theta_local = 0.0 if direction_sign > 0 else math.pi
        theta_world = f.rect_heading_to_world(theta_local)
        path: List[Pose] = []
        for X in xs_dir:
            x_w, y_w = f.rect_to_world(float(X), y_p)
            path.append(Pose(x_w, y_w, theta_world))
        return path

    def _spring_next_y(self, active_weeds_rect: np.ndarray) -> Optional[float]:
        f = self.frame
        c1 = self._y_p + f.S_w / 2.0
        c2 = float(np.min(active_weeds_rect[:, 1] + f.B / 2.0)) if active_weeds_rect.size > 0 else float("inf")
        c3 = f.y_max - f.B / 2.0
        y_next = min(c1, c2, c3)
        if y_next > c3 + 1e-6:
            return None
        return y_next

    def _update_path_length(self) -> None:
        pos_info = self.env.env_state.get_info("agent_position")
        if pos_info and len(pos_info) >= 2:
            last = np.array(pos_info.last, dtype=np.float64)
            cur = np.array(pos_info.current, dtype=np.float64)
            self.total_path_length += float(np.linalg.norm(cur - last))

    # ---- jump planning -----------------------------------------------------

    def _find_jump_plan(self) -> Optional[JumpPlan]:
        f = self.frame
        if len(self._pass_points) == 0:
            return None
        active = self.weeds.get_active_weeds_rect()
        if active.size == 0:
            return None
        L_jump = self.L_JUMP_FACTOR * f.R

        # rect coords of pass points
        xs_rect = np.array([f.world_to_rect(p.x, p.y)[0] for p in self._pass_points], dtype=np.float64)
        agent = self.env.agent
        X_m, Y_m = f.world_to_rect(agent.x, agent.y)
        sgn = self._direction_sign

        best: Optional[JumpPlan] = None
        best_forward = float("inf")

        for weed in active:
            X_w, Y_w = float(weed[0]), float(weed[1])
            if Y_w <= self._y_p + f.B / 2.0:
                continue
            forward = sgn * (X_w - X_m)
            if forward <= 0:
                continue
            X_start = X_w - sgn * L_jump
            X_end = X_w + sgn * L_jump
            if not (f.x_min + f.margin_x <= X_start <= f.x_max - f.margin_x):
                continue
            if not (f.x_min + f.margin_x <= X_end <= f.x_max - f.margin_x):
                continue

            i_start = int(np.argmin(np.abs(xs_rect - X_start)))
            i_end = int(np.argmin(np.abs(xs_rect - X_end)))

            current_idx = max(self.executor._idx, 0)
            if i_start <= current_idx + 1:
                continue

            if forward < best_forward:
                best_forward = forward
                best = JumpPlan(
                    weed_rect=weed.copy(),
                    x_start=X_start,
                    x_end=X_end,
                    i_start=i_start,
                    i_end=i_end,
                )
        return best

    # ---- jump execution ----------------------------------------------------

    def _start_jump_to(self) -> Tuple[float, float]:
        assert self._current_jump is not None
        f = self.frame
        jp = self._current_jump
        X_w, Y_w = float(jp.weed_rect[0]), float(jp.weed_rect[1])
        wx, wy = f.rect_to_world(X_w, Y_w)

        theta_pass = self._theta_p_world
        # 以几何上的最晚起跳点 (x_start, y_p, theta_p) 作为 Dubins 起点，
        # 避免用当前 pose 的微小偏差导致弧段外探出界。
        sx, sy = f.rect_to_world(jp.x_start, self._y_p)
        start_pose = Pose(sx, sy, theta_pass)
        end_pose = Pose(wx, wy, theta_pass)

        path = plan_dubins_path(f, start_pose, end_pose, step_size=1.0)
        self.executor.set_path(path, allow_reanchor=False)
        self._mode = "jump_to"
        action = self.executor.step()
        self._update_path_length()
        return action

    def _start_jump_back(self) -> Tuple[float, float]:
        assert self._current_jump is not None
        f = self.frame
        jp = self._current_jump
        X_end, Y_end = jp.x_end, self._y_p
        ex, ey = f.rect_to_world(X_end, Y_end)

        theta_pass = self._theta_p_world
        # 同理，用 weed 的几何 pose 作为起点，保证闭合环对称且朝向一致。
        X_w, Y_w = float(jp.weed_rect[0]), float(jp.weed_rect[1])
        wx, wy = f.rect_to_world(X_w, Y_w)
        start_pose = Pose(wx, wy, theta_pass)
        end_pose = Pose(ex, ey, theta_pass)

        path = plan_dubins_path(f, start_pose, end_pose, step_size=1.0)
        self.executor.set_path(path, allow_reanchor=False)
        self._mode = "jump_back"
        action = self.executor.step()
        self._update_path_length()
        return action

    def _finish_jump_back(self) -> None:
        if self._current_jump is None:
            self._mode = "pass"
            return
        jp = self._current_jump
        # resume pass from i_end
        self.executor.set_path(self._pass_points)
        self.executor._idx = min(jp.i_end, len(self._pass_points) - 1)
        self._current_jump = None
        self._mode = "pass"

    # ---- pass / row switching ---------------------------------------------

    def _switch_to_next_pass(self) -> Tuple[float, float]:
        # JUMP：仅当“顶边且 W 为空”时终止；否则一律按 Spring 公式 eq.(1) 生成下一条 pass。
        active = self.weeds.get_active_weeds_rect()
        y_top = self.frame.y_max - self.frame.B / 2.0

        # 顶边且无剩余已检测杂草 → 终止
        if abs(self._y_p - y_top) <= 1e-6 and active.size == 0:
            self._mode = "idle"
            return 0.0, 0.0

        # 否则使用 Spring 规则选择下一条 y
        y_next = self._spring_next_y(active)
        if y_next is None:
            # 理论上不会发生；保险起见回退到顶边
            y_next = y_top

        self._y_p = y_next
        self._direction_sign *= -1
        self._theta_p_world = self.frame.rect_heading_to_world(0.0 if self._direction_sign > 0 else math.pi)
        next_pass = self._build_pass(self._y_p, self._direction_sign)

        self._pass_points = next_pass

        agent = self.env.agent
        start_pose = Pose(agent.x, agent.y, agent.direction)
        end_pose = next_pass[0]
        turn_path = plan_dubins_path(self.frame, start_pose, end_pose, step_size=1.0)

        if turn_path:
            self.executor.set_path(turn_path, allow_reanchor=False)
            self._mode = "turn"
        else:
            self.executor.set_path(next_pass)
            self._mode = "pass"
        action = self.executor.step()
        self._update_path_length()
        return action

    def _step_or_switch_pass(self) -> Tuple[float, float]:
        if self.executor.active:
            action = self.executor.step()
            self._update_path_length()
            return action
        return self._switch_to_next_pass()


# ---------------------------------------------------------------------------
# SNAKE planners (detour without return)
# ---------------------------------------------------------------------------


class SnakePlannerV2:
    """SNAKE：跳到杂草，不返回；剩余条带平移到杂草的 y 继续扫。"""

    L_DETOUR_FACTOR: float = 2.0  # 2R

    def __init__(self, env: CppEnv) -> None:
        self.env = env
        self.frame = RectifiedFrame.from_env(env)
        self.executor = PathExecutor(env, lookahead=1.0 * self.frame.B)
        self.weeds = WeedTracker(env, self.frame)

        self.total_path_length: float = 0.0
        self._xs: np.ndarray = np.array([])
        self._y_p: float = 0.0
        self._direction_sign: int = 1
        self._theta_p_world: float = 0.0
        self._pass_points: List[Pose] = []

        self._mode: str = "idle"  # pass | detour_to | shifted_pass | turn | idle
        self._current_target: Optional[Tuple[float, float]] = None  # (X_w, Y_w)

    def reset(self, place_agent: bool = True) -> None:
        self.total_path_length = 0.0
        self._build_xs()
        self._y_p = self.frame.y_min + self.frame.B / 2.0
        self._direction_sign = 1
        self._theta_p_world = self.frame.rect_heading_to_world(0.0)
        self._pass_points = self._build_pass(self._y_p, self._direction_sign)
        start_pose = self._pass_points[0]
        if place_agent:
            self.env.agent.reset((start_pose.x, start_pose.y), start_pose.theta_deg)
        self.executor.set_path(self._pass_points)
        self._mode = "pass"
        self._current_target = None

    def act(self) -> Tuple[float, float]:
        # 每步更新一次 W，使 detour / turn 阶段看到的杂草也会被计入。
        self.weeds.tick()
        if self._mode == "idle":
            return 0.0, 0.0

        # 在 pass/shifted_pass 上尝试触发 detour
        if self._mode in ("pass", "shifted_pass") and self.executor.active:
            tgt = self._find_detour_target()
            if tgt is not None:
                X_w, Y_w = tgt
                self._current_target = (X_w, Y_w)
                # 直接从当前 pose 起跳到 weed
                return self._start_detour_to()

        if self.executor.active:
            action = self.executor.step()
            self._update_path_length()
            return action

        # 路径执行完后的状态转移
        if self._mode == "detour_to":
            return self._start_shifted_pass()
        if self._mode == "shifted_pass":
            return self._switch_to_next_pass()
        if self._mode == "turn":
            self.executor.set_path(self._pass_points)
            self._mode = "pass"
            action = self.executor.step()
            self._update_path_length()
            return action
        if self._mode == "pass":
            return self._switch_to_next_pass()

        return 0.0, 0.0

    # ---- helpers ----------------------------------------------------------
    def _build_xs(self, step: float = 1.0) -> None:
        f = self.frame
        safe_x_min = f.x_min + f.margin_x
        safe_x_max = f.x_max - f.margin_x
        num = max(2, int(math.ceil((safe_x_max - safe_x_min) / step)) + 1)
        self._xs = np.linspace(safe_x_min, safe_x_max, num=num, dtype=np.float64)

    def _build_pass(self, y_p: float, direction_sign: int) -> List[Pose]:
        f = self.frame
        xs_dir = self._xs if direction_sign > 0 else self._xs[::-1]
        theta_local = 0.0 if direction_sign > 0 else math.pi
        theta_world = f.rect_heading_to_world(theta_local)
        path: List[Pose] = []
        for X in xs_dir:
            x_w, y_w = f.rect_to_world(float(X), y_p)
            path.append(Pose(x_w, y_w, theta_world))
        return path

    def _update_path_length(self) -> None:
        pos_info = self.env.env_state.get_info("agent_position")
        if pos_info and len(pos_info) >= 2:
            last = np.array(pos_info.last, dtype=np.float64)
            cur = np.array(pos_info.current, dtype=np.float64)
            self.total_path_length += float(np.linalg.norm(cur - last))

    def _find_detour_target(self) -> Optional[Tuple[float, float]]:
        f = self.frame
        active = self.weeds.get_active_weeds_rect()
        if active.size == 0 or len(self._pass_points) == 0:
            return None
        L_detour = self.L_DETOUR_FACTOR * f.R

        agent = self.env.agent
        ax, ay = agent.x, agent.y
        sgn = self._direction_sign

        best: Optional[Tuple[float, float]] = None
        best_forward = float("inf")
        for weed in active:
            X_w, Y_w = float(weed[0]), float(weed[1])
            # 前方：沿 pass 方向的投影 > 0
            dir_vec = self.frame.ex if sgn > 0 else -self.frame.ex
            wx, wy = f.rect_to_world(X_w, Y_w)
            forward_vec = np.array([wx - ax, wy - ay], dtype=float)
            forward = float(forward_vec @ dir_vec)
            if forward <= 0:
                continue
            # 欧氏距离 ≥ 2R
            dist = math.hypot(wx - ax, wy - ay)
            if dist < L_detour:
                continue
            if forward < best_forward:
                best_forward = forward
                best = (X_w, Y_w)
        return best

    # ---- detour execution -------------------------------------------------
    def _start_detour_to(self) -> Tuple[float, float]:
        assert self._current_target is not None
        f = self.frame
        X_w, Y_w = self._current_target
        wx, wy = f.rect_to_world(X_w, Y_w)
        theta = self._theta_p_world
        start_pose = Pose(self.env.agent.x, self.env.agent.y, self.env.agent.direction)
        end_pose = Pose(wx, wy, theta)
        path = plan_dubins_path(f, start_pose, end_pose, step_size=1.0)
        self.executor.set_path(path, allow_reanchor=False)
        self._mode = "detour_to"
        action = self.executor.step()
        self._update_path_length()
        return action

    def _start_shifted_pass(self) -> Tuple[float, float]:
        assert self._current_target is not None
        f = self.frame
        X_w, Y_w = self._current_target
        # 构造从 X_w 开始的剩余条带，Y = Y_w
        if self._direction_sign > 0:
            xs = self._xs[self._xs >= X_w]
        else:
            xs = self._xs[self._xs <= X_w][::-1]
        theta = self._theta_p_world
        shifted: List[Pose] = []
        for X in xs:
            x_w, y_w = f.rect_to_world(float(X), Y_w)
            shifted.append(Pose(x_w, y_w, theta))
        if not shifted:
            self._mode = "pass"
            return 0.0, 0.0
        self._y_p = Y_w
        self._pass_points = shifted
        self.executor.set_path(shifted)
        self._mode = "shifted_pass"
        action = self.executor.step()
        self._update_path_length()
        return action

    # ---- pass switching ---------------------------------------------------
    def _switch_to_next_pass(self) -> Tuple[float, float]:
        f = self.frame
        y_top = f.y_max - f.B / 2.0
        y_next = self._y_p + f.S_w / 2.0 + f.B / 2.0
        if y_next > y_top + 1e-6:
            # 已到顶边，看是否还有未割（已检测）杂草
            active = self.weeds.get_active_weeds_rect()
            if active.size == 0:
                self._mode = "idle"
                return 0.0, 0.0
            # 还有杂草：在顶边来回扫
            y_next = self._y_p  # 保持顶边
        else:
            y_next = min(y_next, y_top)
        self._y_p = y_next
        self._direction_sign *= -1
        self._theta_p_world = f.rect_heading_to_world(0.0 if self._direction_sign > 0 else math.pi)
        next_pass = self._build_pass(self._y_p, self._direction_sign)
        self._pass_points = next_pass

        agent = self.env.agent
        start_pose = Pose(agent.x, agent.y, agent.direction)
        end_pose = next_pass[0]
        turn_path = plan_dubins_path(f, start_pose, end_pose, step_size=1.0)
        if turn_path:
            self.executor.set_path(turn_path, allow_reanchor=False)
            self._mode = "turn"
        else:
            self.executor.set_path(next_pass)
            self._mode = "pass"
        action = self.executor.step()
        self._update_path_length()
        return action


class RestrictedSnakePlannerV2(SnakePlannerV2):
    """R-SNAKE：在 SNAKE 基础上增加 y 阈值 Y_w >= y_p - 1.5*S_w。"""

    def _find_detour_target(self) -> Optional[Tuple[float, float]]:
        f = self.frame
        active = self.weeds.get_active_weeds_rect()
        if active.size == 0 or len(self._pass_points) == 0:
            return None
        L_detour = self.L_DETOUR_FACTOR * f.R
        agent = self.env.agent
        ax, ay = agent.x, agent.y
        sgn = self._direction_sign
        y_thresh = self._y_p - 1.5 * f.S_w

        best: Optional[Tuple[float, float]] = None
        best_forward = float("inf")
        for weed in active:
            X_w, Y_w = float(weed[0]), float(weed[1])
            if Y_w < y_thresh:
                continue
            # 前方：沿 pass 方向
            dir_vec = self.frame.ex if sgn > 0 else -self.frame.ex
            wx, wy = f.rect_to_world(X_w, Y_w)
            forward_vec = np.array([wx - ax, wy - ay], dtype=float)
            forward = float(forward_vec @ dir_vec)
            if forward <= 0:
                continue
            # 欧氏距离 ≥ 2R
            dist = math.hypot(wx - ax, wy - ay)
            if dist < L_detour:
                continue
            if forward < best_forward:
                best_forward = forward
                best = (X_w, Y_w)
        return best

    def _switch_to_next_pass(self) -> Tuple[float, float]:
        """R-SNAKE：到达最上层 pass 后即终止（不要求 W 为空）。"""
        f = self.frame
        y_top = f.y_max - f.B / 2.0

        # 当前 pass 已在顶边，再往上就停止
        if abs(self._y_p - y_top) <= 1e-6:
            self._mode = "idle"
            return 0.0, 0.0

        # 否则沿 SNAKE 的固定间距向上推进一条 pass
        y_next = self._y_p + f.S_w / 2.0 + f.B / 2.0
        y_next = min(y_next, y_top)
        self._y_p = y_next
        self._direction_sign *= -1
        self._theta_p_world = f.rect_heading_to_world(0.0 if self._direction_sign > 0 else math.pi)
        next_pass = self._build_pass(self._y_p, self._direction_sign)
        self._pass_points = next_pass

        agent = self.env.agent
        start_pose = Pose(agent.x, agent.y, agent.direction)
        end_pose = next_pass[0]
        turn_path = plan_dubins_path(f, start_pose, end_pose, step_size=1.0)
        if turn_path:
            self.executor.set_path(turn_path, allow_reanchor=False)
            self._mode = "turn"
        else:
            self.executor.set_path(next_pass)
            self._mode = "pass"
        action = self.executor.step()
        self._update_path_length()
        return action


# ---------------------------------------------------------------------------
# REACT planner (random-search reactive baseline)
# ---------------------------------------------------------------------------


class ReactPlannerV2:
    """REACT：随机探索 + FIFO 访问已检测杂草。

    与论文描述一致：
    - W 为空时：在 rectified bounding box 中均匀随机一个航点，Dubins 导航过去；
    - W 非空时：中断随机探索，按“检测到的先后顺序（FIFO）”依次访问杂草；
    - 在访问杂草期间仍继续检测新的杂草，并追加到队列尾部；
    - 当队列清空后恢复随机探索；
    - 终止：路径长度达到 BCP 上界的若干倍（默认 5×，便于实验）。
    """

    def __init__(self, env: CppEnv, length_limit_factor: float = 5.0) -> None:
        self.env = env
        self.frame = RectifiedFrame.from_env(env)
        self.executor = PathExecutor(env, lookahead=1.0 * self.frame.B)
        self.weeds = WeedTracker(env, self.frame)

        self.length_limit_factor = float(length_limit_factor)
        self.total_path_length: float = 0.0

        self._mode: str = "idle"  # search | serve | idle

        # FIFO queue stores indices into WeedTracker._all_indices/_all_rect.
        self._queue: List[int] = []
        self._queued_set: set[int] = set()

        self._bcp_upper_bound: float = 0.0

    # ---- public API --------------------------------------------------------

    def reset(self, place_agent: bool = True) -> None:
        """Reset internal state. Optionally place agent at standard start pose."""
        self.total_path_length = 0.0
        self._queue.clear()
        self._queued_set.clear()

        self._bcp_upper_bound = self._estimate_bcp_upper_bound()
        self._mode = "search"

        if place_agent:
            start_pose = self._standard_start_pose()
            self.env.agent.reset((start_pose.x, start_pose.y), start_pose.theta_deg)

        # Start with an initial random search waypoint.
        self._start_random_search()

    def act(self) -> Tuple[float, float]:
        if self._mode == "idle":
            return 0.0, 0.0

        # 每一步都更新 W，并同步 FIFO 队列。
        self.weeds.tick()
        self._sync_fifo_queue()

        # 终止条件：路径长度达到上界。
        if self.total_path_length >= self.length_limit_factor * self._bcp_upper_bound:
            self._mode = "idle"
            return 0.0, 0.0

        # 若在随机搜索中检测到杂草，立即中断搜索。
        if self._mode == "search" and self._queue and self.executor.active:
            self.executor.set_path([])

        # 继续执行当前路径。
        if self.executor.active:
            action = self.executor.step()
            self._update_path_length()
            return action

        # 当前路径走完：根据 W 是否为空选择下一段。
        if self._queue:
            return self._start_serve_next()
        return self._start_random_search()

    # ---- internal helpers --------------------------------------------------

    def _standard_start_pose(self) -> Pose:
        """Same canonical start pose as other planners for fair comparison."""
        f = self.frame
        safe_x_min = f.x_min + f.margin_x
        first_y_p = f.y_min + f.B / 2.0
        wx0, wy0 = f.rect_to_world(safe_x_min, first_y_p)
        theta0 = f.rect_heading_to_world(0.0)
        return Pose(wx0, wy0, theta0)

    def _estimate_bcp_upper_bound(self) -> float:
        """Simple geometric upper bound of BCP path length in rectified frame."""
        f = self.frame
        safe_x_min = f.x_min + f.margin_x
        safe_x_max = f.x_max - f.margin_x
        y_bottom = f.y_min + f.B / 2.0
        y_top = f.y_max - f.B / 2.0

        length_per_pass = max(0.0, safe_x_max - safe_x_min)
        num_passes = int(math.floor((y_top - y_bottom) / f.B)) + 1
        num_passes = max(1, num_passes)
        return length_per_pass * float(num_passes)

    def _sample_random_goal_rect(self) -> Tuple[float, float]:
        """Uniform random waypoint inside safe rectified bounding box."""
        f = self.frame
        safe_x_min = f.x_min + f.margin_x
        safe_x_max = f.x_max - f.margin_x
        safe_y_min = f.y_min + f.B / 2.0
        safe_y_max = f.y_max - f.B / 2.0

        rng = self.env.np_random
        X = float(rng.uniform(safe_x_min, safe_x_max))
        Y = float(rng.uniform(safe_y_min, safe_y_max))
        return X, Y

    def _sync_fifo_queue(self) -> None:
        """Append newly detected weeds to FIFO queue, prune already mowed ones."""
        weed_map = self.env.maps_dict["weed"]

        # Append newly detected weeds in detection order.
        for idx in self.weeds._detected:
            if idx in self._queued_set:
                continue
            y, x = self.weeds._all_indices[idx]
            if weed_map[y, x] == 1:
                self._queue.append(idx)
                self._queued_set.add(idx)

        # Drop mowed weeds from the front.
        while self._queue:
            head = self._queue[0]
            y, x = self.weeds._all_indices[head]
            if weed_map[y, x] == 1:
                break
            self._queue.pop(0)
            self._queued_set.discard(head)

    def _peek_next_weed_rect(self) -> Optional[Tuple[float, float]]:
        weed_map = self.env.maps_dict["weed"]
        while self._queue:
            idx = self._queue[0]
            y, x = self.weeds._all_indices[idx]
            if weed_map[y, x] == 1:
                X, Y = self.weeds._all_rect[idx]
                return float(X), float(Y)
            self._queue.pop(0)
            self._queued_set.discard(idx)
        return None

    def _start_random_search(self) -> Tuple[float, float]:
        """Plan a Dubins path to a new random waypoint."""
        f = self.frame
        X_g, Y_g = self._sample_random_goal_rect()
        gx, gy = f.rect_to_world(X_g, Y_g)

        agent = self.env.agent
        theta_goal = math.degrees(math.atan2(gy - agent.y, gx - agent.x)) % 360.0

        start_pose = Pose(agent.x, agent.y, agent.direction)
        end_pose = Pose(gx, gy, theta_goal)

        path = plan_dubins_path(f, start_pose, end_pose, step_size=1.0)
        self.executor.set_path(path if path else [end_pose], allow_reanchor=False)
        self._mode = "search"

        action = self.executor.step()
        self._update_path_length()
        return action

    def _start_serve_next(self) -> Tuple[float, float]:
        """Plan a Dubins path to the next FIFO weed."""
        f = self.frame
        tgt = self._peek_next_weed_rect()
        if tgt is None:
            return self._start_random_search()

        X_w, Y_w = tgt
        wx, wy = f.rect_to_world(X_w, Y_w)

        agent = self.env.agent
        theta_goal = math.degrees(math.atan2(wy - agent.y, wx - agent.x)) % 360.0

        start_pose = Pose(agent.x, agent.y, agent.direction)
        end_pose = Pose(wx, wy, theta_goal)

        path = plan_dubins_path(f, start_pose, end_pose, step_size=1.0)
        self.executor.set_path(path if path else [end_pose], allow_reanchor=False)
        self._mode = "serve"

        action = self.executor.step()
        self._update_path_length()
        return action

    def _update_path_length(self) -> None:
        pos_info = self.env.env_state.get_info("agent_position")
        if pos_info and len(pos_info) >= 2:
            last = np.array(pos_info.last, dtype=np.float64)
            cur = np.array(pos_info.current, dtype=np.float64)
            self.total_path_length += float(np.linalg.norm(cur - last))

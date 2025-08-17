"""Environment state management for the mowing robot.
Provides encapsulated state tracking with dynamic state variable support.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, TypeVar, Generic, Optional, Any


T = TypeVar('T')
class StateVariable(Generic[T]):
    """可配置历史追踪的状态变量"""

    def __init__(self, name: str, history_length: int = 2, initial_value: T = None):
        self._name = name
        self._history: deque[T] = deque(maxlen=history_length)
        if initial_value is not None:
            self._history.append(initial_value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def current(self) -> Optional[T]:
        return self._history[-1] if self._history else None

    @property
    def last(self) -> Optional[T]:
        return self._history[-2] if len(self._history) > 1 else None

    @property
    def history(self) -> deque[T]:
        return self._history

    def change(self, steps_back: int = 1) -> Any:
        """
        计算从过去到当前值的变化量。
        
        支持数值类型（返回 current - past）和元组类型（返回逐元素差值）。
        非数值类型返回 None。
        """
        if len(self._history) <= steps_back:
            return 0

        current_val = self.current
        past_val = self._history[-(steps_back + 1)]

        # Handle numeric types (int, float)
        if isinstance(current_val, (int, float)) and isinstance(past_val, (int, float)):
            return current_val - past_val

        # Handle tuple types with numeric elements
        if isinstance(current_val, tuple) and isinstance(past_val, tuple):
            if all(isinstance(x, (int, float)) for x in current_val + past_val):
                return tuple(c - p for c, p in zip(current_val, past_val))

        return None

    def update(self, value: T) -> None:
        self._history.append(value)

    def reset(self, initial_value: T = None) -> None:
        self._history.clear()
        if initial_value is not None:
            self._history.append(initial_value)

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"StateVariable(name='{self._name}', current={self.current})"

class EnvironmentState:
    """环境状态容器，支持动态变量管理"""
    
    def __init__(self):
        self._state_infos: Dict[str, StateVariable] = {}
        self._static_info: Dict[str, Any] = {}  # 非序列静态信息
        
        # 核心静态属性，支持直接访问以保持向后兼容
        self.dimensions: Tuple[int, int] = (0, 0)
        self.total_weed_count: int = 0
        self.total_frontier_area: int = 0
        self.max_steps: int = 300000
    
    def set_static_info(self, key: str, value: Any) -> None:
        self._static_info[key] = value
        
        # 为核心属性设置实例属性以便直接访问
        if key in ['dimensions', 'total_weed_count', 'total_frontier_area', 'max_steps']:
            setattr(self, key, value)
    
    def get_static_info(self, key: str, default: Any = None) -> Any:
        return self._static_info.get(key, default)
    
    def add_state_info(self, name: str, history_length: int = 2, initial_value: Any = None) -> StateVariable:
        state_var = StateVariable(name, history_length, initial_value)
        self._state_infos[name] = state_var
        return state_var
    
    def update_state(self, **updates) -> None:
        for key, value in updates.items():
            if key in self._state_infos:
                self._state_infos[key].update(value)
    
    def get_info(self, name: str) -> Optional[StateVariable]:
        """ state_info是序列量，static_info是静态量 """
        return self._state_infos.get(name)
    
    def __getattr__(self, name: str) -> Any:
        """统一访问接口：优先访问序列状态，然后访问静态信息"""
        if name in self._state_infos:
            return self._state_infos[name].current
        elif name in self._static_info:
            return self._static_info[name]
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def reset(self, dimensions: Tuple[int, int], total_weed_count: int, max_steps: int = 3000) -> None:
        """重置环境状态"""
        self.set_static_info('dimensions', dimensions)
        self.set_static_info('total_weed_count', total_weed_count)
        self.set_static_info('max_steps', max_steps)
    
    @property
    def is_done(self) -> bool:
        crashed = self._state_infos.get('crashed', StateVariable('crashed', 1, False)).current
        finished = self._state_infos.get('finished', StateVariable('finished', 1, False)).current
        timeout = self._state_infos.get('timeout', StateVariable('timeout', 1, False)).current
        return bool(crashed or finished or timeout)
    
    @property
    def weed_completion_ratio(self) -> float: #TODO: 这个功能要考虑有没有这个组件
        """杂草清除完成率：0.0（未清除）到 1.0（全部清除）"""
        if self.total_weed_count == 0:
            return 1.0
        current_weed_count = self._state_infos.get('weed_count', StateVariable('weed_count', 1, 0)).current or 0
        return 1.0 - (current_weed_count / self.total_weed_count)


class StateTracker:
    """长期状态历史追踪器，用于分析和可视化。"""

    def __init__(self, long_history_length: int = 1000):
        self.long_history_length = long_history_length
        self.long_term_vars: Dict[str, StateVariable] = {}
        self._is_setup = False

    def setup_long_term_tracking(self, env_state: EnvironmentState, tracked_vars: List[str] = None) -> None:
        """为指定状态变量初始化长期追踪。"""
        if tracked_vars is None:
            tracked_vars = ['frontier_area', 'weed_count', 'agent_position', 'trajectory_length', 'current_step']

        for var_name in tracked_vars:
            state_var = env_state.get_info(var_name)
            if state_var and state_var.current is not None:
                self.long_term_vars[var_name] = StateVariable(
                    f"long_term_{var_name}",
                    self.long_history_length,
                    state_var.current
                )

        self._is_setup = True

    def record_step(self, env_state: EnvironmentState) -> None:
        if not self._is_setup:
            self.setup_long_term_tracking(env_state)

        for name, long_var in self.long_term_vars.items():
            state_var = env_state.get_info(name)
            if state_var and state_var.current is not None:
                long_var.update(state_var.current)

    def get_recent_history(self, var_name: str, n: int = 10) -> List[Any]:
        if var_name in self.long_term_vars:
            history = list(self.long_term_vars[var_name].history)
            return history[-n:] if len(history) >= n else history
        return []

    def clear_history(self) -> None:
        self.long_term_vars.clear()
        self._is_setup = False
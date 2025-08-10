"""
状态验证器

验证系统状态的一致性和合法性
提供状态转换的监控和验证

作者：Rules_new优化团队
版本：2.0.0
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from .coordinate_system import CoordinateSystem as CS
from .exceptions import StateError

logger = logging.getLogger(__name__)


class StateValidator:
    """
    状态验证器
    
    验证：
    - 位置更新的合理性
    - 状态转换的一致性
    - 动作执行的有效性
    """
    
    def __init__(self, 
                 max_speed: float = 10.0,
                 max_angular_speed: float = 90.0,
                 position_tolerance: float = 0.1,
                 angle_tolerance: float = 1.0):
        """
        初始化状态验证器
        
        Args:
            max_speed: 最大线速度
            max_angular_speed: 最大角速度（度/秒）
            position_tolerance: 位置容差
            angle_tolerance: 角度容差（度）
        """
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        
        # 状态历史记录
        self.state_history = []
        self.max_history_size = 100
        
        # 验证统计
        self.validation_stats = {
            'total_validations': 0,
            'position_warnings': 0,
            'angle_warnings': 0,
            'speed_warnings': 0,
            'consistency_warnings': 0
        }
    
    def validate_position_update(self,
                                old_pos: Any,
                                new_pos: Any,
                                action: Optional[Tuple[float, float]] = None,
                                dt: float = 1.0) -> Dict[str, Any]:
        """
        验证位置更新的合理性
        
        Args:
            old_pos: 旧位置
            new_pos: 新位置
            action: 执行的动作 [distance, angle]
            dt: 时间间隔
            
        Returns:
            验证结果字典
        """
        self.validation_stats['total_validations'] += 1
        
        # 标准化坐标
        old_normalized = CS.normalize(old_pos)
        new_normalized = CS.normalize(new_pos)
        
        # 计算实际移动
        actual_distance = CS.distance(old_normalized, new_normalized)
        actual_speed = actual_distance / dt if dt > 0 else 0
        
        result = {
            'valid': True,
            'actual_distance': actual_distance,
            'actual_speed': actual_speed,
            'warnings': []
        }
        
        # 验证速度限制
        if actual_speed > self.max_speed:
            warning = f"速度超限: {actual_speed:.2f} > {self.max_speed}"
            result['warnings'].append(warning)
            result['valid'] = False
            self.validation_stats['speed_warnings'] += 1
            logger.warning(warning)
        
        # 如果提供了动作，验证动作与实际移动的一致性
        if action is not None:
            expected_distance = action[0]
            distance_error = abs(actual_distance - expected_distance)
            
            if distance_error > self.position_tolerance:
                warning = f"位置更新异常: 期望移动{expected_distance:.2f}, 实际{actual_distance:.2f}"
                result['warnings'].append(warning)
                result['distance_error'] = distance_error
                self.validation_stats['position_warnings'] += 1
                logger.warning(warning)
        
        # 记录状态
        self._record_state({
            'timestamp': datetime.now().isoformat(),
            'old_pos': old_normalized,
            'new_pos': new_normalized,
            'actual_distance': actual_distance,
            'valid': result['valid']
        })
        
        return result
    
    def validate_angle_update(self,
                             old_angle: float,
                             new_angle: float,
                             action_angle: Optional[float] = None,
                             dt: float = 1.0) -> Dict[str, Any]:
        """
        验证角度更新的合理性
        
        Args:
            old_angle: 旧角度（度）
            new_angle: 新角度（度）
            action_angle: 动作角度（度）
            dt: 时间间隔
            
        Returns:
            验证结果
        """
        # 计算角度变化
        angle_diff = new_angle - old_angle
        
        # 归一化到 [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        angular_speed = abs(angle_diff) / dt if dt > 0 else 0
        
        result = {
            'valid': True,
            'angle_diff': angle_diff,
            'angular_speed': angular_speed,
            'warnings': []
        }
        
        # 验证角速度限制
        if angular_speed > self.max_angular_speed:
            warning = f"角速度超限: {angular_speed:.2f} > {self.max_angular_speed}"
            result['warnings'].append(warning)
            result['valid'] = False
            self.validation_stats['angle_warnings'] += 1
            logger.warning(warning)
        
        # 验证与动作的一致性
        if action_angle is not None:
            angle_error = abs(angle_diff - action_angle)
            if angle_error > self.angle_tolerance:
                warning = f"角度更新异常: 期望{action_angle:.2f}, 实际{angle_diff:.2f}"
                result['warnings'].append(warning)
                result['angle_error'] = angle_error
                self.validation_stats['angle_warnings'] += 1
                logger.warning(warning)
        
        return result
    
    def validate_state_transition(self,
                                 old_state: Dict[str, Any],
                                 new_state: Dict[str, Any],
                                 action: Optional[Any] = None) -> Dict[str, Any]:
        """
        验证完整的状态转换
        
        Args:
            old_state: 旧状态
            new_state: 新状态
            action: 执行的动作
            
        Returns:
            验证结果
        """
        results = {
            'valid': True,
            'position_validation': None,
            'angle_validation': None,
            'consistency_check': None,
            'warnings': []
        }
        
        # 验证位置
        if 'agent_position' in old_state and 'agent_position' in new_state:
            pos_result = self.validate_position_update(
                old_state['agent_position'],
                new_state['agent_position'],
                action
            )
            results['position_validation'] = pos_result
            results['valid'] &= pos_result['valid']
            results['warnings'].extend(pos_result.get('warnings', []))
        
        # 验证角度
        if 'agent_direction' in old_state and 'agent_direction' in new_state:
            angle_action = action[1] if action and len(action) > 1 else None
            angle_result = self.validate_angle_update(
                old_state['agent_direction'],
                new_state['agent_direction'],
                angle_action
            )
            results['angle_validation'] = angle_result
            results['valid'] &= angle_result['valid']
            results['warnings'].extend(angle_result.get('warnings', []))
        
        # 一致性检查
        consistency = self._check_consistency(old_state, new_state)
        results['consistency_check'] = consistency
        if not consistency['consistent']:
            results['valid'] = False
            results['warnings'].extend(consistency.get('issues', []))
            self.validation_stats['consistency_warnings'] += 1
        
        return results
    
    def _check_consistency(self,
                          old_state: Dict[str, Any],
                          new_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查状态的内部一致性
        
        Args:
            old_state: 旧状态
            new_state: 新状态
            
        Returns:
            一致性检查结果
        """
        consistency = {
            'consistent': True,
            'issues': []
        }
        
        # 检查覆盖率是否单调递增
        if 'coverage_rate' in old_state and 'coverage_rate' in new_state:
            if new_state['coverage_rate'] < old_state['coverage_rate']:
                issue = f"覆盖率下降: {old_state['coverage_rate']:.3f} -> {new_state['coverage_rate']:.3f}"
                consistency['issues'].append(issue)
                consistency['consistent'] = False
        
        # 检查杂草数量是否单调递减
        if 'weed_count' in old_state and 'weed_count' in new_state:
            if new_state['weed_count'] > old_state['weed_count']:
                issue = f"杂草数量增加: {old_state['weed_count']} -> {new_state['weed_count']}"
                consistency['issues'].append(issue)
                consistency['consistent'] = False
        
        return consistency
    
    def _record_state(self, state_info: Dict[str, Any]):
        """记录状态到历史"""
        self.state_history.append(state_info)
        
        # 限制历史大小
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        return {
            'stats': self.validation_stats.copy(),
            'history_size': len(self.state_history),
            'recent_validations': self.state_history[-10:] if self.state_history else []
        }
    
    def reset(self):
        """重置验证器"""
        self.state_history.clear()
        self.validation_stats = {
            'total_validations': 0,
            'position_warnings': 0,
            'angle_warnings': 0,
            'speed_warnings': 0,
            'consistency_warnings': 0
        }
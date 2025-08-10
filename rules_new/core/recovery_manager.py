"""
错误恢复管理器

提供智能的错误恢复策略和自动恢复机制
支持断点续传、状态回滚、渐进式降级

作者：Rules_new优化团队
版本：2.0.0
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from collections import deque

from .exceptions import RecoverableError, RulesNewError
from .coordinate_system import CoordinateSystem as CS

logger = logging.getLogger(__name__)


class RecoveryManager:
    """
    错误恢复管理器
    
    功能：
    - 状态检查点管理
    - 自动恢复策略
    - 降级操作
    - 错误模式学习
    """
    
    def __init__(self, 
                 checkpoint_dir: Path = Path("./checkpoints"),
                 max_checkpoints: int = 10,
                 auto_recovery: bool = True):
        """
        初始化恢复管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大检查点数量
            auto_recovery: 是否启用自动恢复
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.auto_recovery = auto_recovery
        
        # 检查点队列
        self.checkpoints = deque(maxlen=max_checkpoints)
        
        # 恢复策略注册表
        self.recovery_strategies = {}
        self._register_default_strategies()
        
        # 错误历史记录
        self.error_history = []
        self.recovery_success_rate = {}
        
    def _register_default_strategies(self):
        """注册默认恢复策略"""
        # 坐标错误恢复
        self.register_strategy(
            'COORD_ERROR',
            self._recover_coordinate_error,
            priority=1
        )
        
        # 算法错误恢复
        self.register_strategy(
            'ALG_ERROR',
            self._recover_algorithm_error,
            priority=2
        )
        
        # 环境错误恢复
        self.register_strategy(
            'ENV_ERROR',
            self._recover_environment_error,
            priority=3
        )
        
        # 通用恢复策略
        self.register_strategy(
            'GENERIC',
            self._recover_generic_error,
            priority=99
        )
    
    def register_strategy(self, 
                         error_type: str,
                         recovery_func: Callable,
                         priority: int = 50):
        """
        注册恢复策略
        
        Args:
            error_type: 错误类型
            recovery_func: 恢复函数
            priority: 优先级（越小越高）
        """
        self.recovery_strategies[error_type] = {
            'func': recovery_func,
            'priority': priority
        }
    
    def save_checkpoint(self, 
                       state: Dict[str, Any],
                       checkpoint_name: Optional[str] = None) -> Path:
        """
        保存状态检查点
        
        Args:
            state: 要保存的状态
            checkpoint_name: 检查点名称
            
        Returns:
            检查点文件路径
        """
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}"
        
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        # 标准化坐标后保存
        normalized_state = self._normalize_state_coordinates(state)
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(normalized_state, f)
            
            # 添加到检查点队列
            self.checkpoints.append({
                'name': checkpoint_name,
                'path': checkpoint_file,
                'timestamp': datetime.now(),
                'state_summary': self._get_state_summary(state)
            })
            
            logger.info(f"保存检查点: {checkpoint_file}")
            return checkpoint_file
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            raise
    
    def load_checkpoint(self, 
                       checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        加载状态检查点
        
        Args:
            checkpoint_name: 检查点名称（None表示加载最新）
            
        Returns:
            恢复的状态
        """
        if checkpoint_name is None and self.checkpoints:
            # 加载最新检查点
            checkpoint_info = self.checkpoints[-1]
            checkpoint_file = checkpoint_info['path']
        else:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_file}")
        
        try:
            with open(checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"加载检查点: {checkpoint_file}")
            return state
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise
    
    def recover_from_error(self, 
                          error: Exception,
                          current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        从错误中恢复
        
        Args:
            error: 发生的错误
            current_state: 当前状态（可选）
            
        Returns:
            恢复后的状态
        """
        # 记录错误
        self._record_error(error)
        
        # 如果不是自动恢复模式，直接抛出
        if not self.auto_recovery:
            raise error
        
        # 确定错误类型
        error_type = self._classify_error(error)
        
        # 选择恢复策略
        strategy = self._select_recovery_strategy(error_type)
        
        if strategy:
            try:
                logger.info(f"尝试恢复策略: {error_type}")
                recovered_state = strategy['func'](error, current_state)
                
                # 记录恢复成功
                self._record_recovery_success(error_type)
                
                return recovered_state
                
            except Exception as recovery_error:
                logger.error(f"恢复失败: {recovery_error}")
                # 尝试下一个策略或回滚到检查点
                return self._fallback_recovery(error, current_state)
        else:
            # 没有合适的策略，尝试通用恢复
            return self._fallback_recovery(error, current_state)
    
    def _classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        if hasattr(error, 'error_code'):
            # 使用错误代码的前缀
            code_prefix = error.error_code.split('_')[0]
            return f"{code_prefix}_ERROR"
        elif 'coordinate' in str(error).lower():
            return 'COORD_ERROR'
        elif 'algorithm' in str(error).lower():
            return 'ALG_ERROR'
        elif 'environment' in str(error).lower():
            return 'ENV_ERROR'
        else:
            return 'GENERIC'
    
    def _select_recovery_strategy(self, error_type: str) -> Optional[Dict[str, Any]]:
        """选择恢复策略"""
        # 直接匹配
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
        
        # 尝试通用策略
        if 'GENERIC' in self.recovery_strategies:
            return self.recovery_strategies['GENERIC']
        
        return None
    
    def _recover_coordinate_error(self, 
                                 error: Exception,
                                 current_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """恢复坐标错误"""
        logger.info("执行坐标错误恢复")
        
        if current_state is None:
            # 从最近的检查点恢复
            return self.load_checkpoint()
        
        # 尝试修复坐标
        fixed_state = current_state.copy()
        
        # 标准化所有坐标
        if 'agent_position' in fixed_state:
            fixed_state['agent_position'] = CS.normalize(fixed_state['agent_position'])
        
        if 'discovered_weeds' in fixed_state:
            fixed_state['discovered_weeds'] = [
                CS.normalize(w) for w in fixed_state['discovered_weeds']
            ]
        
        return fixed_state
    
    def _recover_algorithm_error(self,
                                error: Exception,
                                current_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """恢复算法错误"""
        logger.info("执行算法错误恢复")
        
        # 检查是否有上下文信息
        if hasattr(error, 'context'):
            context = error.context
            phase = context.get('phase', 'unknown')
            
            if phase == 'planning':
                # 规划阶段错误，重置算法状态
                if current_state:
                    # 清除可能导致问题的状态
                    current_state.pop('goal_path', None)
                    current_state.pop('current_goal', None)
                    return current_state
            elif phase == 'execution':
                # 执行阶段错误，回滚到上一个安全状态
                return self.load_checkpoint()
        
        # 默认恢复：加载最近的检查点
        return self.load_checkpoint()
    
    def _recover_environment_error(self,
                                  error: Exception,
                                  current_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """恢复环境错误"""
        logger.info("执行环境错误恢复")
        
        # 环境错误通常需要重置环境
        # 这里返回一个标记，让调用者知道需要重置环境
        recovery_state = {
            'needs_env_reset': True,
            'last_checkpoint': None
        }
        
        # 尝试加载最近的有效状态
        if self.checkpoints:
            try:
                last_state = self.load_checkpoint()
                recovery_state['last_checkpoint'] = last_state
            except:
                pass
        
        return recovery_state
    
    def _recover_generic_error(self,
                              error: Exception,
                              current_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """通用错误恢复"""
        logger.info("执行通用错误恢复")
        
        # 尝试从最近的检查点恢复
        if self.checkpoints:
            try:
                return self.load_checkpoint()
            except:
                pass
        
        # 如果没有检查点，返回安全的默认状态
        return {
            'agent_position': [0, 0],
            'agent_direction': 0,
            'discovered_weeds': [],
            'coverage_rate': 0.0,
            'needs_reset': True
        }
    
    def _fallback_recovery(self,
                          error: Exception,
                          current_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """降级恢复（最后的恢复手段）"""
        logger.warning("执行降级恢复")
        
        # 1. 尝试从最近的检查点恢复
        if self.checkpoints:
            try:
                return self.load_checkpoint()
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")
        
        # 2. 如果有当前状态，尝试修复它
        if current_state:
            try:
                return self._normalize_state_coordinates(current_state)
            except:
                pass
        
        # 3. 返回最小安全状态
        return {
            'agent_position': [0, 0],
            'agent_direction': 0,
            'discovered_weeds': [],
            'coverage_rate': 0.0,
            'needs_full_reset': True
        }
    
    def _normalize_state_coordinates(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """标准化状态中的所有坐标"""
        normalized = state.copy()
        
        # 标准化各种坐标字段
        coord_fields = ['agent_position', 'current_position', 'goal_position']
        for field in coord_fields:
            if field in normalized:
                try:
                    normalized[field] = CS.normalize(normalized[field])
                except:
                    pass
        
        # 标准化坐标列表
        list_fields = ['discovered_weeds', 'path_points', 'goal_path']
        for field in list_fields:
            if field in normalized and normalized[field]:
                try:
                    normalized[field] = [CS.normalize(p) for p in normalized[field]]
                except:
                    pass
        
        return normalized
    
    def _get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """获取状态摘要"""
        summary = {}
        
        # 关键信息
        if 'agent_position' in state:
            summary['position'] = state['agent_position']
        if 'coverage_rate' in state:
            summary['coverage'] = state['coverage_rate']
        if 'iteration_count' in state:
            summary['iteration'] = state['iteration_count']
        
        return summary
    
    def _record_error(self, error: Exception):
        """记录错误"""
        self.error_history.append({
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'message': str(error),
            'error_code': getattr(error, 'error_code', 'UNKNOWN')
        })
        
        # 限制历史记录大小
        if len(self.error_history) > 100:
            self.error_history.pop(0)
    
    def _record_recovery_success(self, error_type: str):
        """记录恢复成功"""
        if error_type not in self.recovery_success_rate:
            self.recovery_success_rate[error_type] = {'success': 0, 'total': 0}
        
        self.recovery_success_rate[error_type]['success'] += 1
        self.recovery_success_rate[error_type]['total'] += 1
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """获取恢复统计信息"""
        stats = {
            'total_errors': len(self.error_history),
            'checkpoints_saved': len(self.checkpoints),
            'recovery_rates': {}
        }
        
        # 计算成功率
        for error_type, counts in self.recovery_success_rate.items():
            if counts['total'] > 0:
                rate = counts['success'] / counts['total']
                stats['recovery_rates'][error_type] = {
                    'success_rate': rate,
                    'total_attempts': counts['total']
                }
        
        # 最近的错误
        if self.error_history:
            stats['recent_errors'] = self.error_history[-5:]
        
        return stats
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """清理旧的检查点"""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pkl"),
            key=lambda p: p.stat().st_mtime
        )
        
        if len(checkpoint_files) > keep_last:
            for checkpoint_file in checkpoint_files[:-keep_last]:
                try:
                    checkpoint_file.unlink()
                    logger.info(f"删除旧检查点: {checkpoint_file}")
                except Exception as e:
                    logger.warning(f"删除检查点失败: {e}")
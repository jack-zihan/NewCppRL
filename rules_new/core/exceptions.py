"""
分层异常体系

提供结构化的错误处理和恢复机制
支持错误追踪、状态保存和智能恢复

作者：Rules_new优化团队
版本：2.0.0
"""

import traceback
import json
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class RulesNewError(Exception):
    """
    Rules_new基础异常类
    
    所有自定义异常的基类，提供：
    - 错误上下文保存
    - 错误恢复建议
    - 调试信息收集
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 recovery_hint: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码（用于分类）
            context: 错误上下文信息
            recovery_hint: 恢复建议
            original_error: 原始异常（如果是包装的）
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'UNKNOWN'
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.original_error = original_error
        self.timestamp = datetime.now().isoformat()
        
        # 收集调用栈
        self.traceback = traceback.format_stack()
        
        # 记录到日志
        self._log_error()
    
    def _log_error(self):
        """记录错误到日志"""
        logger.error(f"[{self.error_code}] {self.message}")
        if self.context:
            logger.error(f"Context: {json.dumps(self.context, indent=2, default=str)}")
        if self.recovery_hint:
            logger.info(f"Recovery hint: {self.recovery_hint}")
        if self.original_error:
            logger.error(f"Original error: {self.original_error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'recovery_hint': self.recovery_hint,
            'timestamp': self.timestamp,
            'traceback': self.traceback[-5:] if self.traceback else []  # 只保留最近5层
        }
    
    def save_checkpoint(self, checkpoint_dir: Path = Path("./checkpoints")):
        """
        保存错误检查点（用于调试和恢复）
        
        Args:
            checkpoint_dir: 检查点保存目录
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"error_{self.error_code}_{self.timestamp}.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info(f"Error checkpoint saved to {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")


class AlgorithmError(RulesNewError):
    """
    算法相关错误
    
    包括路径规划失败、参数错误、状态不一致等
    """
    
    def __init__(self, 
                 algorithm_name: str,
                 phase: str,
                 message: str,
                 **kwargs):
        """
        初始化算法错误
        
        Args:
            algorithm_name: 算法名称
            phase: 错误发生的阶段（init, reset, planning, execution）
            message: 错误消息
        """
        error_code = f"ALG_{algorithm_name}_{phase}".upper()
        
        # 添加算法特定的上下文
        context = kwargs.get('context', {})
        context.update({
            'algorithm': algorithm_name,
            'phase': phase
        })
        kwargs['context'] = context
        
        # 根据阶段提供恢复建议
        if not kwargs.get('recovery_hint'):
            kwargs['recovery_hint'] = self._get_recovery_hint(phase)
        
        super().__init__(message, error_code=error_code, **kwargs)
    
    @staticmethod
    def _get_recovery_hint(phase: str) -> str:
        """根据阶段获取恢复建议"""
        hints = {
            'init': "检查算法配置参数是否正确",
            'reset': "验证初始状态是否完整",
            'planning': "检查环境状态和算法约束",
            'execution': "验证动作是否在有效范围内"
        }
        return hints.get(phase, "检查算法状态和环境配置")


class ExperimentError(RulesNewError):
    """
    实验运行错误
    
    包括环境创建失败、数据收集错误等
    """
    
    def __init__(self,
                 experiment_name: str,
                 stage: str,
                 message: str,
                 **kwargs):
        """
        初始化实验错误
        
        Args:
            experiment_name: 实验名称
            stage: 错误阶段（setup, running, collection, cleanup）
            message: 错误消息
        """
        error_code = f"EXP_{stage}".upper()
        
        context = kwargs.get('context', {})
        context.update({
            'experiment': experiment_name,
            'stage': stage
        })
        kwargs['context'] = context
        
        super().__init__(message, error_code=error_code, **kwargs)


class CoordinateError(RulesNewError):
    """
    坐标系统错误
    
    包括坐标格式错误、转换失败、边界越界等
    """
    
    def __init__(self,
                 operation: str,
                 coordinate: Any,
                 message: str,
                 **kwargs):
        """
        初始化坐标错误
        
        Args:
            operation: 操作类型（normalize, convert, validate）
            coordinate: 问题坐标
            message: 错误消息
        """
        error_code = f"COORD_{operation}".upper()
        
        context = kwargs.get('context', {})
        context.update({
            'operation': operation,
            'coordinate': str(coordinate),
            'coordinate_type': type(coordinate).__name__
        })
        kwargs['context'] = context
        
        kwargs['recovery_hint'] = kwargs.get('recovery_hint', 
            "确保坐标是2维数值数组，格式为[y, x]")
        
        super().__init__(message, error_code=error_code, **kwargs)


class StateError(RulesNewError):
    """
    状态管理错误
    
    包括状态不一致、状态转换非法等
    """
    
    def __init__(self,
                 component: str,
                 expected_state: Any,
                 actual_state: Any,
                 message: str,
                 **kwargs):
        """
        初始化状态错误
        
        Args:
            component: 组件名称
            expected_state: 期望状态
            actual_state: 实际状态
            message: 错误消息
        """
        error_code = f"STATE_{component}".upper()
        
        context = kwargs.get('context', {})
        context.update({
            'component': component,
            'expected': str(expected_state),
            'actual': str(actual_state)
        })
        kwargs['context'] = context
        
        super().__init__(message, error_code=error_code, **kwargs)


class ConfigurationError(RulesNewError):
    """
    配置错误
    
    包括配置文件缺失、参数无效、配置冲突等
    """
    
    def __init__(self,
                 config_type: str,
                 config_path: Optional[str],
                 message: str,
                 **kwargs):
        """
        初始化配置错误
        
        Args:
            config_type: 配置类型（algorithm, experiment, environment）
            config_path: 配置文件路径
            message: 错误消息
        """
        error_code = f"CONFIG_{config_type}".upper()
        
        context = kwargs.get('context', {})
        context.update({
            'config_type': config_type,
            'config_path': config_path
        })
        kwargs['context'] = context
        
        kwargs['recovery_hint'] = kwargs.get('recovery_hint',
            f"检查配置文件 {config_path} 是否存在且格式正确")
        
        super().__init__(message, error_code=error_code, **kwargs)


class EnvironmentError(RulesNewError):
    """
    环境相关错误
    
    包括环境创建失败、动作执行失败、观测异常等
    """
    
    def __init__(self,
                 env_type: str,
                 operation: str,
                 message: str,
                 **kwargs):
        """
        初始化环境错误
        
        Args:
            env_type: 环境类型
            operation: 操作类型（create, reset, step, close）
            message: 错误消息
        """
        error_code = f"ENV_{operation}".upper()
        
        context = kwargs.get('context', {})
        context.update({
            'env_type': env_type,
            'operation': operation
        })
        kwargs['context'] = context
        
        super().__init__(message, error_code=error_code, **kwargs)


class RecoverableError(RulesNewError):
    """
    可恢复错误
    
    提供自动恢复机制
    """
    
    def __init__(self,
                 message: str,
                 recovery_action: Optional[callable] = None,
                 max_retries: int = 3,
                 **kwargs):
        """
        初始化可恢复错误
        
        Args:
            message: 错误消息
            recovery_action: 恢复动作（可调用对象）
            max_retries: 最大重试次数
        """
        self.recovery_action = recovery_action
        self.max_retries = max_retries
        self.retry_count = 0
        
        super().__init__(message, **kwargs)
    
    def attempt_recovery(self) -> bool:
        """
        尝试恢复
        
        Returns:
            是否恢复成功
        """
        if not self.recovery_action:
            return False
        
        if self.retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded")
            return False
        
        try:
            self.retry_count += 1
            logger.info(f"Attempting recovery (attempt {self.retry_count}/{self.max_retries})")
            self.recovery_action()
            logger.info("Recovery successful")
            return True
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False


# 错误处理装饰器
def handle_errors(error_class=RulesNewError, 
                  save_checkpoint=False,
                  reraise=True):
    """
    错误处理装饰器
    
    Args:
        error_class: 要捕获的错误类
        save_checkpoint: 是否保存检查点
        reraise: 是否重新抛出错误
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_class as e:
                if save_checkpoint:
                    e.save_checkpoint()
                
                # 如果是可恢复错误，尝试恢复
                if isinstance(e, RecoverableError):
                    if e.attempt_recovery():
                        # 重试原函数
                        return func(*args, **kwargs)
                
                if reraise:
                    raise
                else:
                    logger.error(f"Error handled: {e}")
                    return None
            except Exception as e:
                # 包装未预期的错误
                wrapped_error = error_class(
                    message=f"Unexpected error in {func.__name__}",
                    original_error=e,
                    context={'function': func.__name__, 'args': str(args)[:100]}
                )
                if save_checkpoint:
                    wrapped_error.save_checkpoint()
                if reraise:
                    raise wrapped_error
                else:
                    logger.error(f"Unexpected error handled: {e}")
                    return None
        
        return wrapper
    return decorator
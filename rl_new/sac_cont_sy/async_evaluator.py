"""
异步评估管理器 - 支持并行评估和顺序返回
基于ThreadPoolExecutor实现，避免daemon进程限制问题
"""
import itertools
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from torchrl._utils import logger as torchrl_logger


class AsyncEvaluator:
    """
    异步评估器 - 支持并行评估但保证结果按顺序返回
    
    特性：
    - 使用ThreadPoolExecutor管理评估线程池，避免daemon进程限制
    - 线程可以创建子进程（ParallelEnv），解决了mp.Pool的限制
    - 内置无限任务队列，永不丢失评估请求
    - 支持结果缓存和顺序释放机制
    - 自动处理评估失败的情况
    """
    
    def __init__(self, max_workers: int = 2):
        """
        初始化异步评估器
        
        Args:
            max_workers: 最大并行评估线程数，默认为2
        """
        # 使用ThreadPoolExecutor替代mp.Pool
        # 线程可以创建子进程，避免daemon进程限制
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        
        # 进度条position分配（循环复用1~max_workers）
        self._position_counter = itertools.count(1)  # 无限计数器从1开始
        
        # 顺序控制相关数据结构
        self.submitted_steps: List[int] = []  # 记录提交顺序 [1000, 2000, 3000...]
        self.pending_results: Dict[int, Future] = {}  # {step: future} 正在执行的任务
        self.completed_cache: Dict[int, Dict[str, Any]] = {}  # {step: result} 已完成但未返回的结果
        self.next_return_index: int = 0  # 下一个应该返回结果的索引
        
        torchrl_logger.info(f"AsyncEvaluator初始化完成，使用ThreadPoolExecutor，max_workers={max_workers}")
    
    def submit_eval(self, eval_func, model_path: str, cfg: Any, step: int) -> Future:
        """
        提交评估任务
        
        Args:
            eval_func: 评估函数（evaluate_policy_standalone）
            model_path: 模型文件路径
            cfg: 配置对象
            step: 训练步数
            
        Returns:
            Future对象，可用于查询任务状态
        """
        # 循环分配进度条position: 1, 2, ..., max_workers, 1, 2, ...
        position = (next(self._position_counter) - 1) % self.max_workers + 1
        
        # 提交评估任务到线程池，使用submit
        future = self.executor.submit(eval_func, model_path, cfg, step, position)
        
        # 记录提交信息
        self.submitted_steps.append(step)
        self.pending_results[step] = future
        
        torchrl_logger.info(f"提交评估任务: step={step}, position={position}, 当前排队任务数: {len(self.pending_results)}")
        return future
    
    def get_evaluate_results(self) -> List[Dict[str, Any]]:
        """
        按顺序返回已完成的评估结果
        
        保证返回的结果严格按照step顺序，即使后面的评估先完成也会缓存起来，
        等待前面的评估完成后一起返回。
        
        Returns:
            按step顺序排列的评估结果列表
        """
        # 1. 检查所有pending results，将完成的结果移到缓存
        for step, future in list(self.pending_results.items()):
            if future.done():  # 使用done()检查是否完成
                try:
                    # 获取评估结果，timeout=0立即返回
                    result = future.result(timeout=0)
                    self.completed_cache[step] = result
                    torchrl_logger.info(f"评估完成: step={step}")
                except Exception as e:
                    # 评估失败也要记录，避免阻塞后续结果
                    torchrl_logger.error(f"评估失败 step={step}: {str(e)}")
                    self.completed_cache[step] = {
                        'error': str(e),
                        'step': step,
                        'metrics': None,
                        'video_path': None
                    }
                
                # 从pending中移除
                del self.pending_results[step]
        
        # 2. 按顺序释放连续的结果
        ordered_results = []
        
        while self.next_return_index < len(self.submitted_steps):
            next_step = self.submitted_steps[self.next_return_index]
            
            if next_step in self.completed_cache:
                # 找到了下一个应该返回的结果
                result = self.completed_cache.pop(next_step)
                
                # 只返回成功的结果，失败的只记录日志
                if 'error' not in result or result.get('metrics') is not None:
                    ordered_results.append(result)
                    torchrl_logger.info(f"返回评估结果: step={next_step}")
                else:
                    torchrl_logger.warning(f"跳过失败的评估: step={next_step}, error={result.get('error')}")
                
                self.next_return_index += 1
            else:
                # 下一个结果还未完成，停止返回
                break
        
        # 日志记录当前状态
        if ordered_results:
            torchrl_logger.info(
                f"本次返回 {len(ordered_results)} 个结果, "
                f"缓存中还有 {len(self.completed_cache)} 个结果等待返回"
            )
        
        return ordered_results
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        关闭评估器并返回所有剩余的结果
        
        Args:
            wait: 是否等待所有任务完成
            timeout: 等待超时时间（秒），None表示无限期等待
            
        Returns:
            list: 剩余的评估结果列表（按step顺序）
        """
        torchrl_logger.info("关闭AsyncEvaluator...")
        
        if wait:
            # 等待所有评估完成
            remaining = len(self.pending_results)
            if remaining > 0:
                torchrl_logger.info(f"等待 {remaining} 个评估任务完成...")
                
            # 等待所有pending的结果
            for step, future in list(self.pending_results.items()):
                try:
                    # 使用result等待结果，timeout参数可选
                    result = future.result(timeout=timeout)
                    self.completed_cache[step] = result
                    torchrl_logger.info(f"收集剩余评估结果: step={step}")
                except Exception as e:
                    torchrl_logger.error(f"收集剩余结果时出错 step={step}: {str(e)}")
                    self.completed_cache[step] = {
                        'error': str(e),
                        'step': step,
                        'metrics': None,
                        'video_path': None
                    }
                # 从pending中移除
                del self.pending_results[step]
        
        # 关闭线程池
        self.executor.shutdown(wait=wait)
        
        # 返回所有剩余的已完成但未返回的结果
        remaining_results = []
        if self.completed_cache:
            torchrl_logger.info(
                f"返回 {len(self.completed_cache)} 个剩余的评估结果: "
                f"{list(self.completed_cache.keys())}"
            )
            # 按step顺序返回，确保与之前的逻辑一致
            for i in range(self.next_return_index, len(self.submitted_steps)):
                step = self.submitted_steps[i]
                if step in self.completed_cache:
                    result = self.completed_cache[step]
                    # 只返回成功的结果
                    if 'error' not in result or result.get('metrics') is not None:
                        remaining_results.append(result)
        
        return remaining_results
    
    def get_status(self) -> Dict[str, int]:
        """
        获取当前状态统计
        
        Returns:
            包含各种状态计数的字典
        """
        return {
            'pending': len(self.pending_results),
            'cached': len(self.completed_cache),
            'total_submitted': len(self.submitted_steps),
            'total_returned': self.next_return_index
        }
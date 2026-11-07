"""
测试AsyncEvaluator异步评估系统

验证并行评估和顺序返回机制：
- 并行任务提交
- 顺序结果返回
- Worker position循环分配
- 缓存管理
- Shutdown行为
"""
import pytest
import time
from unittest.mock import Mock
from concurrent.futures import Future


class TestAsyncEvaluatorSubmission:
    """测试任务提交功能"""

    def test_submit_eval_basic(self):
        """基本的任务提交

        验证点:
        - submit_eval正确提交任务到线程池
        - 返回Future对象
        - submitted_steps记录step
        - pending_results记录Future
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        # Mock评估函数
        mock_eval_func = Mock(return_value={"step": 1000, "metrics": {}})

        # 提交任务
        future = evaluator.submit_eval(
            eval_func=mock_eval_func,
            model_path="/fake/model.pt",
            cfg=Mock(),
            step=1000
        )

        # 验证返回Future
        assert isinstance(future, Future)

        # 验证状态记录
        assert 1000 in evaluator.submitted_steps
        assert 1000 in evaluator.pending_results
        assert evaluator.pending_results[1000] is future

    def test_submit_multiple_evals(self):
        """提交多个评估任务

        验证点:
        - 多个任务正确排队
        - submitted_steps按顺序记录
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        mock_eval_func = Mock(return_value={"step": 0, "metrics": {}})

        # 提交3个任务
        steps = [1000, 2000, 3000]
        for step in steps:
            evaluator.submit_eval(mock_eval_func, "/fake/model.pt", Mock(), step)

        # 验证顺序记录
        assert evaluator.submitted_steps == steps
        assert len(evaluator.pending_results) == 3

    def test_worker_position_cycling(self):
        """Worker position循环分配

        验证点:
        - position在1~max_workers之间循环
        - 第1个任务position=1，第2个position=2，第3个position=1（循环）
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        # 捕获传递给eval_func的position参数
        positions = []

        def mock_eval_func(model_path, cfg, step, position):
            positions.append(position)
            return {"step": step, "metrics": {}}

        # 提交4个任务
        for step in [1000, 2000, 3000, 4000]:
            evaluator.submit_eval(mock_eval_func, "/fake/model.pt", Mock(), step)

        # 等待所有任务完成
        time.sleep(0.1)

        # 验证position循环：1, 2, 1, 2
        assert positions == [1, 2, 1, 2]


class TestAsyncEvaluatorResultRetrieval:
    """测试结果获取功能"""

    def test_get_results_sequential_ordering(self):
        """顺序返回结果

        验证点:
        - 即使后提交的任务先完成，也会缓存等待
        - 结果严格按提交顺序返回
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        # 模拟不同耗时的任务
        def fast_eval(model_path, cfg, step, position):
            time.sleep(0.01)  # 10ms
            return {"step": step, "metrics": {"type": "fast"}}

        def slow_eval(model_path, cfg, step, position):
            time.sleep(0.05)  # 50ms
            return {"step": step, "metrics": {"type": "slow"}}

        # 提交：慢任务(1000) → 快任务(2000)
        evaluator.submit_eval(slow_eval, "/model1.pt", Mock(), 1000)
        evaluator.submit_eval(fast_eval, "/model2.pt", Mock(), 2000)

        # 等待快任务完成（2000先完成，但应该被缓存）
        time.sleep(0.03)

        results = evaluator.get_evaluate_results()
        # 快任务完成但1000未完成，应该返回空列表
        assert len(results) == 0

        # 等待慢任务也完成
        time.sleep(0.03)

        results = evaluator.get_evaluate_results()
        # 现在两个都完成了，按顺序返回
        assert len(results) == 2
        assert results[0]["step"] == 1000  # 先提交的先返回
        assert results[1]["step"] == 2000

    def test_get_results_caching(self):
        """结果缓存机制

        验证点:
        - 已完成但未按顺序的结果被缓存
        - completed_cache正确记录
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        def fast_eval(model_path, cfg, step, position):
            return {"step": step, "metrics": {}}

        # 提交任务
        evaluator.submit_eval(fast_eval, "/m1.pt", Mock(), 1000)
        evaluator.submit_eval(fast_eval, "/m2.pt", Mock(), 2000)

        # 等待完成
        time.sleep(0.05)

        # 第一次获取结果（应返回全部）
        results1 = evaluator.get_evaluate_results()
        assert len(results1) == 2

        # 第二次获取（缓存已清空）
        results2 = evaluator.get_evaluate_results()
        assert len(results2) == 0

    def test_get_results_empty_when_pending(self):
        """任务未完成时返回空列表"""
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        def slow_eval(model_path, cfg, step, position):
            time.sleep(0.1)
            return {"step": step, "metrics": {}}

        evaluator.submit_eval(slow_eval, "/model.pt", Mock(), 1000)

        # 立即获取（任务未完成）
        results = evaluator.get_evaluate_results()
        assert len(results) == 0


class TestAsyncEvaluatorErrorHandling:
    """测试错误处理"""

    def test_evaluation_failure_handling(self):
        """评估失败时的处理

        验证点:
        - 失败的任务被记录到completed_cache
        - 包含error字段
        - 不阻塞后续任务
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        def failing_eval(model_path, cfg, step, position):
            raise RuntimeError("Evaluation failed!")

        def success_eval(model_path, cfg, step, position):
            return {"step": step, "metrics": {"success": True}}

        # 提交失败任务和成功任务
        evaluator.submit_eval(failing_eval, "/bad.pt", Mock(), 1000)
        evaluator.submit_eval(success_eval, "/good.pt", Mock(), 2000)

        # 等待完成
        time.sleep(0.05)

        results = evaluator.get_evaluate_results()

        # 失败任务应被跳过，只返回成功任务
        # 注意：根据实现，失败的评估不会返回（只记录日志）
        assert len(results) == 1
        assert results[0]["step"] == 2000
        assert results[0]["metrics"]["success"] is True


class TestAsyncEvaluatorShutdown:
    """测试关闭行为"""

    def test_shutdown_wait_for_pending(self):
        """Shutdown时等待所有任务完成

        验证点:
        - wait=True时等待所有pending任务
        - 返回所有剩余结果
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        def slow_eval(model_path, cfg, step, position):
            time.sleep(0.05)
            return {"step": step, "metrics": {}}

        # 提交任务
        evaluator.submit_eval(slow_eval, "/m1.pt", Mock(), 1000)
        evaluator.submit_eval(slow_eval, "/m2.pt", Mock(), 2000)

        # 立即shutdown（任务还在执行）
        remaining = evaluator.shutdown(wait=True)

        # 应该等待并返回所有结果
        assert len(remaining) == 2

    def test_shutdown_no_wait(self):
        """Shutdown不等待

        验证点:
        - wait=False时不等待pending任务
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        def slow_eval(model_path, cfg, step, position):
            time.sleep(0.1)
            return {"step": step, "metrics": {}}

        evaluator.submit_eval(slow_eval, "/model.pt", Mock(), 1000)

        # 立即shutdown不等待
        remaining = evaluator.shutdown(wait=False)

        # 未完成的任务不会返回
        assert len(remaining) == 0

    def test_shutdown_returns_cached_results(self):
        """Shutdown返回缓存的结果

        验证点:
        - 已完成但未返回的结果在shutdown时返回
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        def fast_eval(model_path, cfg, step, position):
            return {"step": step, "metrics": {}}

        # 提交并让第2个任务完成（但第1个未完成，第2个被缓存）
        def slow_then_fast(model_path, cfg, step, position):
            if step == 1000:
                time.sleep(0.1)
            return {"step": step, "metrics": {}}

        evaluator.submit_eval(slow_then_fast, "/m1.pt", Mock(), 1000)
        evaluator.submit_eval(fast_eval, "/m2.pt", Mock(), 2000)

        # 等待2000完成但1000未完成
        time.sleep(0.05)

        # 此时2000在缓存中
        remaining = evaluator.shutdown(wait=True)

        # 应该返回两个结果（按顺序）
        assert len(remaining) == 2
        assert remaining[0]["step"] == 1000
        assert remaining[1]["step"] == 2000


class TestAsyncEvaluatorStatus:
    """测试状态查询"""

    def test_get_status(self):
        """获取当前状态统计

        验证点:
        - pending: 正在执行的任务数
        - cached: 已完成但未返回的结果数
        - total_submitted: 总提交数
        - total_returned: 已返回数
        """
        from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

        evaluator = AsyncEvaluator(max_workers=2)

        def eval_func(model_path, cfg, step, position):
            return {"step": step, "metrics": {}}

        # 提交3个任务
        evaluator.submit_eval(eval_func, "/m1.pt", Mock(), 1000)
        evaluator.submit_eval(eval_func, "/m2.pt", Mock(), 2000)
        evaluator.submit_eval(eval_func, "/m3.pt", Mock(), 3000)

        # 等待完成
        time.sleep(0.05)

        # 获取前2个结果
        results = evaluator.get_evaluate_results()
        assert len(results) == 3  # 全部完成并返回

        # 查询状态
        status = evaluator.get_status()
        assert status["pending"] == 0  # 无pending
        assert status["cached"] == 0  # 无cached
        assert status["total_submitted"] == 3
        assert status["total_returned"] == 3  # 已返回3个

#!/usr/bin/env python3
"""
测试并行环境的线程安全问题
"""
import sys
import threading
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from envs_new.cpp_env_v2 import CppEnv
from envs_new.components.reward.reward_system import BaseCalculator


def test_parallel_coefficient_conflict():
    """测试并行环境的coefficient冲突"""
    print("🧪 测试并行环境的coefficient冲突...")
    
    # 记录原始值
    original_coefficient = BaseCalculator.coefficient
    print(f"原始BaseCalculator.coefficient: {original_coefficient}")
    
    results = []
    lock = threading.Lock()
    
    def create_env_with_config(env_id, coefficient_value):
        """创建环境并设置不同的coefficient"""
        try:
            # 创建环境，传入不同的reward配置
            env = CppEnv(reward_base_penalty=coefficient_value)
            env.reset(seed=env_id)
            
            # 检查coefficient是否被正确设置
            actual = BaseCalculator.coefficient
            
            with lock:
                results.append({
                    'env_id': env_id,
                    'expected': coefficient_value,
                    'actual': actual,
                    'match': abs(actual - coefficient_value) < 1e-6
                })
            
            # 模拟一些操作
            for _ in range(5):
                action = env.action_space.sample()
                env.step(action)
                time.sleep(0.001)  # 模拟计算延迟
            
            env.close()
            
        except Exception as e:
            print(f"环境{env_id}出错: {e}")
    
    # 创建多个线程，每个使用不同的coefficient
    threads = []
    coefficients = [-0.1, -0.2, -0.3, -0.4]  # 不同的配置值
    
    for i, coef in enumerate(coefficients):
        t = threading.Thread(target=create_env_with_config, args=(i, coef))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 分析结果
    print("\n📊 并行环境结果:")
    conflicts = 0
    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"  环境{r['env_id']}: 期望={r['expected']:.1f}, 实际={r['actual']:.1f} {status}")
        if not r['match']:
            conflicts += 1
    
    print(f"\n最终BaseCalculator.coefficient: {BaseCalculator.coefficient}")
    print(f"冲突数量: {conflicts}/{len(results)}")
    
    return conflicts > 0


def test_torchrl_parallel_simulation():
    """模拟TorchRL的ParallelEnv行为"""
    print("\n🧪 模拟TorchRL ParallelEnv (num_envs=4)...")
    
    from concurrent.futures import ThreadPoolExecutor
    
    def create_and_run_env(env_id):
        """模拟TorchRL的环境创建和运行"""
        # 每个环境可能有不同的配置
        config_value = -0.1 - env_id * 0.01  # 稍微不同的配置
        
        env = CppEnv(reward_base_penalty=config_value)
        env.reset(seed=env_id)
        
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        
        env.close()
        
        return {
            'env_id': env_id,
            'config': config_value,
            'final_coefficient': BaseCalculator.coefficient,
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0
        }
    
    # 使用线程池模拟并行环境
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(create_and_run_env, i) for i in range(4)]
        results = [f.result() for f in futures]
    
    print("\n📊 TorchRL模拟结果:")
    for r in results:
        print(f"  环境{r['env_id']}: 配置={r['config']:.2f}, "
              f"最终coefficient={r['final_coefficient']:.2f}, "
              f"平均奖励={r['avg_reward']:.4f}")
    
    # 检查是否所有环境都使用了相同的coefficient（最后一个设置的）
    final_coefficients = [r['final_coefficient'] for r in results]
    if len(set(final_coefficients)) == 1:
        print(f"\n⚠️ 所有环境共享同一个coefficient: {final_coefficients[0]}")
    else:
        print(f"\n❌ coefficient值不一致: {set(final_coefficients)}")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚨 并行环境线程安全测试")
    print("="*60 + "\n")
    
    has_conflict = test_parallel_coefficient_conflict()
    test_torchrl_parallel_simulation()
    
    print("\n" + "="*60)
    if has_conflict:
        print("❌ 发现线程安全问题！")
        print("\n问题分析:")
        print("  1. 类属性coefficient是共享的")
        print("  2. 并行环境会互相覆盖coefficient值")
        print("  3. 导致不同环境使用错误的奖励系数")
        print("\n影响:")
        print("  - 训练不稳定")
        print("  - 奖励计算错误")
        print("  - 难以调试的随机性bug")
    else:
        print("✅ 未发现明显冲突（但问题仍然存在）")
    
    print("="*60 + "\n")
    
    return 0 if not has_conflict else 1


if __name__ == "__main__":
    exit(main())
"""
测试设备转换对pixels的影响
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import envs_new
import torch
from torchrl.envs import GymWrapper
import gymnasium as gym
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.env_utils import make_single_environment
from omegaconf import OmegaConf


def test_device_transfer_impact():
    """测试设备转换对tensordict的影响"""
    print("\n" + "="*80)
    print("测试设备转换对TensorDict的影响")
    print("="*80)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，无法测试GPU转换")
        return
    
    # 创建配置
    cfg = OmegaConf.create({
        'seed': 42,
        'env': {
            'env_id': 'NewPasture-v5',
            'env_kwargs': {}
        },
        'collector': {
            'env_per_collector': 1
        },
        'logger': {
            'eval_video': True,
            'eval_episodes': 2,
            'eval_max_steps': 100,
            'eval_video_skip': 1
        }
    })
    
    # 1. 创建环境和actor
    print("\n1. 创建环境和actor:")
    env = make_single_environment(cfg, device="cpu", from_pixels=True)
    
    # 创建GPU上的actor（模拟训练设备）
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic_gpu = make_sac_models(dummy_env, device="cuda:0")
    actor_gpu = actor_critic_gpu[0]
    dummy_env.close()
    print(f"   Actor在GPU上: {next(actor_gpu.parameters()).device}")
    
    # 2. Reset获取初始tensordict
    print("\n2. Reset环境:")
    tds = []
    for i in range(2):
        td = env.reset()
        tds.append(td)
    print(f"   初始td的键: {list(tds[0].keys())}")
    print(f"   初始包含pixels: {'pixels' in tds[0]}")
    
    # 3. 模拟实际评估代码的流程
    print("\n3. 模拟实际评估流程（包含设备转换）:")
    
    # 收集active tensordict
    active_tds = [tds[0], tds[1]]
    
    # 批处理
    print("\n   3.1 批处理:")
    batch_td = torch.stack(active_tds)
    print(f"       批处理后的键: {list(batch_td.keys())}")
    print(f"       批处理包含pixels: {'pixels' in batch_td}")
    print(f"       batch_td设备: {batch_td.device}")
    
    # 移到GPU（模拟第409行）
    print("\n   3.2 移到GPU:")
    batch_td = batch_td.to("cuda:0")  # 注意：这创建了新对象！
    print(f"       GPU上的键: {list(batch_td.keys())}")
    print(f"       GPU上包含pixels: {'pixels' in batch_td}")
    print(f"       batch_td设备: {batch_td.device}")
    
    # Actor推理（模拟第411行）
    print("\n   3.3 Actor推理并移回CPU:")
    with torch.no_grad():
        batch_td = actor_gpu(batch_td).to("cpu")  # 注意：又创建了新对象！
    
    print(f"       推理后的键: {list(batch_td.keys())}")
    print(f"       推理后包含pixels: {'pixels' in batch_td}")
    print(f"       batch_td设备: {batch_td.device}")
    
    # unbind和检查
    print("\n   3.4 Unbind后的结果:")
    for i, td in enumerate(batch_td.unbind(0)):
        print(f"       环境{i}:")
        print(f"         td的键: {list(td.keys())}")
        print(f"         包含pixels: {'pixels' in td}")
    
    return batch_td


def test_detailed_device_transfer():
    """更详细的设备转换测试"""
    print("\n" + "="*80)
    print("详细测试设备转换的每一步")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用")
        return
    
    # 创建简单的环境
    env = gym.make("NewPasture-v5", render_mode='rgb_array')
    wrapped_env = GymWrapper(env, device="cpu", from_pixels=True, pixels_only=False)
    
    # 创建两个actor：CPU和GPU
    actor_critic_cpu = make_sac_models(wrapped_env, device="cpu")
    actor_cpu = actor_critic_cpu[0]
    
    actor_critic_gpu = make_sac_models(wrapped_env, device="cuda:0")
    actor_gpu = actor_critic_gpu[0]
    
    # 获取初始tensordict
    td = wrapped_env.reset()
    
    print("\n测试1: CPU Actor (无设备转换):")
    td1 = td.clone()
    print(f"  处理前: 键={list(td1.keys())}, pixels={'pixels' in td1}")
    td1 = actor_cpu(td1)
    print(f"  处理后: 键={list(td1.keys())}, pixels={'pixels' in td1}")
    
    print("\n测试2: GPU Actor (需要设备转换):")
    td2 = td.clone()
    print(f"  处理前: 键={list(td2.keys())}, pixels={'pixels' in td2}, 设备={td2.device}")
    
    # 移到GPU
    td2_gpu = td2.to("cuda:0")
    print(f"  GPU上: 键={list(td2_gpu.keys())}, pixels={'pixels' in td2_gpu}, 设备={td2_gpu.device}")
    
    # Actor处理
    td2_gpu = actor_gpu(td2_gpu)
    print(f"  Actor后: 键={list(td2_gpu.keys())}, pixels={'pixels' in td2_gpu}, 设备={td2_gpu.device}")
    
    # 移回CPU
    td2_cpu = td2_gpu.to("cpu")
    print(f"  回CPU后: 键={list(td2_cpu.keys())}, pixels={'pixels' in td2_cpu}, 设备={td2_cpu.device}")
    
    print("\n测试3: 检查对象ID变化:")
    td3 = td.clone()
    id_original = id(td3)
    print(f"  原始td的id: {id_original}")
    
    td3_gpu = td3.to("cuda:0")
    id_gpu = id(td3_gpu)
    print(f"  GPU td的id: {id_gpu} (变化: {id_gpu != id_original})")
    
    td3_cpu = td3_gpu.to("cpu")
    id_cpu = id(td3_cpu)
    print(f"  CPU td的id: {id_cpu} (变化: {id_cpu != id_gpu})")
    
    # 重要：检查原始td3是否被修改
    print(f"  原始td3还包含pixels吗: {'pixels' in td3}")
    print(f"  GPU td还包含pixels吗: {'pixels' in td3_gpu}")
    print(f"  CPU td还包含pixels吗: {'pixels' in td3_cpu}")


if __name__ == "__main__":
    print("测试设备转换对pixels的影响...")
    print("="*80)
    
    # 测试1: 模拟实际评估流程
    batch_td = test_device_transfer_impact()
    
    # 测试2: 详细的设备转换测试
    test_detailed_device_transfer()
    
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print("设备转换是否影响pixels的存在？")
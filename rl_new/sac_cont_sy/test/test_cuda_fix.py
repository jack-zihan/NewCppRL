#!/usr/bin/env python
"""
测试CUDA设备修复
验证模型创建是否能在正确的设备上运行
"""
import sys
import torch
sys.path.append('/home/lzh/NewCppRL')

from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.env_utils import make_train_environment
from omegaconf import DictConfig, OmegaConf


def test_model_creation():
    """测试模型创建的设备处理"""
    print("=" * 60)
    print("测试SAC模型CUDA设备修复")
    print("=" * 60)
    
    # 创建简单配置
    cfg = OmegaConf.create({
        'env': {
            'env_id': 'NewPasture-v2',
            'env_kwargs': {}
        },
        'collector': {
            'env_per_collector': 1
        },
        'seed': 42
    })
    
    # 测试CPU设备
    print("\n1. 测试CPU设备创建...")
    env = make_train_environment(cfg, device="cpu")
    model = make_sac_models(env, device="cpu")
    
    # 验证模型在CPU上
    for name, param in model.named_parameters():
        assert not param.is_cuda, f"参数 {name} 应该在CPU上"
    print("✅ CPU设备测试通过")
    env.close()
    
    # 如果有CUDA，测试GPU设备
    if torch.cuda.is_available():
        print("\n2. 测试CUDA设备创建...")
        
        # 测试cuda:0
        env = make_train_environment(cfg, device="cpu")  # 环境保持在CPU
        model = make_sac_models(env, device="cuda:0")
        
        # 验证关键组件在正确设备上
        print("   检查模型参数设备...")
        for name, param in model.named_parameters():
            assert param.is_cuda, f"参数 {name} 应该在CUDA上"
            assert param.device.index == 0, f"参数 {name} 应该在cuda:0上"
        
        print("   检查action_spec设备...")
        # 通过运行一次前向传播来验证
        try:
            with torch.no_grad():
                td = env.fake_tensordict().to("cuda:0")
                output = model[0](td)  # Policy前向传播
                assert output["action"].is_cuda, "输出应该在CUDA上"
            print("✅ CUDA设备测试通过")
        except Exception as e:
            print(f"❌ CUDA测试失败: {e}")
            raise
        finally:
            env.close()
        
        # 如果有多个GPU，测试不同设备
        if torch.cuda.device_count() > 1:
            print("\n3. 测试多GPU设备...")
            env1 = make_train_environment(cfg, device="cpu")
            env2 = make_train_environment(cfg, device="cpu")
            
            model1 = make_sac_models(env1, device="cuda:0")
            model2 = make_sac_models(env2, device="cuda:1")
            
            # 验证在不同设备上
            for name, param in model1.named_parameters():
                assert param.device.index == 0, f"model1的{name}应该在cuda:0"
            
            for name, param in model2.named_parameters():
                assert param.device.index == 1, f"model2的{name}应该在cuda:1"
            
            print("✅ 多GPU设备测试通过")
            env1.close()
            env2.close()
    else:
        print("\n⚠️ 没有CUDA设备，跳过GPU测试")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！CUDA设备问题已修复")
    print("=" * 60)


if __name__ == "__main__":
    test_model_creation()
#!/usr/bin/env python3
"""
测试配置访问方式问题
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

from omegaconf import OmegaConf, DictConfig

def test_config_access():
    """测试不同的配置访问方式"""
    print("\n" + "="*80)
    print("测试配置访问方式")
    print("="*80)
    
    # 创建一个OmegaConf配置
    config = OmegaConf.create({
        'logger': {
            'backend': 'wandb',
            'eval_video': True,
            'eval_episodes': 4,
            'eval_max_steps': 10000,
            'eval_video_skip': 100,
            'show_progress': True
        }
    })
    
    print("\n1. 原始config类型:")
    print(f"   type(config): {type(config)}")
    print(f"   type(config.logger): {type(config.logger)}")
    
    # 模拟evaluate_policy中的代码
    print("\n2. 模拟evaluate_policy中的访问:")
    eval_cfg = config.logger
    print(f"   eval_cfg类型: {type(eval_cfg)}")
    
    # 测试点号访问
    print("\n3. 测试点号访问:")
    try:
        value = eval_cfg.eval_video
        print(f"   ✅ eval_cfg.eval_video = {value}")
    except AttributeError as e:
        print(f"   ❌ 点号访问失败: {e}")
    
    # 测试方括号访问
    print("\n4. 测试方括号访问:")
    try:
        value = eval_cfg['eval_video']
        print(f"   ✅ eval_cfg['eval_video'] = {value}")
    except KeyError as e:
        print(f"   ❌ 方括号访问失败: {e}")
    
    # 测试条件判断
    print("\n5. 测试条件判断 (第371行的逻辑):")
    logger = "fake_logger"  # 模拟logger不为None
    try:
        if eval_cfg.eval_video and logger is not None:
            print(f"   ✅ 条件通过，recorder会被创建")
        else:
            print(f"   ❌ 条件失败，recorder不会被创建")
    except Exception as e:
        print(f"   ❌ 条件判断出错: {e}")
    
    # 测试转换为字典后的访问
    print("\n6. 如果转换为普通字典:")
    eval_cfg_dict = dict(config.logger)
    print(f"   转换后类型: {type(eval_cfg_dict)}")
    
    try:
        value = eval_cfg_dict.eval_video
        print(f"   ✅ dict.eval_video = {value}")
    except AttributeError as e:
        print(f"   ❌ 字典不支持点号访问: {e}")
    
    try:
        value = eval_cfg_dict['eval_video']
        print(f"   ✅ dict['eval_video'] = {value}")
    except KeyError as e:
        print(f"   ❌ 方括号访问失败: {e}")
    
    # 测试可能的类型转换问题
    print("\n7. 测试OmegaConf的to_container:")
    container = OmegaConf.to_container(config.logger)
    print(f"   to_container后类型: {type(container)}")
    print(f"   是否有eval_video: {'eval_video' in container}")
    
    return eval_cfg


def test_real_config_loading():
    """测试实际的配置加载"""
    print("\n" + "="*80)
    print("测试实际配置加载")
    print("="*80)
    
    try:
        config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
        print(f"\n1. 加载的配置类型: {type(config)}")
        print(f"   config.logger类型: {type(config.logger)}")
        
        eval_cfg = config.logger
        print(f"\n2. eval_cfg类型: {type(eval_cfg)}")
        
        # 测试关键的访问
        print(f"\n3. 测试关键访问:")
        print(f"   eval_cfg.eval_video = {eval_cfg.eval_video}")
        print(f"   eval_cfg['eval_video'] = {eval_cfg['eval_video']}")
        
        # 测试条件
        if eval_cfg.eval_video:
            print(f"   ✅ eval_cfg.eval_video条件为True")
        else:
            print(f"   ❌ eval_cfg.eval_video条件为False")
            
    except Exception as e:
        print(f"   ❌ 加载配置失败: {e}")


if __name__ == "__main__":
    eval_cfg = test_config_access()
    test_real_config_loading()
    
    print("\n" + "="*80)
    print("诊断结果:")
    print("如果点号访问失败，第371行的条件判断会出错")
    print("这会导致recorder永远不会被创建，因此没有视频上传")
    print("="*80)
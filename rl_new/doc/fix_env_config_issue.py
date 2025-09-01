#!/usr/bin/env python3
"""
快速修复SAC训练脚本中的环境ID配置问题
运行此脚本将自动修复env_utils.py中的配置读取错误
"""

import os
from pathlib import Path

def fix_env_utils():
    """修复env_utils.py中的配置键名错误"""
    
    # 定位文件
    base_path = Path(__file__).parent.parent  # rl_new目录
    env_utils_path = base_path / "sac_cont_sy" / "env_utils.py"
    
    if not env_utils_path.exists():
        print(f"❌ 找不到文件: {env_utils_path}")
        return False
    
    # 读取文件内容
    with open(env_utils_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录原始内容用于备份
    original_content = content
    
    # 执行替换
    replacements = [
        # 第52行附近
        ('partial = functools.partial(make_env_lambda, env_id=cfg.env.name,',
         'partial = functools.partial(make_env_lambda, env_id=cfg.env.env_id,'),
        
        # 第58行附近  
        ('partial_eval = functools.partial(make_env_lambda, env_id=cfg.env.name,',
         'partial_eval = functools.partial(make_env_lambda, env_id=cfg.env.env_id,'),
        
        # 第70行附近
        ('partial = functools.partial(make_env_lambda, env_id=cfg.env.name,',
         'partial = functools.partial(make_env_lambda, env_id=cfg.env.env_id,'),
    ]
    
    changes_made = 0
    for old_text, new_text in replacements:
        if old_text in content:
            content = content.replace(old_text, new_text)
            changes_made += 1
            print(f"✅ 已替换: cfg.env.name → cfg.env.env_id")
    
    if changes_made == 0:
        print("ℹ️ 未发现需要修复的问题（可能已经修复）")
        return True
    
    # 创建备份
    backup_path = env_utils_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"📄 已创建备份: {backup_path}")
    
    # 写入修复后的内容
    with open(env_utils_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已修复文件: {env_utils_path}")
    
    return True


def verify_fix():
    """验证修复是否成功"""
    print("\n🔍 验证修复结果...")
    
    from omegaconf import DictConfig
    import sys
    
    # 添加路径以导入模块
    base_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(base_path))
    
    try:
        from rl_new.sac_cont_sy.env_utils import make_train_environment
        
        # 测试各环境
        test_passed = True
        for env_id in ["NewPasture-v2", "NewPasture-v4", "NewPasture-v5"]:
            cfg = DictConfig({
                "env": {
                    "env_id": env_id,
                    "seed": 42
                },
                "collector": {
                    "env_per_collector": 1
                }
            })
            
            try:
                env = make_train_environment(cfg, device="cpu")
                obs = env.reset()
                print(f"✅ {env_id}: 环境创建成功 (obs shape = {obs['observation'].shape})")
                env.close()
            except Exception as e:
                print(f"❌ {env_id}: {e}")
                test_passed = False
        
        return test_passed
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def main():
    print("=" * 60)
    print("SAC训练脚本环境配置修复工具")
    print("=" * 60)
    
    # 执行修复
    print("\n🔧 开始修复...")
    if not fix_env_utils():
        print("\n❌ 修复失败")
        return 1
    
    # 验证修复
    if verify_fix():
        print("\n✅ 修复成功！现在可以使用以下命令训练不同环境：")
        print("\n# 训练v2环境（APF增强）")
        print("python rl_new/sac_cont_sy/sac-async.py")
        print("\n# 训练v4环境（田地覆盖）")
        print("python rl_new/sac_cont_sy/sac-async.py env.env_id=NewPasture-v4")
        print("\n# 训练v5环境（HIF引导）")
        print("python rl_new/sac_cont_sy/sac-async.py env.env_id=NewPasture-v5")
        return 0
    else:
        print("\n⚠️ 修复已应用但验证未完全通过，请检查是否有其他问题")
        return 2


if __name__ == "__main__":
    exit(main())
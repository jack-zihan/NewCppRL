#!/usr/bin/env python3
"""
测试新的缓冲区管理系统
验证临时文件是否正确存储到/home/lzh/data/rl_buffers
"""

import sys
import tempfile
import time
from pathlib import Path
import yaml
from omegaconf import DictConfig

# 添加项目根目录到路径
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))


def test_buffer_directory_creation():
    """测试缓冲区目录创建"""
    print("🔍 测试缓冲区目录管理系统...")
    print("=" * 60)
    
    # 1. 测试配置加载
    print("\n1️⃣ 测试配置文件加载...")
    config_path = base_dir / 'configs/train_area_coverage_sac_cont_config.yaml'
    cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
    cfg = DictConfig(cfg)
    
    scratch_dir = cfg.buffer.get('scratch_dir', None)
    print(f"✅ 配置的scratch_dir: {scratch_dir}")
    assert scratch_dir == '/home/lzh/data/rl_buffers', "配置路径不正确"
    
    # 2. 测试目录创建
    print("\n2️⃣ 测试目录创建...")
    algo_name = 'area_coverage_sac_cont'
    ckpt_dir = 'test_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    
    scratch_base_dir = cfg.buffer.get('scratch_dir', '/home/lzh/data/rl_buffers')
    scratch_parent = Path(scratch_base_dir) / algo_name / ckpt_dir
    
    print(f"   创建目录: {scratch_parent}")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    assert scratch_parent.exists(), "目录创建失败"
    print(f"✅ 目录创建成功")
    
    # 3. 测试临时文件创建
    print("\n3️⃣ 测试临时文件创建...")
    tempdir = tempfile.TemporaryDirectory(dir=str(scratch_parent), prefix='buffer_')
    scratch_dir = tempdir.name
    
    print(f"   临时目录: {scratch_dir}")
    assert Path(scratch_dir).exists(), "临时目录创建失败"
    assert str(scratch_parent) in scratch_dir, "临时目录位置不正确"
    print(f"✅ 临时目录创建成功")
    
    # 4. 测试文件写入
    print("\n4️⃣ 测试文件写入...")
    test_file = Path(scratch_dir) / 'test.memmap'
    test_file.write_text("test data")
    assert test_file.exists(), "文件写入失败"
    print(f"✅ 文件写入成功: {test_file}")
    
    # 5. 测试清理
    print("\n5️⃣ 测试清理机制...")
    tempdir.cleanup()
    assert not Path(scratch_dir).exists(), "清理失败"
    print(f"✅ 临时目录已清理")
    
    # 6. 清理测试目录
    print("\n6️⃣ 清理测试目录...")
    try:
        scratch_parent.rmdir()
        print(f"✅ 测试目录已清理")
    except:
        print(f"⚠️ 测试目录非空，保留")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
    print("\n📋 总结:")
    print("✅ 配置文件正确设置了scratch_dir路径")
    print("✅ 目录结构正确创建在/home/lzh/data/rl_buffers下")
    print("✅ 临时文件正确存储在指定位置")
    print("✅ 清理机制正常工作")
    
    return True


def test_cleanup_tool():
    """测试清理工具"""
    print("\n🧹 测试清理工具...")
    print("=" * 60)
    
    # 导入清理工具
    sys.path.insert(0, str(base_dir / 'rl_new/utils'))
    from buffer_cleanup import monitor_usage
    
    print("监控当前使用情况:")
    monitor_usage('/home/lzh/data/rl_buffers')
    
    print("\n✅ 清理工具正常工作")
    return True


if __name__ == "__main__":
    try:
        # 运行测试
        success = test_buffer_directory_creation()
        
        # 测试清理工具
        test_cleanup_tool()
        
        print("\n💡 使用说明:")
        print("1. 训练时临时文件将存储在: /home/lzh/data/rl_buffers/<算法>/<时间戳>/buffer_*")
        print("2. 训练结束后自动清理")
        print("3. 使用清理工具手动清理: python rl_new/utils/buffer_cleanup.py")
        print("4. 监控使用情况: python rl_new/utils/buffer_cleanup.py --monitor")
        
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
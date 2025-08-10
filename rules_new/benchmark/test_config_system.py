#!/usr/bin/env python3
"""
测试新的配置系统
"""

import sys
import tempfile
from pathlib import Path
import yaml
import logging

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_direct_config_path():
    """测试直接指定配置文件"""
    logger.info("测试: 直接指定配置文件")
    
    from rules_new.benchmark import BenchmarkRunner
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'benchmark': {
                'algorithms': {
                    'JUMP': {'enabled': True}
                },
                'scenarios': {
                    'seeds': [1, 2],
                    'difficulties': ['easy']
                }
            }
        }
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # 使用直接路径创建runner
        runner = BenchmarkRunner(config_path=config_path)
        
        # 验证配置已加载
        assert 'benchmark' in runner.config
        assert runner.config['benchmark']['scenarios']['seeds'] == [1, 2]
        
        logger.info("✅ 直接指定配置文件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False
    finally:
        # 清理临时文件
        Path(config_path).unlink(missing_ok=True)


def test_config_dir_and_name():
    """测试配置目录 + 配置名称"""
    logger.info("测试: 配置目录 + 配置名称")
    
    from rules_new.benchmark import BenchmarkRunner
    
    # 创建临时目录和配置文件
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # 创建基础配置
        base_config = {
            'environment': {
                'width': 600,
                'height': 600
            }
        }
        with open(config_dir / 'base_config.yaml', 'w') as f:
            yaml.dump(base_config, f)
        
        # 创建自定义配置
        custom_config = {
            'benchmark': {
                'algorithms': {
                    'SNAKE': {'enabled': True}
                },
                'scenarios': {
                    'seeds': [42],
                    'difficulties': ['medium']
                }
            }
        }
        with open(config_dir / 'my_test.yaml', 'w') as f:
            yaml.dump(custom_config, f)
        
        try:
            # 使用config_dir + config_name
            runner = BenchmarkRunner(
                config_dir=config_dir,
                config_name='my_test'
            )
            
            # 验证配置已加载
            assert runner.config['benchmark']['scenarios']['seeds'] == [42]
            assert runner.config['environment']['width'] == 600  # 基础配置也加载了
            
            logger.info("✅ 配置目录 + 配置名称测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            return False


def test_auto_discovery():
    """测试自动发现配置文件"""
    logger.info("测试: 自动发现配置文件")
    
    from rules_new.benchmark import BenchmarkRunner
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # 创建唯一的benchmark配置
        config = {
            'benchmark': {
                'algorithms': {
                    'BCP': {'enabled': True}
                }
            }
        }
        with open(config_dir / 'test_benchmark.yaml', 'w') as f:
            yaml.dump(config, f)
        
        try:
            # 不指定config_name，应该自动找到
            runner = BenchmarkRunner(config_dir=config_dir)
            
            # 验证自动发现的配置
            assert 'BCP' in runner.config['benchmark']['algorithms']
            
            logger.info("✅ 自动发现配置文件测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            return False


def test_multiple_configs_error():
    """测试多个配置文件时的错误提示"""
    logger.info("测试: 多个配置文件错误提示")
    
    from rules_new.benchmark import BenchmarkRunner
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # 创建多个benchmark配置
        for i in range(3):
            with open(config_dir / f'benchmark_{i}.yaml', 'w') as f:
                yaml.dump({'benchmark': {}}, f)
        
        try:
            # 应该抛出错误，提示有多个配置
            runner = BenchmarkRunner(config_dir=config_dir)
            logger.error("❌ 应该抛出错误但没有")
            return False
            
        except ValueError as e:
            if "找到多个基准测试配置文件" in str(e):
                logger.info("✅ 多配置文件错误提示测试通过")
                return True
            else:
                logger.error(f"❌ 错误信息不正确: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ 意外错误: {e}")
            return False


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("测试配置系统改进")
    logger.info("=" * 60)
    
    tests = [
        ("直接配置路径", test_direct_config_path),
        ("配置目录+名称", test_config_dir_and_name),
        ("自动发现配置", test_auto_discovery),
        ("多配置错误提示", test_multiple_configs_error)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n运行: {test_name}")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"测试完成: {passed} 通过, {failed} 失败")
    logger.info("=" * 60)
    
    if failed == 0:
        logger.info("🎉 所有配置系统测试通过！")
        logger.info("\n现在您可以使用：")
        logger.info("1. --config my_config.yaml 直接指定配置文件")
        logger.info("2. --config-dir ./configs --config-name my_experiment")
        logger.info("3. 复制 benchmark_config_template.yaml 创建自己的配置")


if __name__ == '__main__':
    main()
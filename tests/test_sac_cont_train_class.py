#!/usr/bin/env python3
"""
Test script for validating the optimized SAC training class implementation.
Tests component initialization, configuration handling, and basic training loop.
"""

import math
import os
import sys
import tempfile
from pathlib import Path
import yaml
import torch
import numpy as np
from omegaconf import DictConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_new.sac_cont.sac_cont_train_class import (
    TrainingConfig,
    ComponentFactory,
    OptimizedSACTrainer,
    CheckpointManager,
    PerformanceMonitor,
    EarlyStopping
)
from rl.sac_cont.sac_cont_utils import make_sac_models


def test_training_config():
    """Test TrainingConfig dataclass initialization and device detection."""
    print("Testing TrainingConfig...")
    
    # Test with default config
    cfg_dict = {
        'seed': 42,
        'device': 'cpu',
        'training_device': 'cpu',
        'ckpt_name': 'test_run',
        'pretrained_model': None,
        'collector': {
            'frames_per_batch': 1000,
            'total_frames': 10000,
            'init_random_frames': 1000,
            'gpu_devices': None,
            'processes_per_gpu': 2,
            'cpu_workers': None
        },
        'buffer': {
            'buffer_size': 100000,
            'batch_size': 256
        },
        'loss': {
            'gamma': 0.99,
            'target_update_polyak': 0.995,
            'loss_function': 'smooth_l1',
            'utd_ratio': 1.0
        },
        'optim': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_alpha': 3e-4,
            'weight_decay_actor': 0.0,
            'weight_decay_critic': 0.0,
            'weight_decay_alpha': 0.0,
            'max_grad_norm': None
        },
        'logger': {
            'backend': None,
            'test_interval': 10000
        },
        'training': {
            'use_amp': False,
            'checkpoint_interval': 10000,
            'use_early_stopping': False,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.001
        }
    }
    
    cfg = DictConfig(cfg_dict)
    config = TrainingConfig.from_yaml(cfg)
    
    # Check device configuration
    assert config.training_device in ['cuda:0', 'cpu'], f"Invalid device: {config.training_device}"
    print(f"  ✓ Device configured: {config.training_device}")
    
    # Check basic configuration
    assert config.seed == 42, "Seed not set correctly"
    assert config.batch_size == 256, "Batch size not set correctly"
    print("  ✓ Configuration loaded correctly")
    
    print("  ✓ TrainingConfig test passed!\n")


def test_component_factory():
    """Test ComponentFactory for creating training components."""
    print("Testing ComponentFactory...")
    
    # Create minimal config
    cfg_dict = {
        'seed': 42,
        'device': 'cpu',  # Force CPU for testing
        'training_device': 'cpu',
        'ckpt_name': 'test_factory',
        'pretrained_model': None,
        'collector': {
            'frames_per_batch': 100,
            'total_frames': 1000,
            'init_random_frames': 100,
            'gpu_devices': None,
            'processes_per_gpu': 1,
            'cpu_workers': 2
        },
        'buffer': {
            'buffer_size': 1000,
            'batch_size': 32
        },
        'loss': {
            'gamma': 0.99,
            'target_update_polyak': 0.995,
            'loss_function': 'smooth_l1',
            'utd_ratio': 1.0
        },
        'optim': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_alpha': 3e-4,
            'weight_decay_actor': 0.0,
            'weight_decay_critic': 0.0,
            'weight_decay_alpha': 0.0,
            'max_grad_norm': None
        },
        'logger': {
            'backend': None,
            'test_interval': 10000
        },
        'training': {
            'use_amp': False,
            'checkpoint_interval': 10000,
            'use_early_stopping': False,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.001
        }
    }
    
    cfg = DictConfig(cfg_dict)
    config = TrainingConfig.from_yaml(cfg)
    
    # Test actor-critic creation using make_sac_models (like in OptimizedSACTrainer)
    actor_critic = make_sac_models()
    assert actor_critic is not None, "Failed to create actor-critic"
    assert len(actor_critic) == 2, "Actor-critic should have 2 components"
    actor_critic = actor_critic.to('cpu')
    print("  ✓ Actor-critic created successfully")
    
    # Test collector creation
    collector = ComponentFactory.create_collector(config, actor_critic[0])
    assert collector is not None, "Failed to create collector"
    print("  ✓ Collector created successfully")
    
    # Test replay buffer creation (create_replay_buffer creates its own tempdir)
    replay_buffer = ComponentFactory.create_replay_buffer(config)
    assert replay_buffer is not None, "Failed to create replay buffer"
    print("  ✓ Replay buffer created successfully")
    
    # Test loss module creation
    loss_module = ComponentFactory.create_loss_module(config, actor_critic[0], actor_critic[1])
    assert loss_module is not None, "Failed to create loss module"
    print("  ✓ Loss module created successfully")
    
    # Test optimizer creation
    optimizers = ComponentFactory.create_optimizers(config, loss_module)
    assert len(optimizers) == 3, "Should create 3 optimizers"
    print("  ✓ Optimizers created successfully")
    
    # Clean up
    collector.shutdown()
    
    print("  ✓ ComponentFactory test passed!\n")


def test_checkpoint_manager():
    """Test CheckpointManager functionality."""
    print("Testing CheckpointManager...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir)
        manager = CheckpointManager(ckpt_path)
        
        # Create dummy model
        model = torch.nn.Linear(10, 10)
        
        # Save checkpoints
        optimizers = {'actor': torch.optim.Adam(model.parameters())}
        schedulers = {'actor': None}
        metrics = {'reward': 10.5}
        
        path1 = manager.save(model, optimizers, schedulers, 1000, metrics)
        assert path1.exists(), "Failed to save checkpoint"
        print("  ✓ Checkpoint saving working")
        
        # Test loading
        loaded = manager.load_latest()
        assert loaded is not None, "Failed to load checkpoint"
        assert loaded['collected_frames'] == 1000, "Incorrect frames loaded"
        print("  ✓ Checkpoint loading working")
    
    print("  ✓ CheckpointManager test passed!\n")


def test_performance_monitor():
    """Test PerformanceMonitor functionality."""
    print("Testing PerformanceMonitor...")
    
    monitor = PerformanceMonitor()
    
    # Add some metrics
    monitor.track('reward', 10.0)
    monitor.track('reward', 15.0)
    monitor.track('reward', 20.0)
    monitor.track('loss', 0.5)
    monitor.track('loss', 0.3)
    
    # Test statistics
    stats = monitor.get_stats()
    assert 'reward' in stats, "Missing reward in stats"
    assert 'loss' in stats, "Missing loss in stats"
    assert stats['reward']['mean'] == 15.0, f"Expected mean 15.0, got {stats['reward']['mean']}"
    print("  ✓ Statistics collection working")
    
    print("  ✓ PerformanceMonitor test passed!\n")


def test_early_stopping():
    """Test EarlyStopping functionality."""
    print("Testing EarlyStopping...")
    
    early_stop = EarlyStopping(patience=3, min_delta=0.01)
    
    # Simulate improving metrics
    assert not early_stop(10.0), "Should not stop on first metric"
    assert not early_stop(10.05), "Should not stop on small improvement"
    assert not early_stop(10.02), "Should not stop within min_delta"
    
    # Reset and simulate no improvement
    early_stop = EarlyStopping(patience=2, min_delta=0.01)
    assert not early_stop(10.0), "Should not stop on first call"
    assert not early_stop(9.99), "Should not stop on second call (patience=2)"
    assert early_stop(9.99), "Should stop after patience exceeded"
    print("  ✓ Early stopping logic working")
    
    print("  ✓ EarlyStopping test passed!\n")


def test_trainer_initialization():
    """Test OptimizedSACTrainer initialization."""
    print("Testing OptimizedSACTrainer initialization...")
    
    # Create minimal config for testing
    cfg_dict = {
        'seed': 42,
        'device': 'cpu',  # Force CPU for testing
        'training_device': 'cpu',
        'ckpt_name': 'test_trainer',
        'pretrained_model': None,
        'collector': {
            'frames_per_batch': 100,
            'total_frames': 500,
            'init_random_frames': 100,
            'gpu_devices': None,
            'processes_per_gpu': 1,
            'cpu_workers': 2
        },
        'buffer': {
            'buffer_size': 1000,
            'batch_size': 32
        },
        'loss': {
            'gamma': 0.99,
            'target_update_polyak': 0.995,
            'loss_function': 'smooth_l1',
            'utd_ratio': 1.0
        },
        'optim': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_alpha': 3e-4,
            'weight_decay_actor': 0.0,
            'weight_decay_critic': 0.0,
            'weight_decay_alpha': 0.0,
            'max_grad_norm': 1.0
        },
        'logger': {
            'backend': None,
            'test_interval': 10000
        },
        'training': {
            'use_amp': False,
            'checkpoint_interval': 10000,
            'use_early_stopping': False,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.001
        }
    }
    
    cfg = DictConfig(cfg_dict)
    
    # Create TrainingConfig from yaml
    config = TrainingConfig.from_yaml(cfg)
    
    # Initialize trainer with TrainingConfig
    trainer = OptimizedSACTrainer(config)
    assert trainer is not None, "Failed to initialize trainer"
    print("  ✓ Trainer initialized successfully")
    
    # Test component creation
    assert trainer.actor_critic is not None, "Actor-critic not created"
    assert trainer.collector is not None, "Collector not created"
    assert trainer.replay_buffer is not None, "Replay buffer not created"
    assert trainer.loss_module is not None, "Loss module not created"
    print("  ✓ All components created successfully")
    
    # Clean up
    trainer.collector.shutdown()
    
    print("  ✓ OptimizedSACTrainer initialization test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running SAC Training Class Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_training_config,
        test_component_factory,
        test_checkpoint_manager,
        test_performance_monitor,
        test_early_stopping,
        test_trainer_initialization
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}\n")
            failed_tests.append((test.__name__, e))
    
    print("=" * 60)
    if failed_tests:
        print(f"Tests completed with {len(failed_tests)} failures:")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")
    else:
        print("All tests passed successfully! ✓")
    print("=" * 60)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
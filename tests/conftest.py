"""
pytest配置文件和共享fixtures

提供最小化配置和Mock对象,用于组件级独立测试
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from unittest.mock import MagicMock, Mock
from tensordict import TensorDict


@pytest.fixture
def minimal_cfg():
    """最小化配置对象（CPU、小batch、小buffer）

    用于本地快速测试，不依赖GPU资源
    """
    cfg_dict = {
        # 基础设置
        'seed': 42,
        'in_server': False,

        # 设备配置
        'training': {
            'device': 'cpu',
            'use_amp': False,
        },

        # 环境配置
        'env': {
            'env_id': 'NewPasture-v4',
            'env_kwargs': {
                'reward_field_group_coef': 1.0,
                'reward_turning_group_coef': 0.5,
                'reward_overlap_penalty': 0.0,
                'overlap_tolerance': 0.0,
                'field_scale_enabled': False,
                'field_scale_range': (1.0, 1.0),
            }
        },

        # 采集器配置
        'collector': {
            'num_collectors': 1,
            'env_per_collector': 2,
            'frames_per_batch': 10,
            'total_frames': 100,
            'init_random_frames': 20,
        },

        # 缓冲区配置
        'buffer': {
            'buffer_size': 100,
            'batch_size': 2,
            'bucketed': True,
            'success_threshold': 0.99,
            'near_end_threshold': 0.90,
            'bucket_capacity_ratio': [0.25, 0.25, 0.50],
            'alpha': 0.6,
            'beta': 0.4,
            'prefetch': 0,
            'pin_memory': False,
            'temp_dir': None,
        },

        # 损失配置
        'loss': {
            'gamma': 0.99,
            'n_steps': 1,
            'loss_function': 'smooth_l1',
            'alpha_init': 1.0,
            'target_update_polyak': 0.005,
            'utd_ratio': 1.0,
        },

        # 优化器配置
        'optim': {
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_alpha': 3e-4,
            'weight_decay_actor': 0.0,
            'weight_decay_critic': 0.0,
            'weight_decay_alpha': 0.0,
            'eps_actor': 1e-8,
            'eps_critic': 1e-8,
            'max_grad_norm': 0.0,
        },

        # HIF配置
        'hif': {
            'enabled': True,
            'decoder_type': 'lightweight',
            'tv_weight': 0.0,
            'use_tv': False,
            'pretrain': {
                'enabled': True,
                'actor_lr': 1e-3,
                'min_buffer_frames': 20,
                'max_updates': 10,
            },
            'weights': {
                'S1': 0.5,
                'S2': 0.3,
                'S3': 0.1,
            }
        },

        # 课程学习配置
        'curriculum': {
            'enabled': True,
            's1_consecutive_k': 3,
            's1_min_completion': 0.90,
            's2_consecutive_k': 5,
            's2_min_completion': 0.95,
            's2s3_threshold': 0.05,
            'stages': [
                {
                    'name': 'S1',
                    'reward_field_group_coef': 1.0,
                    'reward_turning_group_coef': 0.5,
                    'reward_overlap_penalty': 0.0,
                    'overlap_tolerance': 0.0,
                    'field_scale_enabled': False,
                    'field_scale_range': (1.0, 1.0),
                    'sampling_ratio': (0.40, 0.30, 0.30),
                },
                {
                    'name': 'S2',
                    'reward_field_group_coef': 0.5,
                    'reward_turning_group_coef': 0.2,
                    'reward_overlap_penalty': -0.10,
                    'overlap_tolerance': 0.0,
                    'field_scale_enabled': False,
                    'field_scale_range': (1.0, 1.0),
                    'sampling_ratio': (0.30, 0.30, 0.40),
                },
                {
                    'name': 'S3',
                    'reward_field_group_coef': 0.10,
                    'reward_turning_group_coef': 0.05,
                    'reward_overlap_penalty': -0.20,
                    'overlap_tolerance': 0.0,
                    'field_scale_enabled': False,
                    'field_scale_range': (1.0, 1.0),
                    'sampling_ratio': (0.20, 0.30, 0.50),
                },
            ]
        },

        # Logger配置
        'logger': {
            'backend': None,
            'mode': 'disabled',
            'model_name': 'test',
            'exp_name': 'test',
            'project_name': 'test',
            'group_name': 'test',
            'eval_episodes': 2,
            'eval_max_steps': 100,
            'eval_interval': 50,
            'eval_worker': 1,
            'eval_video': False,
            'eval_video_skip': 10,
        },

        # 编译配置
        'compile': {
            'enable': False,
            'mode': None,
            'cudagraphs': False,
        },

        # Checkpoint配置
        'checkpoint': {
            'save_interval': 1000,
        },

        # 预训练模型路径
        'pretrained_model': None,
    }

    return OmegaConf.create(cfg_dict)


@pytest.fixture
def mock_env():
    """Mock环境（用于测试采集器创建）"""
    env = Mock()
    env.device = torch.device('cpu')
    env.batch_size = torch.Size([2])  # 2个并行环境

    # observation_spec
    obs_spec = Mock()
    obs_spec.shape = torch.Size([2, 4, 64, 64])  # [B, C, H, W]
    obs_spec.dtype = torch.float32
    obs_spec.device = torch.device('cpu')
    env.observation_spec = obs_spec

    # action_spec
    action_spec = Mock()
    action_spec.shape = torch.Size([2, 2])  # [B, action_dim]
    action_spec.dtype = torch.float32
    action_spec.device = torch.device('cpu')
    action_spec.space = Mock()
    action_spec.space.low = -1.0
    action_spec.space.high = 1.0
    action_spec.to = Mock(return_value=action_spec)
    env.action_spec = action_spec

    # reward_spec
    reward_spec = Mock()
    reward_spec.shape = torch.Size([2, 1])
    reward_spec.dtype = torch.float32
    env.reward_spec = reward_spec

    # done_spec
    done_spec = Mock()
    done_spec.shape = torch.Size([2, 1])
    done_spec.dtype = torch.bool
    env.done_spec = done_spec

    return env


@pytest.fixture
def mock_actor():
    """Mock Actor模型"""
    actor = Mock()
    actor.device = torch.device('cpu')

    # state_dict for checkpoint
    actor.state_dict = Mock(return_value={'dummy': torch.tensor([1.0])})
    actor.load_state_dict = Mock()

    # forward pass
    def forward(td):
        B = td.batch_size[0]
        return TensorDict({
            'action': torch.randn(B, 2),
            'loc': torch.randn(B, 2),
            'scale': torch.ones(B, 2) * 0.1,
        }, batch_size=[B])

    actor.forward = forward
    actor.__call__ = forward

    # parameters
    actor.parameters = Mock(return_value=[torch.randn(10, 10)])

    return actor


@pytest.fixture
def mock_critic():
    """Mock Critic模型"""
    critic = Mock()
    critic.device = torch.device('cpu')

    # state_dict for checkpoint
    critic.state_dict = Mock(return_value={'dummy': torch.tensor([2.0])})
    critic.load_state_dict = Mock()

    # forward pass
    def forward(td):
        B = td.batch_size[0]
        return TensorDict({
            'state_action_value': torch.randn(B, 1),
        }, batch_size=[B])

    critic.forward = forward
    critic.__call__ = forward

    # parameters
    critic.parameters = Mock(return_value=[torch.randn(20, 20)])

    return critic


@pytest.fixture
def mock_logger():
    """Mock Logger（wandb或tensorboard）"""
    logger = MagicMock()
    logger.experiment = MagicMock()
    logger.experiment.log = MagicMock()
    logger.experiment.define_metric = MagicMock()
    logger.experiment.finish = MagicMock()
    return logger


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """临时checkpoint目录"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def sample_tensordict():
    """示例TensorDict（用于测试回放缓冲区和损失函数）"""
    B = 2  # batch_size

    td = TensorDict({
        'observation': torch.randn(B, 4, 64, 64),
        'action': torch.randn(B, 2),
        'reward': torch.randn(B, 1),
        'done': torch.tensor([[False], [True]]),
        'truncated': torch.tensor([[False], [False]]),
        'completion_ratio': torch.tensor([[0.85], [0.99]]),
        'next': TensorDict({
            'observation': torch.randn(B, 4, 64, 64),
            'reward': torch.randn(B, 1),
            'done': torch.tensor([[False], [True]]),
            'truncated': torch.tensor([[False], [False]]),
            'completion_ratio': torch.tensor([[0.86], [1.00]]),
            'episode_reward': torch.tensor([[120.5], [250.3]]),
            'step_count': torch.tensor([[100], [200]]),
        }, batch_size=[B]),
    }, batch_size=[B])

    return td


@pytest.fixture
def sample_evaluation_metrics():
    """示例评估指标列表"""
    return [
        {
            'step': 1000,
            'metrics': {
                'eval/reward_mean': 150.5,
                'eval/episode_length': 180.0,
                'eval/completion_ratio': 0.92,
                'eval/ratio_95_to_done_mean': 0.15,
            }
        },
        {
            'step': 2000,
            'metrics': {
                'eval/reward_mean': 200.3,
                'eval/episode_length': 190.0,
                'eval/completion_ratio': 0.96,
                'eval/ratio_95_to_done_mean': 0.14,
            }
        },
    ]


# pytest配置
def pytest_configure(config):
    """pytest全局配置：设置随机种子保证可重复性。"""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture(scope="session", autouse=True)
def _force_cpu():
    """强制 CPU 测试（确定性、硬件无关）。

    用 session-scoped autouse fixture 替代在 pytest_configure 里直接全局改写
    torch.cuda.is_available：测试结束后自动还原原函数，不把"GPU 不可用"这一谎言
    泄漏给 conftest 之后在同一解释器里 import torch 的任何代码/工具。
    """
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original

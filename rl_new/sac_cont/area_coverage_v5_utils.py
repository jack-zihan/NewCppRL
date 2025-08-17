"""
Area Coverage V5 SAC专用工具函数
为Pasture-v5环境提供专用的环境创建和模型构建功能
V5使用SGCNN多尺度观测（20通道）和方向场奖励
"""
import torch
import torch.nn
from pathlib import Path
import yaml
from omegaconf import DictConfig
import gymnasium as gym
from torchrl.envs import RewardSum, StepCounter, TransformedEnv, ParallelEnv
from torchrl.envs.libs.gym import GymWrapper
from tensordict.nn import TensorDictModule, InteractionType, NormalParamExtractor
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from torchrl_utils.model.deep_q_net import DeepQNet
import envs  # noqa

# 使用V5专用的配置文件
base_dir = Path(__file__).parent.parent.parent
area_coverage_v5_cfg = DictConfig(yaml.load(
    open(f'{base_dir}/configs/env_config_area_coverage_v5.yaml'), 
    Loader=yaml.FullLoader
))


def make_area_coverage_v5_env_lambda(
        device="cpu",
        from_pixels=False,
):
    """创建单个area_coverage V5环境"""
    env = gym.make(
        render_mode='rgb_array' if from_pixels else None,
        **area_coverage_v5_cfg.env.params,
    )
    env = GymWrapper(
        env,
        device=device,
        from_pixels=from_pixels,
        pixels_only=False,
    )
    return env


def make_area_coverage_v5_env(
        num_envs=1,
        device="cpu",
        from_pixels=False,
):
    """
    创建area_coverage V5环境（单个或并行）
    使用env_config_area_coverage_v5.yaml配置
    V5强制使用SGCNN，输出20通道
    """
    if num_envs == 1:
        env = make_area_coverage_v5_env_lambda(
            device=device,
            from_pixels=from_pixels,
        )
    else:
        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=lambda: make_area_coverage_v5_env_lambda(
                device=device,
                from_pixels=from_pixels,
            ),
        )
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


def make_area_coverage_v5_sac_modules(proof_environment):
    """根据V5环境创建SAC模块（20通道输入）"""
    # 获取输入形状
    input_shape = proof_environment.observation_spec["observation"].shape
    action_spec = proof_environment.action_spec
    if proof_environment.batch_size:
        action_spec = action_spec[(0,) * len(proof_environment.batch_size)]

    # 定义分布类和参数
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": True,
    }
    
    # 定义输入键
    in_keys = ["observation", "vector"]

    # 创建策略网络（适配20通道输入）
    encoder_out_dim = 512
    policy_net = DeepQNet(
        raster_shape=input_shape,  # (20, 16, 16) for V5
        cnn_channels=(32, 64, 64),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        vec_dim=1,
        hidden_dim=encoder_out_dim,
        output_num=2 * action_spec.shape[-1],
        cnn_activation_class=torch.nn.SiLU,
        mlp_activation_class=torch.nn.SiLU,
        action_head=NormalParamExtractor(
            scale_mapping=f"biased_softplus_{1.0}",
            scale_lb=1e-4,
        ),
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=in_keys,
        out_keys=["loc", "scale"],
    )
    policy_module = ProbabilisticActor(
        spec=action_spec,
        module=policy_module,
        in_keys=["loc", "scale"],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )
    
    # 创建Q值网络（适配20通道输入）
    qvalue_net = DeepQNet(
        raster_shape=input_shape,  # (20, 16, 16) for V5
        cnn_channels=(32, 64, 64),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        vec_dim=1 + 2,  # vector + action
        hidden_dim=encoder_out_dim,
        output_num=1,
        cnn_activation_class=torch.nn.SiLU,
        mlp_activation_class=torch.nn.SiLU,
    )
    qvalue_module = ValueOperator(
        in_keys=in_keys + ["action"],
        module=qvalue_net,
    )
    
    return policy_module, qvalue_module


def make_area_coverage_v5_sac_models():
    """
    创建area_coverage V5 SAC模型
    使用V5环境配置，自动适配20通道输入
    """
    # 使用V5环境创建模型
    proof_environment = make_area_coverage_v5_env(device="cpu")
    
    # 验证环境确实是20通道
    obs_shape = proof_environment.observation_spec["observation"].shape
    print(f"V5环境观察空间: {obs_shape}")
    assert obs_shape[0] == 20, f"V5环境应该是20通道，但得到{obs_shape[0]}通道"
    
    policy_module, qvalue_module = make_area_coverage_v5_sac_modules(
        proof_environment
    )
    actor_critic = torch.nn.ModuleList([policy_module, qvalue_module])

    # 验证模型
    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        for net in actor_critic:
            td_ = net(td)
        del td

    del proof_environment

    return actor_critic
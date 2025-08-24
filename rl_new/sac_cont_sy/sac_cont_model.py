"""
SAC 模型创建模块
简洁优雅的设计，自动适配不同环境版本（v1-v5）
"""
import torch
import torch.nn
from tensordict.nn import TensorDictModule, InteractionType, NormalParamExtractor
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from torchrl_utils.model.deep_q_net import DeepQNet


def make_sac_modules(env):
    """
    根据环境规格创建 SAC 核心模块
    
    Args:
        env: TorchRL 环境实例，提供 observation_spec 和 action_spec
    
    Returns:
        tuple: (policy_module, qvalue_module)
    """
    # 从环境获取规格
    input_shape = env.observation_spec["observation"].shape
    action_spec = env.action_spec
    if env.batch_size:
        action_spec = action_spec[(0,) * len(env.batch_size)]

    # 验证动作空间类型
    if not hasattr(action_spec.space, 'low') or not hasattr(action_spec.space, 'high'):
        raise ValueError(
            "SAC需要连续动作空间！当前环境可能使用了离散动作空间。\n"
            "请使用 make_sac_env() 或在 make_env() 中指定 action_type='continuous'"
        )

    # 分布配置
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": True,
    }
    
    # 输入键
    in_keys = ["observation", "vector"]
    
    # 网络架构参数
    encoder_out_dim = 512
    cnn_channels = (32, 64, 64)
    kernel_sizes = (3, 3, 3)
    strides = (1, 1, 1)
    
    # Policy 网络：输出动作的均值和标准差
    policy_net = DeepQNet(
        raster_shape=input_shape,  # 自动适配不同环境的观测维度
        cnn_channels=cnn_channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        vec_dim=1,  # vector 输入维度
        hidden_dim=encoder_out_dim,
        output_num=2 * action_spec.shape[-1],  # loc + scale
        cnn_activation_class=torch.nn.SiLU,
        mlp_activation_class=torch.nn.SiLU,
        action_head=NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
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
    
    # Q 网络：评估状态-动作对的价值
    qvalue_net = DeepQNet(
        raster_shape=input_shape,  # 自动适配不同环境的观测维度
        cnn_channels=cnn_channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        vec_dim=3,  # 1 (vector) + 2 (action)
        hidden_dim=encoder_out_dim,
        output_num=1,  # Q值
        cnn_activation_class=torch.nn.SiLU,
        mlp_activation_class=torch.nn.SiLU,
    )
    
    qvalue_module = ValueOperator(
        in_keys=in_keys + ["action"],
        module=qvalue_net,
    )
    
    return policy_module, qvalue_module


def make_sac_models(env=None, env_id="NewPasture-v2", **env_kwargs):
    """
    创建 SAC 模型（Actor-Critic），支持自动创建环境
    
    Args:
        env: TorchRL 环境实例（可选）
            - 如果提供，直接使用
            - 如果为None，根据env_id创建
        env_id: 环境ID（当env为None时使用）
            - 默认: "NewPasture-v2"
            - 可选: v1, v2, v3, v4, v5
        **env_kwargs: 传递给环境的额外参数
    
    Returns:
        torch.nn.ModuleList: [policy_module, qvalue_module]
    
    Example:
        >>> # 方式1: 自动创建环境
        >>> model = make_sac_models(env_id="NewPasture-v3")
        >>> 
        >>> # 方式2: 使用已有环境
        >>> from torchrl_utils_new import make_sac_env
        >>> env = make_sac_env(env_id="NewPasture-v2", device="cuda")
        >>> model = make_sac_models(env=env)
        >>> env.close()
    """
    # 如果没有提供环境，创建一个
    if env is None:
        from torchrl_utils_new.utils_env import make_sac_env
        env = make_sac_env(env_id=env_id, num_envs=1, **env_kwargs)
        need_close = True
    else:
        need_close = False
    
    # 创建核心模块
    policy_module, qvalue_module = make_sac_modules(env)
    
    # 如果是临时创建的环境，关闭它
    if need_close:
        env.close()
    
    # 组合成 Actor-Critic
    actor_critic = torch.nn.ModuleList([policy_module, qvalue_module])
    
    return actor_critic
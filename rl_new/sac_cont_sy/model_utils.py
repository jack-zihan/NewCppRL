"""
SAC 模型创建模块
简洁优雅的设计，支持设备指定
"""
import torch
import torch.nn
from tensordict.nn import TensorDictModule, InteractionType, NormalParamExtractor
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl_utils.model.deep_q_net import DeepQNet


def make_sac_models(env, device="cpu"):
    """
    创建 SAC 模型（Actor-Critic），支持设备指定
    
    Args:
        env: TorchRL 环境实例
        device: 目标设备 (默认 "cpu")
    
    Returns:
        torch.nn.ModuleList: [policy_module, qvalue_module]
    """
    # 1. 从环境获取规格
    input_shape = env.observation_spec["observation"].shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]  # 去除batch维度
    
    # 2. 获取action_spec并移到正确设备（关键！）
    action_spec = env.action_spec
    if env.batch_size:
        action_spec = action_spec[(0,) * len(env.batch_size)]
    action_spec = action_spec.to(device)  # 这是关键，确保bounds在正确设备
    
    # 3. 验证动作空间类型
    if not hasattr(action_spec.space, 'low') or not hasattr(action_spec.space, 'high'):
        raise ValueError(
            "SAC需要连续动作空间！当前环境可能使用了离散动作空间。\n"
            "请使用 make_sac_env() 或在 make_env() 中指定 action_type='continuous'"
        )
    
    # 4. 网络架构参数（保持原有配置）
    in_keys = ["observation", "vector"]
    encoder_out_dim = 512
    cnn_channels = (32, 64, 64)
    kernel_sizes = (3, 3, 3)
    strides = (1, 1, 1)
    
    # 5. Policy 网络：输出动作的均值和标准差
    policy_net = DeepQNet(
        raster_shape=input_shape,
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
    ).to(device)  # 网络移到设备
    
    policy_module = TensorDictModule(
        policy_net,
        in_keys=in_keys,
        out_keys=["loc", "scale"],
    )
    
    policy_module = ProbabilisticActor(
        spec=action_spec,  # action_spec已经在正确设备上
        module=policy_module,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,    # 现在这些bounds在正确设备上！
            "high": action_spec.space.high,
            "tanh_loc": True,
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )
    
    # 6. Q 网络：评估状态-动作对的价值
    qvalue_net = DeepQNet(
        raster_shape=input_shape,
        cnn_channels=cnn_channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        vec_dim=3,  # 1 (vector) + 2 (action)
        hidden_dim=encoder_out_dim,
        output_num=1,  # Q值
        cnn_activation_class=torch.nn.SiLU,
        mlp_activation_class=torch.nn.SiLU,
    ).to(device)  # 网络移到设备
    
    qvalue_module = ValueOperator(
        in_keys=in_keys + ["action"],
        module=qvalue_net,
    )
    
    # 7. 组合成 Actor-Critic
    actor_critic = torch.nn.ModuleList([policy_module, qvalue_module])
    
    # 8. 懒加载初始化（与官方一致）
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict().to(device)
        for net in actor_critic:
            net(td)
    # 清理dummy环境
    env.close()
    del env

    return actor_critic


if __name__ == "__main__":
    from torchrl_utils_new import make_sac_env

    # 测试CPU设备
    env = make_sac_env(env_id="NewPasture-v4", device="cpu")
    model = make_sac_models(env, device="cpu")
    obs, _ = env.reset()
    action = model[0](obs)
    print(f"CPU test passed: {action}")
    env.close()
    
    # 如果有CUDA，测试GPU设备
    if torch.cuda.is_available():
        env = make_sac_env(env_id="NewPasture-v4", device="cuda")
        model = make_sac_models(env, device="cuda")
        obs, _ = env.reset()
        action = model[0](obs)
        print(f"CUDA test passed: {action}")
        env.close()
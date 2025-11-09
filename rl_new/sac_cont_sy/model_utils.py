"""SAC 模型创建模块：简洁优雅的设计，支持设备指定"""
import torch
import torch.nn as nn
from tensordict import TensorDict as _TensorDict
from tensordict.nn import TensorDictModule, InteractionType, NormalParamExtractor, TensorDictSequential
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl_utils.model.deep_q_net import DeepQNet
from torchrl_utils.model.resnet_fpn_dual import ResNetFPNDualHeadActor, ResNetFPNCritic


def _prepare_env_specs(env, device):
    """提取环境规格并移到目标设备（关键：action_spec的bounds必须在目标设备上）"""
    input_shape = env.observation_spec["observation"].shape
    input_shape = input_shape[1:] if len(input_shape) == 4 else input_shape

    action_spec = env.action_spec
    if env.batch_size:
        action_spec = action_spec[(0,) * len(env.batch_size)]
    action_spec = action_spec.to(device)  # 关键：TanhNormal的bounds继承此设备

    if not hasattr(action_spec.space, 'low') or not hasattr(action_spec.space, 'high'):
        raise ValueError("SAC需要连续动作空间！请确保环境配置为 action_type='continuous'")

    vec_dim = env.observation_spec["vector"].shape[-1]
    action_dim = action_spec.shape[-1]

    return input_shape, action_spec, vec_dim, action_dim


class DualOutputActorWrapper(nn.Module):
    """双输出Actor包装器：同时返回action_params和pred_ego_hif [B,2,H,W]
    注：必须在模块顶层定义以支持多进程采集器的pickling"""

    def __init__(self, actor_net):
        super().__init__()
        self.actor_net = actor_net

    def forward(self, observation, vector):
        action_params, pred_ego_hif = self.actor_net(observation, vector, return_hif=True)
        return {"action_params": action_params, "pred_ego_hif": pred_ego_hif}


def make_sac_models(env, device="cpu"):
    """创建SAC模型（Actor-Critic），支持设备指定"""
    # 准备环境规格（关键：action_spec的bounds必须在目标设备上，否则TanhNormal会报错）
    input_shape, action_spec, vec_dim, action_dim = _prepare_env_specs(env, device)

    # CNN架构配置
    in_keys = ["observation", "vector"]
    encoder_out_dim = 512
    cnn_channels, kernel_sizes, strides = (64, 128, 256), (3, 3, 3), (1, 1, 1)

    # Policy网络：CNN编码器 + MLP头部 → (loc, scale)
    policy_net = DeepQNet(
        raster_shape=input_shape, cnn_channels=cnn_channels, kernel_sizes=kernel_sizes, strides=strides,
        vec_dim=vec_dim, hidden_dim=encoder_out_dim, output_num=2*action_dim,
        cnn_activation_class=torch.nn.SiLU, mlp_activation_class=torch.nn.SiLU,
        action_head=NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4)
    ).to(device)

    policy_module = ProbabilisticActor(
        spec=action_spec,
        module=TensorDictModule(policy_net, in_keys=in_keys, out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"], distribution_class=TanhNormal,
        distribution_kwargs={"low": action_spec.space.low, "high": action_spec.space.high, "tanh_loc": True},
        default_interaction_type=InteractionType.RANDOM, return_log_prob=False
    )

    # Q网络：输入观察+动作 → Q值标量
    qvalue_net = DeepQNet(
        raster_shape=input_shape, cnn_channels=cnn_channels, kernel_sizes=kernel_sizes, strides=strides,
        vec_dim=vec_dim+action_dim, hidden_dim=encoder_out_dim, output_num=1,
        cnn_activation_class=torch.nn.SiLU, mlp_activation_class=torch.nn.SiLU
    ).to(device)

    qvalue_module = ValueOperator(in_keys=in_keys+["action"], module=qvalue_net)

    # 懒加载初始化（TorchRL要求）
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict().to(device)
        policy_module(td)
        qvalue_module(td)

    env.close()
    del env

    return policy_module, qvalue_module


def make_sac_resnet_dual_models(env, device="cpu", hif_decoder_type="two_stage"):
    """创建ResNet-FPN SAC模型（Actor同时输出action和HIF预测）

    返回两个独立模块：
    1. Actor：ProbabilisticActor，同时输出action和pred_ego_hif
    2. Critic：ValueOperator，独立编码器

    Args:
        env: TorchRL环境实例
        device: 目标设备（默认"cpu"）
        hif_decoder_type: HIF解码器类型（"two_stage"或"full"）
    """
    # 准备环境规格
    input_shape, action_spec, vec_dim, action_dim = _prepare_env_specs(env, device)
    in_channels = input_shape[0]

    # Actor网络：ResNet-FPN编码器 + 双头（action + HIF）
    actor_net = ResNetFPNDualHeadActor(
        in_channels=in_channels, vec_dim=vec_dim, action_dim=action_dim,
        fpn_channels=256, hidden_dim=512, pretrained=True,
        hif_decoder_type=hif_decoder_type, decoder_channels=128
    ).to(device)

    # 双输出包装器 → 参数提取器 → ProbabilisticActor
    actor_base = TensorDictModule(
        DualOutputActorWrapper(actor_net),
        in_keys=["observation", "vector"],
        out_keys=["action_params", "pred_ego_hif"]
    )

    param_module = TensorDictModule(
        NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
        in_keys=["action_params"], out_keys=["loc", "scale"]
    )

    actor = ProbabilisticActor(
        spec=action_spec,
        module=TensorDictSequential(actor_base, param_module),
        in_keys=["loc", "scale"], distribution_class=TanhNormal,
        distribution_kwargs={"low": action_spec.space.low, "high": action_spec.space.high, "tanh_loc": True},
        default_interaction_type=InteractionType.RANDOM, return_log_prob=False
    )

    # Critic网络：独立的ResNet-FPN编码器
    critic_net = ResNetFPNCritic(
        in_channels=in_channels, vec_dim=vec_dim, action_dim=action_dim,
        fpn_channels=256, hidden_dim=512, pretrained=True
    ).to(device)

    critic = ValueOperator(in_keys=["observation", "vector", "action"], module=critic_net)

    # 懒加载初始化
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td0 = env.fake_tensordict().to(device)
        td = _TensorDict({
            "observation": td0["observation"].unsqueeze(0) if td0["observation"].dim()==3 else td0["observation"],
            "vector": td0["vector"].unsqueeze(0) if td0["vector"].dim()==1 else td0["vector"],
        }, batch_size=[1])

        actor(td)
        critic(td)

    env.close()
    del env

    return actor, critic


if __name__ == "__main__":
    from torchrl_utils_new import make_sac_env

    # 测试CPU设备
    env_cpu = make_sac_env(env_id="NewPasture-v4", device="cpu")
    policy_cpu, qvalue_cpu = make_sac_models(env_cpu, device="cpu")
    print(f"✅ CPU模型创建成功: policy={type(policy_cpu).__name__}, qvalue={type(qvalue_cpu).__name__}")

    # 如果有CUDA，测试GPU设备
    if torch.cuda.is_available():
        env_cuda = make_sac_env(env_id="NewPasture-v4", device="cuda")
        policy_cuda, qvalue_cuda = make_sac_models(env_cuda, device="cuda")
        print(f"✅ CUDA模型创建成功: policy={type(policy_cuda).__name__}, qvalue={type(qvalue_cuda).__name__}")
    else:
        print("⚠️  CUDA不可用，跳过GPU测试")

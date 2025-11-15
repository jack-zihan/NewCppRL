"""TorchRL环境工具：envs_new环境创建、HIF包装、Transform链（无YAML依赖）"""
import envs_new  # noqa - 触发Gymnasium注册

import functools, torch, numpy as np, gymnasium as gym
from torchrl.envs import (Compose, DoubleToFloat, EnvBase, EnvCreator, GymWrapper,
                         ParallelEnv, TransformedEnv)
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, Transform
from torchrl.record import VideoRecorder
from torchrl.data import UnboundedContinuous

class PredEgoHIFInjectionEnv(EnvBase):
    """HIF预测注入环境包装（v5/v6可视化）：policy(td)["pred_ego_hif"] → env.maps_dict → render()"""

    def __init__(self, env):
        super().__init__(device=env.device, batch_size=env.batch_size)
        self._env = env
        self._make_spec_from_env()

    def _get_base_env(self):
        """获取最底层Gymnasium环境"""
        env = self._env
        while hasattr(env, '_env'):
            env = env._env
            if hasattr(env, 'env'): env = env.env  # GymWrapper
        return env.unwrapped

    def _step(self, tensordict):
        """注入HIF预测到环境（fail-fast访问pred_ego_hif）"""
        pred = tensordict["pred_ego_hif"]  # [B,2,H,W] 或 [2,H,W], fail-fast if model不产生
        cos2, sin2 = (pred[0,0], pred[0,1]) if pred.dim()==4 else (pred[0], pred[1])
        if pred.dim() not in (3,4): raise RuntimeError(f"Unexpected pred_ego_hif shape: {pred.shape}")

        cos_np, sin_np = cos2.detach().cpu().numpy(), sin2.detach().cpu().numpy()
        mag = np.sqrt(sin_np**2 + cos_np**2)
        pred_dict = {'cosine2': cos_np, 'sine2': sin_np, 'confidence': (mag > 0.5).astype(np.float32)}
        self._get_base_env().set_pred_hif(pred_dict)

        tensordict.pop('pred_ego_hif', None)  # 清理避免内存浪费
        return self._env._step(tensordict)

    def _set_seed(self, seed):
        return self._env._set_seed(seed)

    # EnvCreator.init_() 阶段用rand_step避免缺pred_ego_hif触发fail-fast，rollout用_step
    def rand_step(self, tensordict):
        return self._env.rand_step(tensordict)

    def _reset(self, tensordict=None, **kwargs):
        return self._env._reset(tensordict, **kwargs)

    def _make_spec_from_env(self):
        # 批量克隆所有spec（包括state_spec）
        for attr in ['action_spec', 'reward_spec', 'done_spec', 'observation_spec', 'state_spec']:
            setattr(self, attr, getattr(self._env, attr).clone())

        h, w = self.observation_spec["observation"].shape[-2:]
        self.state_spec["pred_ego_hif"] = UnboundedContinuous(shape=torch.Size([2, h, w]),
                                                              dtype=torch.float32, device=self.device)

class HIFBidirectionalEnv(EnvBase):
    """双向HIF环境：root写入label_ego_hif [3,H,W]（cos2/sin2/confidence）"""

    def __init__(self, env):
        super().__init__(device=env.device, batch_size=env.batch_size)
        self._env = env
        self._make_spec_from_env()

    def _get_base_env(self):
        """获取最底层Gymnasium环境"""
        env = self._env
        while hasattr(env, '_env'):
            env = env._env
            if hasattr(env, 'env'): env = env.env  # GymWrapper
        return env.unwrapped

    def _make_label_hif_tensor(self, label_ego_hif):
        """提取并转换HIF标签为tensor [3,H,W]"""
        return torch.from_numpy(np.stack([label_ego_hif['cosine2'], label_ego_hif['sine2'],
                                         label_ego_hif['confidence']], axis=0)).float()

    def _step(self, tensordict):
        next_tensordict = self._env._step(tensordict)  # 先获取输出
        next_label_hif = self._make_label_hif_tensor(self._get_base_env().maps_dict['label_ego_hif'])
        return next_tensordict.set("label_ego_hif", next_label_hif.to(dtype=torch.float32, device=self.device))  # 返回包含label的输出

    def _reset(self, tensordict=None, **kwargs):
        tensordict = self._env._reset(tensordict, **kwargs)
        label_hif = self._make_label_hif_tensor(self._get_base_env().maps_dict['label_ego_hif'])
        return tensordict.set("label_ego_hif", label_hif.to(dtype=torch.float32, device=self.device))


    def _set_seed(self, seed):
        return self._env._set_seed(seed)

    # EnvCreator.init_() 阶段用rand_step确保底层reward等键正确生成
    def rand_step(self, tensordict):
        return self._env.rand_step(tensordict)

    def _make_spec_from_env(self):
        """从 wrapped env 复制 specs，并注册 label_ego_hif 到 observation_spec。
        设计要点：采集器/并行环境会依据 observation_spec 的键集筛选输出； 因此 wrapper 新增的键必须在此处显式声明，否则会被过滤。
        """
        # 克隆所有 specs
        for attr in ['action_spec', 'reward_spec', 'done_spec', 'observation_spec']:
            setattr(self, attr, getattr(self._env, attr).clone())

        # 从 observation 推断空间尺寸并注册 label_ego_hif
        h, w = self.observation_spec["observation"].shape[-2:]
        self.observation_spec["label_ego_hif"] = UnboundedContinuous(
            shape=torch.Size([3, h, w]), dtype=torch.float32, device=self.device)


class DropPixels(Transform):
    """剔除pixels键减少内存：删除'pixels'与('next','pixels')"""

    def _call(self, tensordict):
        tensordict.pop("pixels", None)
        if "next" in tensordict.keys(): tensordict["next"].pop("pixels", None)
        return tensordict

    def _reset(self, tensordict, tensordict_reset):
        tensordict_reset.pop("pixels", None)
        return tensordict_reset


class KeepLastPixels(Transform):
    """缓存并行环境的最后有效像素，done/黑帧时复用上一帧避免黑屏（VideoRecorder前置）"""

    def __init__(self):
        super().__init__(in_keys=[])
        self._last = None

    def _maybe_init(self, pix: torch.Tensor):
        if self._last is None or self._last.shape != pix.shape or self._last.device != pix.device:
            self._last = torch.zeros_like(pix)

    def _replace_mask(self, tensordict, pix: torch.Tensor) -> torch.Tensor:
        """检测需要替换的帧：像素和为0（黑帧）或done=True"""
        b = pix.shape[0] if pix.ndim > 0 else 1
        flat = pix.reshape(b, -1)
        sums = flat.to(torch.int64).sum(dim=1) if pix.dtype == torch.uint8 else flat.abs().sum(dim=1)
        zero_mask = sums == 0

        done = tensordict.get("done", None)
        done_mask = done.reshape(b, -1).any(dim=1) if isinstance(done, torch.Tensor) \
                    else torch.zeros(b, dtype=torch.bool, device=pix.device)
        return zero_mask | done_mask

    def _call(self, tensordict):
        if not isinstance(pix := tensordict.get("pixels", None), torch.Tensor): return tensordict

        self._maybe_init(pix)
        mask = self._replace_mask(tensordict, pix)

        if mask.any():
            new_pix = pix.clone()
            new_pix[mask] = self._last[mask]
            tensordict.set("pixels", new_pix)
            pix = new_pix

        if (valid := ~mask).any(): self._last[valid] = pix[valid]
        return tensordict

    def _reset(self, tensordict, tensordict_reset):
        pix = tensordict_reset.get("pixels", None)
        if isinstance(pix, torch.Tensor):
            self._maybe_init(pix)
            self._last.copy_(pix)
        else:
            self._last = None
        return tensordict_reset


def make_env_lambda(env_id="NewPasture-v2", device="cpu", from_pixels=False, **env_kwargs):
    """创建单环境：gym.make → GymWrapper，可选注册info键（overlap_count）"""
    env = gym.make(env_id, render_mode='rgb_array' if from_pixels else None, **env_kwargs)
    wrapper = GymWrapper(env, device=device, from_pixels=from_pixels, pixels_only=False)
    try:
        wrapper = wrapper.auto_register_info_dict(info_dict_reader=default_info_dict_reader(keys=['overlap_count']))
    except Exception as e:
        from torchrl._utils import logger as torchrl_logger
        torchrl_logger.warning(f"auto_register_info_dict failed: {e}, continue without info registration")
    return wrapper


class Steps95ToDoneCounter(Transform):
    """统计达到completion≥95%后的步数（并行环境矢量化，暴露于root和next）"""

    def __init__(self, threshold: float = 0.95, out_key: str = "steps_95_to_done"):
        super().__init__(in_keys=[])
        self.threshold, self.out_key = float(threshold), out_key
        self._armed, self._count = None, None  # 将初始化为[B] tensor

    def _call(self, tensordict):
        completion = tensordict.get("completion_ratio")
        comp = completion
        while comp.ndim > 1 and comp.shape[-1] == 1: comp = comp.squeeze(-1)
        comp, B = comp.reshape(-1), comp.reshape(-1).shape[0]

        # 初始化per-env状态
        if self._armed is None or self._armed.shape[0] != B:
            self._armed = torch.zeros(B, dtype=torch.bool, device=completion.device)
            self._count = torch.zeros(B, dtype=torch.int64, device=completion.device)

        # 累计95%→done步数
        self._armed |= (comp >= self.threshold)
        self._count = torch.where(self._armed, self._count + 1, self._count)

        tensordict.set(self.out_key, self._count.reshape(completion.shape).clone())
        return tensordict

    def _reset(self, tensordict, tensordict_reset):
        if self._armed is not None: self._armed.zero_()
        if self._count is not None: self._count.zero_()

        device = self._count.device if self._count is not None else \
                 (tensordict_reset.device if tensordict_reset.device else torch.device("cpu"))
        tensordict_reset.set(self.out_key, torch.zeros(tensordict_reset.batch_size, dtype=torch.int64, device=device))
        return tensordict_reset


def _wrap_with_hif_injection(base_env, env_id):
    """HIF预测注入包装（v5/v6）"""
    return PredEgoHIFInjectionEnv(base_env) if 'v5' in str(env_id).lower() or 'v6' in str(env_id).lower() else base_env

def _wrap_with_hif_extraction(base_env, cfg):
    """HIF标签提取包装（v5/v6+HIF训练）"""
    return HIFBidirectionalEnv(base_env) if cfg.hif.enabled and \
           ('v5' in str(cfg.env.env_id).lower() or 'v6' in str(cfg.env.env_id).lower()) else base_env

def make_train_eval_environment(cfg, logger=None, train_device="cpu", eval_device="cpu"):
    """创建训练+评估环境（评估可选VideoRecorder）"""
    train_parallel = _create_parallel_env(cfg, train_device, cfg.collector.env_per_collector, False, cfg.seed)
    train_env = TransformedEnv(train_parallel, Compose(InitTracker(), StepCounter(), Steps95ToDoneCounter(),
                                                       DoubleToFloat(), RewardSum()))
    eval_parallel = _create_parallel_env(cfg, eval_device, cfg.logger.eval_episodes, cfg.logger.eval_video, None)
    trsf_clone = train_env.transform.clone()
    if cfg.logger.eval_video:
        trsf_clone.insert(0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"],
                                           make_grid=True, skip=cfg.logger.eval_video_skip))
    return train_env, TransformedEnv(eval_parallel, trsf_clone)

def make_train_environment(cfg, device="cpu"):
    """创建训练环境"""
    parallel_env = _create_parallel_env(cfg, device, cfg.collector.env_per_collector, False, cfg.seed)
    return TransformedEnv(parallel_env, Compose(InitTracker(), StepCounter(), Steps95ToDoneCounter(),
                                                DoubleToFloat(), RewardSum()))

def make_single_environment(cfg, device="cpu", seed=None, from_pixels=False):
    """创建单环境实例"""
    env = make_env_lambda(env_id=cfg.env.env_id, device=device, from_pixels=from_pixels,
                          **(cfg.env.get('env_kwargs') or {}))
    env = TransformedEnv(env, Compose(InitTracker(), StepCounter(), Steps95ToDoneCounter(),
                                      DoubleToFloat(), RewardSum()))
    if seed is not None: env.set_seed(seed)
    return env

def _create_parallel_env(cfg, device, num_parallel, from_pixels=False, seed=None):
    """创建并行环境（核心复用逻辑）"""
    env_creator = functools.partial(make_env_lambda, env_id=cfg.env.env_id, device=device, from_pixels=from_pixels,
                                    **(cfg.env.get('env_kwargs') or {}))
    parallel_env = ParallelEnv(num_parallel, EnvCreator(lambda: _wrap_with_hif_extraction(env_creator(), cfg)),
                               serial_for_single=True)
    if seed is not None: parallel_env.set_seed(seed)
    return parallel_env

def make_drop_pixels_eval_environment(cfg, logger=None, eval_device="cpu"):
    """评估环境（HIF预测可视化+KeepLastPixels→VideoRecorder→DropPixels链）"""
    env_creator = functools.partial(make_env_lambda, env_id=cfg.env.env_id, device=eval_device,
                                    from_pixels=cfg.logger.eval_video, **(cfg.env.get("env_kwargs") or {}))
    eval_parallel = ParallelEnv(cfg.logger.eval_episodes, EnvCreator(lambda: _wrap_with_hif_injection(env_creator(),
                                                         cfg.env.env_id)), serial_for_single=True)

    trsf = Compose(InitTracker(), StepCounter(max_steps=cfg.logger.eval_max_steps),
                   Steps95ToDoneCounter(), DoubleToFloat(), RewardSum())

    if cfg.logger.eval_video and logger is not None:
        trsf.insert(0, KeepLastPixels())
        trsf.insert(1, VideoRecorder(logger=logger, tag="eval/video", in_keys=["pixels"],
                                     make_grid=True, skip=cfg.logger.eval_video_skip))
        trsf.insert(2, DropPixels())

    return None, TransformedEnv(eval_parallel, trsf)

"""Lazy pre-upgrade pickle-compat shim.

旧 ckpt 反序列化兼容层：把已改名的 torchrl spec 旧类名别名回新 API，并让
InteractionType 接受旧的整数枚举值。改为「懒加载」——解释器启动只 import stdlib
并注册一个 sys.meta_path 后置导入钩子，torch 只在 torchrl/tensordict 真正被
import 时才加载。正确性不变：pickle.find_class 会先 import 目标模块再 getattr，
钩子正好在 getattr 之前把别名装上。
"""
import sys
import warnings
import importlib.machinery


def _patch_tensor_specs(_specs):
    from torchrl.data.tensor_specs import (
        Composite, OneHot, Bounded, Categorical, MultiOneHot,
        MultiCategorical, NonTensor, Stacked, StackedComposite,
    )
    _specs.CompositeSpec = Composite
    _specs.OneHotDiscreteTensorSpec = OneHot
    _specs.DiscreteTensorSpec = Categorical
    _specs.BoundedTensorSpec = Bounded
    _specs.MultiOneHotDiscreteTensorSpec = MultiOneHot
    _specs.MultiDiscreteTensorSpec = MultiCategorical
    _specs.NonTensorSpec = NonTensor
    _specs.LazyStackedTensorSpec = Stacked
    _specs.LazyStackedCompositeSpec = StackedComposite
    try:
        from torchrl.data.tensor_specs import Binary, UnboundedContinuous, UnboundedDiscrete
        _specs.BinaryDiscreteTensorSpec = Binary
        _specs.UnboundedContinuousTensorSpec = UnboundedContinuous
        _specs.UnboundedDiscreteTensorSpec = UnboundedDiscrete
    except ImportError:
        pass


def _patch_interaction_type(_mod):
    InteractionType = getattr(_mod, "InteractionType", None)
    if InteractionType is None or getattr(InteractionType, "_unpickle_compat_patched", False):
        return
    _LEGACY = {1: "mode", 2: "median", 3: "mean", 4: "random", 5: "deterministic"}
    _prev = getattr(InteractionType, "_missing_", None)
    def _legacy_missing(cls, value):
        if isinstance(value, int) and value in _LEGACY:
            return cls(_LEGACY[value])
        if _prev is not None:
            return _prev(value)
        return None
    InteractionType._missing_ = classmethod(_legacy_missing)
    InteractionType._unpickle_compat_patched = True


_PATCHES = {
    "torchrl.data.tensor_specs": _patch_tensor_specs,
    "tensordict.nn.probabilistic": _patch_interaction_type,
}


def _safe_patch(name, module):
    try:
        _PATCHES[name](module)
    except Exception as e:
        warnings.warn(f"unpickle_compat: patch for {name} failed: {e}")


class _CompatFinder:
    """后置导入钩子：用 PathFinder.find_spec(fullname, path) 定位目标（path 由导入
    机制传入父包 __path__，不重导父包、无递归、恰好一次 exec），包一层 exec_module
    在模块执行后立即施加 patch，对其它模块一律返回 None。"""
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        real_exec = loader.exec_module
        def exec_module(module, _real=real_exec, _name=fullname):
            _real(module)
            _safe_patch(_name, module)
        loader.exec_module = exec_module
        return spec


# 1) shim 加载时若目标已被 import，立即补 patch
for _name in _PATCHES:
    _m = sys.modules.get(_name)
    if _m is not None:
        _safe_patch(_name, _m)

# 2) 为尚未 import 的目标注册懒钩子（幂等）
if not any(isinstance(f, _CompatFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _CompatFinder())

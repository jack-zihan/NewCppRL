"""Pre-upgrade pickle-compat shim: aliases removed _LegacySpecMeta names back onto current torchrl
public API, and patches InteractionType to accept legacy plain-Enum integer values from old ckpts."""
import warnings

# Part 1: tensor_specs legacy alias map
import torchrl.data.tensor_specs as _specs
from torchrl.data.tensor_specs import (
    Composite, OneHot, Bounded, Categorical, MultiOneHot, MultiCategorical, NonTensor,
    Stacked, StackedComposite,
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

# Part 2: InteractionType legacy Enum bridge
try:
    from tensordict.nn.probabilistic import InteractionType
    _LEGACY_INT_TO_NAME = {1: 'mode', 2: 'median', 3: 'mean', 4: 'random', 5: 'deterministic'}
    _prev_missing = getattr(InteractionType, '_missing_', None)
    def _legacy_enum_missing(cls, value):
        if isinstance(value, int) and value in _LEGACY_INT_TO_NAME:
            return cls(_LEGACY_INT_TO_NAME[value])
        if _prev_missing is not None:
            return _prev_missing(value)
        return None
    InteractionType._missing_ = classmethod(_legacy_enum_missing)
except Exception as e:
    warnings.warn(f"unpickle_compat: InteractionType bridge failed: {e}")

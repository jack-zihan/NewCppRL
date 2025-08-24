"""
Environment utilities for TorchRL training with new environment system.
This version uses envs_new environments without YAML dependency.
"""
from typing import Optional, Dict, Any
from torchrl.envs import InitTracker, StepCounter, DoubleToFloat, RewardSum, TransformedEnv, ParallelEnv
import gymnasium as gym
from torchrl.envs.libs.gym import GymWrapper

# Import to trigger Gymnasium registration
import envs_new  # noqa

# ====================================================================
# Environment utils - New implementation without YAML
# --------------------------------------------------------------------

def make_env_lambda(
        env_id: str = "NewPasture-v2",  # Default to v2 (APF environment)
        device: str = "cpu",
        from_pixels: bool = False,
        **env_kwargs: Dict[str, Any]  # Additional environment configuration
):
    """
    Create a single environment instance.
    
    Args:
        env_id: Gymnasium environment ID (NewPasture-v1/v2/v3/v4/v5)
        device: Device to run the environment on
        from_pixels: Whether to use pixel observations
        **env_kwargs: Additional keyword arguments passed to the environment
            Examples:
            - use_apf: bool (for v2)
            - use_multiscale: bool (for v3)
            - map_dir: str (custom map directory)
            - num_obstacles_range: tuple (obstacle configuration)
            - reward_*: float (reward coefficients)
    
    Returns:
        GymWrapper: Wrapped environment ready for TorchRL
    """
    # Create environment with flexible configuration
    env = gym.make(
        env_id,
        render_mode='rgb_array' if from_pixels else None,
        **env_kwargs
    )
    
    # Wrap for TorchRL compatibility
    env = GymWrapper(
        env,
        device=device,
        from_pixels=from_pixels,
        pixels_only=False,
    )
    return env


def make_env(
        env_id: str = "NewPasture-v2",
        num_envs: int = 1,
        device: str = "cpu",
        from_pixels: bool = False,
        **env_kwargs: Dict[str, Any]
):
    """
    Create single or parallel environments with transforms.
    
    Args:
        env_id: Gymnasium environment ID
            - NewPasture-v1: Base environment with weed
            - NewPasture-v2: APF-enhanced environment (default)
            - NewPasture-v3: Multi-scale features environment
            - NewPasture-v4: Field coverage only (no weed)
            - NewPasture-v5: Field coverage with HIF guidance
        num_envs: Number of parallel environments
        device: Device to run on
        from_pixels: Whether to use pixel observations
        **env_kwargs: Additional environment configuration
    
    Returns:
        TransformedEnv: Environment(s) with transforms applied
    
    Examples:
        # Simple usage with defaults
        env = make_env()
        
        # Specify version
        env = make_env(env_id="NewPasture-v3")
        
        # Custom configuration
        env = make_env(
            env_id="NewPasture-v2",
            use_apf=True,
            num_obstacles_range=(3, 5),
            reward_field_coverage=2.0
        )
        
        # Parallel environments
        env = make_env(num_envs=8, device="cuda:0")
    """
    if num_envs == 1:
        env = make_env_lambda(
            env_id=env_id,
            device=device,
            from_pixels=from_pixels,
            **env_kwargs
        )
    else:
        # Create parallel environments
        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=lambda: make_env_lambda(
                env_id=env_id,
                device=device,
                from_pixels=from_pixels,
                **env_kwargs
            ),
        )
    
    # Apply standard transforms
    env = TransformedEnv(env)
    env.append_transform(InitTracker())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat())
    env.append_transform(RewardSum())

    return env


# ====================================================================
# Convenience functions for common configurations
# --------------------------------------------------------------------

def make_sac_env(env_id: str = "NewPasture-v2", num_envs: int = 1, 
                 device: str = "cpu", **kwargs):
    """
    Create environment for SAC training with continuous action space.
    
    SAC (Soft Actor-Critic) requires continuous action spaces.
    This function automatically sets action_type="continuous".
    
    Args:
        env_id: Environment ID (default: NewPasture-v2 for APF)
        num_envs: Number of parallel environments
        device: Device to run on
        **kwargs: Additional environment parameters
    
    Returns:
        TransformedEnv: Environment configured for SAC training
    """
    # 强制使用连续动作空间
    kwargs['action_type'] = 'continuous'
    return make_env(env_id, num_envs, device, **kwargs)


def make_base_env(num_envs: int = 1, device: str = "cpu", **kwargs):
    """Create base environment (v1) - simple weed coverage."""
    return make_env("NewPasture-v1", num_envs, device, **kwargs)


def make_apf_env(num_envs: int = 1, device: str = "cpu", **kwargs):
    """Create APF environment (v2) - with artificial potential field."""
    return make_env("NewPasture-v2", num_envs, device, use_apf=True, **kwargs)


def make_multiscale_env(num_envs: int = 1, device: str = "cpu", **kwargs):
    """Create multi-scale environment (v3) - with multi-scale features."""
    return make_env("NewPasture-v3", num_envs, device, 
                   use_multiscale=True, use_global_features=True, **kwargs)


def make_field_coverage_env(num_envs: int = 1, device: str = "cpu", **kwargs):
    """Create field coverage environment (v4) - no weed, pure coverage."""
    return make_env("NewPasture-v4", num_envs, device, **kwargs)


def make_hif_env(num_envs: int = 1, device: str = "cpu", **kwargs):
    """Create HIF environment (v5) - field coverage with direction guidance."""
    return make_env("NewPasture-v5", num_envs, device, reward_hif=0.01, **kwargs)


# ====================================================================
# Backward compatibility layer (if needed)
# --------------------------------------------------------------------

# Default environment configuration for backward compatibility
DEFAULT_ENV_CONFIG = {
    "action_type": "continuous",
    "num_obstacles_range": (0, 8),
    "state_size": (128, 128),
    "state_downsize": (128, 128),
    "use_apf": True,
    "use_box_boundary": True,
    "use_trajectory": True,
}

def make_env_from_config(config: Optional[Dict[str, Any]] = None, 
                         num_envs: int = 1,
                         device: str = "cpu",
                         from_pixels: bool = False):
    """
    Create environment from configuration dictionary.
    This function provides backward compatibility for code expecting config-based creation.
    
    Args:
        config: Configuration dictionary (uses defaults if None)
        num_envs: Number of parallel environments
        device: Device to run on
        from_pixels: Whether to use pixel observations
    
    Returns:
        TransformedEnv: Configured environment(s)
    """
    if config is None:
        config = DEFAULT_ENV_CONFIG.copy()
    
    # Extract environment ID if specified, otherwise use v2
    env_id = config.pop("id", "NewPasture-v2")
    
    # Convert old-style ID to new if necessary
    id_mapping = {
        "Pasture-v1": "NewPasture-v1",
        "Pasture-v2": "NewPasture-v2",
        "Pasture-v3": "NewPasture-v3",
        "Pasture-v4": "NewPasture-v4",
        "Pasture-v5": "NewPasture-v5",
    }
    env_id = id_mapping.get(env_id, env_id)
    
    return make_env(
        env_id=env_id,
        num_envs=num_envs,
        device=device,
        from_pixels=from_pixels,
        **config
    )